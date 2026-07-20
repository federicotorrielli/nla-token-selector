"""scripts/token_selector/signals.py — information-theory token selector, controlled test.

Validates the *token selector* from
`findings/2026-06-19_information-theory-token-selection.md`: the function that
scores which token positions in a transcript deserve an (expensive) NLA
explanation. FineWeb cannot test this — its last-10-token anchors are arbitrary
truncation points with no ground truth about which positions matter. Following
the NLA paper's own self-contained validation (the "Reasoning about Rewards"
eval-awareness study: construct inputs more/less likely to be perceived as an
evaluation and check whether a signal varies predictably), we build matched
contrastive PAIRS with a planted cause at a known location and ask whether the
information-theory scores localize it.

Two testbeds, one engine (§5b: same mechanism, swap the reference):
  - injection : ref = same transcript with the injection span replaced by
                neutral filler. Steered tokens = the injection-compliant span.
  - evalaware : ref = the same prompt reframed as ordinary deployment. Decision
                tokens = where eval-awareness surfaces.

Per assistant (teacher-forced) token we compute:
  Tier 0 (free, one forward pass over the full transcript):
    surprisal  S        = -log p(token)
    entropy    H        = E[-log p]            (full vocab)
    varentropy VH       = Var[-log p]          (full vocab)
    override            = z(S) - z(H)          (§3: high-surprisal/low-entropy)
  Tier 1 (contrastive, second pass over the reference):
    dsurprisal          = S_ref - S_full       (§5a)
    kl                  = KL(P_full || P_ref)  (full vocab, §5a)

We then report AUROC of each signal against the planted-span label, pooled over
cases per mode, with a case-cluster bootstrap CI. This tests the pre-registered
hypotheses (note §9): H1 raw surprisal is a weak localizer (attacks are fluent /
low-surprisal); H2 contrastive (kl, dsurprisal) >> raw surprisal; H3 override >
raw surprisal even without a counterfactual; H4 varentropy marks eval decision
forks.

Why transformers here, not sglang (the repo's serving default): the Tier-0/1
signals need the FULL-vocabulary next-token distribution at each position
(entropy, varentropy, and especially KL between two runs). sglang exposes only
top-k logprobs for autoregressive models (`return_full_logits` is hardcoded to
diffusion LMs only; `SamplingParams` has no full-distribution field), so those
quantities would be unreliable top-k approximations. A teacher-forced
transformers forward returns `logits[seq, vocab]` directly. This is the exact
scope justification as `scripts/pipeline/build_corpus.py` — one-shot research code in
`scripts/`, outside the no-HF-in-`src/` rule (`tests/test_no_hf_elsewhere.py`).
When this graduates to real (generated) transcripts, sglang does the generation;
transformers still does the full-vocab scoring.

Usage:
    # offline math + alignment self-test (no GPU, no model, no network):
    python scripts/token_selector/signals.py --selftest

    # real run on the B200 (conda env `pao`, workdir /work/nla_token_selector):
    python scripts/token_selector/signals.py \
        --model Qwen/Qwen2.5-7B-Instruct \
        --cases scripts/token_selector/data/token_selector_cases.json \
        --out results/token_selector
"""

from __future__ import annotations

import argparse
import difflib
import json
import logging
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

log = logging.getLogger("token_selector")

_MARK_RE = re.compile(r"«(.+?)»", re.DOTALL)

# Signals we score. (key, human label, higher-is-more-interesting).
# All are oriented so that a larger value should mean "more worth explaining",
# so AUROC > 0.5 = the signal localizes the planted span.
# Reference-free, one forward pass — usable BLIND (no counterfactual needed).
TIER0_SIGNALS = ("surprisal", "entropy", "varentropy", "override")
# Reference-free state-change (Bayesian/temporal surprise) + attention-ratio
# signals from the literature. All single-pass, usable blind. lookback_ratio =
# attention to context vs self (Lookback Lens, arXiv:2407.07071); attn_distraction
# = drop in attention to the instruction/system span (Attention Tracker,
# arXiv:2411.00348). The two attention signals need eager attention.
REFFREE_SIGNALS = ("temporal_kl", "resid_jump", "lookback_ratio", "attn_distraction")
# Combined attention signals (still blind, same single pass). Both base signals
# are coarse reads of one object: how token t splits its attention among three
# channels — instruction span (I), rest of the context (D), generated so far (G).
#   inj_orient   = z(distraction) - z(lookback): fixed injection-oriented combo
#                  (steered tokens are distracted AND look back less).
#   attn_anomaly = Mahalanobis distance of (lookback, distraction) from the
#                  transcript's own mean — unsigned, so it survives the sign
#                  flip of lookback between threat models.
#   comp_kl      = KL(p_t || p_mean) of the (I, D, G) attention composition vs
#                  the transcript mean composition — the same idea on the full
#                  3-channel simplex.
COMBINED_SIGNALS = ("inj_orient", "attn_anomaly", "comp_kl")
# Blind attention statistics recovered from the follow-up probes (README
# "Combining the signals"). Same single eager-attention pass, no reference.
#   sink_drain        = -(attention onto the template-prefix / sink span before
#                       the system instruction). Steered tokens empty the
#                       attention sink as heads leave the idle position to grab
#                       injected content (Xiao et al. 2023 sink, arXiv:2309.17453).
#   head_disagreement = entropy(head-mean attention) - mean(per-head entropy),
#                       summed over layers. Zero when heads look alike, grows as
#                       they scatter — which is what steered tokens do.
#   w                 = z(sink_drain) - z(lookback_ratio): one signed axis whose
#                       SIGN separates injection (high) from eval awareness (low).
BLIND_ATTN2_SIGNALS = ("sink_drain", "head_disagreement", "w")
# Reference-dependent: need a counterfactual transcript (cause removed/reframed).
TIER1_SIGNALS = ("dsurprisal", "kl")
# Prior-art baseline (note §9): attention rollout onto the (known) cause span.
BASELINE_SIGNALS = ("attn_rollout",)
ALL_SIGNALS = (TIER0_SIGNALS + REFFREE_SIGNALS + COMBINED_SIGNALS
               + BLIND_ATTN2_SIGNALS + TIER1_SIGNALS + BASELINE_SIGNALS)
# Which signals work without knowing where the cause is.
REFERENCE_FREE = (set(TIER0_SIGNALS) | set(REFFREE_SIGNALS)
                  | set(COMBINED_SIGNALS) | set(BLIND_ATTN2_SIGNALS))


# --------------------------------------------------------------------------- #
# Case loading + inline span markers                                          #
# --------------------------------------------------------------------------- #
@dataclass
class Case:
    id: str
    mode: str
    system: str
    user_full: str
    user_ref: str
    assistant: str  # marker-stripped clean text
    label_spans: list[tuple[int, int]] = field(default_factory=list)  # char ranges


def strip_markers(marked: str) -> tuple[str, list[tuple[int, int]]]:
    """Remove «...» markers; return clean text + char ranges of the marked spans
    *in the clean text*. Robust to my hand-counting — the span is whatever was
    inside the guillemets."""
    clean_parts: list[str] = []
    spans: list[tuple[int, int]] = []
    cursor = 0  # length of clean text so far
    pos = 0
    for m in _MARK_RE.finditer(marked):
        before = marked[pos : m.start()]
        clean_parts.append(before)
        cursor += len(before)
        inner = m.group(1)
        spans.append((cursor, cursor + len(inner)))
        clean_parts.append(inner)
        cursor += len(inner)
        pos = m.end()
    clean_parts.append(marked[pos:])
    return "".join(clean_parts), spans


def load_cases(path: Path) -> list[Case]:
    raw = json.loads(path.read_text())
    cases: list[Case] = []
    for c in raw["cases"]:
        clean, spans = strip_markers(c["assistant"])
        cases.append(
            Case(
                id=c["id"],
                mode=c["mode"],
                system=c["system"],
                user_full=c["user_full"],
                user_ref=c["user_ref"],
                assistant=clean,
                label_spans=spans,
            )
        )
    return cases


# --------------------------------------------------------------------------- #
# Information-theory primitives (numpy; unit-tested offline)                   #
# --------------------------------------------------------------------------- #
def _log_softmax(logits: np.ndarray) -> np.ndarray:
    """Stable log-softmax over the last axis."""
    m = logits.max(axis=-1, keepdims=True)
    shifted = logits - m
    lse = np.log(np.exp(shifted).sum(axis=-1, keepdims=True))
    return shifted - lse


def entropy_varentropy(logits_row: np.ndarray) -> tuple[float, float]:
    """Shannon entropy and varentropy (variance of surprisal) of one
    next-token distribution, in nats. H = E[-log p]; VH = Var[-log p]."""
    logp = _log_softmax(logits_row)
    p = np.exp(logp)
    surp = -logp  # self-information per token
    h = float((p * surp).sum())
    vh = float((p * (surp - h) ** 2).sum())
    return h, vh


def _prob_entropy(p: np.ndarray, eps: float = 1e-12) -> float:
    """Shannon entropy (nats) of a probability vector already on the simplex —
    e.g. one attention row (sums to ~1). Zeros contribute nothing, so trailing
    causal-masked keys are harmless."""
    p = p[p > eps]
    return float(-(p * np.log(p)).sum())


def kl_divergence(logits_p: np.ndarray, logits_q: np.ndarray) -> float:
    """KL(P || Q) in nats from two logit rows over the same vocabulary."""
    logp = _log_softmax(logits_p)
    logq = _log_softmax(logits_q)
    p = np.exp(logp)
    return float((p * (logp - logq)).sum())


def _mahalanobis2(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Per-point Mahalanobis distance of (x, y) pairs from their own mean —
    an unsigned within-transcript outlier score on the 2D signal cloud. NaN
    where either input is NaN, or everywhere if fewer than 3 finite points."""
    out = np.full(len(x), np.nan)
    m = ~(np.isnan(x) | np.isnan(y))
    if m.sum() < 3:
        return out
    pts = np.stack([x[m], y[m]], axis=1)
    centered = pts - pts.mean(axis=0)
    cov = (centered.T @ centered) / (len(pts) - 1)
    icov = np.linalg.pinv(cov)
    out[m] = np.sqrt(np.einsum("ij,jk,ik->i", centered, icov, centered))
    return out


def _composition_kl(comp: np.ndarray, eps: float = 1e-9) -> np.ndarray:
    """Per-row KL(p_row || p_mean) over rows of nonnegative channel masses.
    Rows are normalized to the simplex; the reference is the mean composition
    of the finite rows (the transcript's typical attention allocation)."""
    out = np.full(len(comp), np.nan)
    m = ~np.isnan(comp).any(axis=1)
    if not m.any():
        return out
    p = comp[m] + eps
    p = p / p.sum(axis=1, keepdims=True)
    q = p.mean(axis=0)
    out[m] = (p * np.log(p / q)).sum(axis=1)
    return out


# --------------------------------------------------------------------------- #
# AUROC + case-cluster bootstrap (numpy; no sklearn dependency)               #
# --------------------------------------------------------------------------- #
def auroc(scores: np.ndarray, labels: np.ndarray) -> float:
    """AUROC via the Mann-Whitney rank statistic. NaN if a class is absent."""
    scores = np.asarray(scores, dtype=float)
    labels = np.asarray(labels, dtype=int)
    keep = ~np.isnan(scores)  # drop positions with no score (e.g. baseline NaN)
    scores, labels = scores[keep], labels[keep]
    pos = labels == 1
    neg = labels == 0
    n_pos, n_neg = int(pos.sum()), int(neg.sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = np.argsort(scores, kind="mergesort")
    ranks = np.empty(len(scores), dtype=float)
    ranks[order] = np.arange(1, len(scores) + 1)
    # average ties so ties contribute 0.5
    _, inv, counts = np.unique(scores, return_inverse=True, return_counts=True)
    tie_mean = np.zeros(len(counts))
    sums = np.zeros(len(counts))
    np.add.at(sums, inv, ranks)
    tie_mean = sums / counts
    ranks = tie_mean[inv]
    sum_pos = ranks[pos].sum()
    return float((sum_pos - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def bootstrap_auroc(
    per_case: list[tuple[np.ndarray, np.ndarray]],
    n_boot: int = 2000,
    seed: int = 0,
) -> tuple[float, float, float]:
    """Case-cluster bootstrap: resample whole cases (not tokens — tokens within a
    case are correlated). Returns (point, lo95, hi95)."""
    rng = np.random.default_rng(seed)
    all_s = np.concatenate([s for s, _ in per_case]) if per_case else np.array([])
    all_l = np.concatenate([lab for _, lab in per_case]) if per_case else np.array([])
    point = auroc(all_s, all_l)
    n = len(per_case)
    boots: list[float] = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        s = np.concatenate([per_case[i][0] for i in idx])
        lab = np.concatenate([per_case[i][1] for i in idx])
        a = auroc(s, lab)
        if not np.isnan(a):
            boots.append(a)
    if not boots:
        return point, float("nan"), float("nan")
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return point, float(lo), float(hi)


# --------------------------------------------------------------------------- #
# Scoring one case (needs torch + a model; imported lazily)                    #
# --------------------------------------------------------------------------- #
def _build_ids(tokenizer: Any, system: str, user: str) -> list[int]:
    """Chat-template ids up to (and including) the assistant generation prompt.

    Render to string then tokenize: transformers 5.x `apply_chat_template(
    tokenize=True)` returns a BatchEncoding (not a flat id list), and the
    string-render path is robust across families and versions. The template
    emits its own special tokens, so `add_special_tokens=False`.
    """
    msgs = [{"role": "system", "content": system}, {"role": "user", "content": user}]
    rendered = tokenizer.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)
    return tokenizer(rendered, add_special_tokens=False)["input_ids"]


def _assistant_token_labels(
    tokenizer: Any, assistant: str, label_spans: list[tuple[int, int]]
) -> tuple[list[int], list[int]]:
    """Tokenize the assistant continuation; return (token_ids, per-token label)
    where label=1 if a token's char span overlaps any labelled span."""
    enc = tokenizer(assistant, add_special_tokens=False, return_offsets_mapping=True)
    ids = enc["input_ids"]
    offsets = enc["offset_mapping"]
    labels = []
    for s_char, e_char in offsets:
        if e_char <= s_char:  # zero-width (some specials); never positive
            labels.append(0)
            continue
        hit = any(e_char > ls and s_char < le for ls, le in label_spans)
        labels.append(1 if hit else 0)
    return ids, labels


def _cause_token_indices(pre_full: list[int], pre_ref: list[int]) -> list[int]:
    """Token positions in `pre_full` that are the planted cause — i.e. present in
    the full prefix but absent from the reference prefix (the injection span /
    the eval framing). Found by diffing the two tokenized prefixes. Empty for
    negative controls whose `user_ref` only *adds* text."""
    sm = difflib.SequenceMatcher(a=pre_ref, b=pre_full, autojunk=False)
    idx: list[int] = []
    for tag, _i1, _i2, j1, j2 in sm.get_opcodes():
        if tag in ("replace", "insert"):
            idx.extend(range(j1, j2))
    return idx


def _instruction_token_indices(tokenizer: Any, system: str, user: str) -> list[int]:
    """Positions of the system/instruction tokens inside the templated prefix —
    the span the Attention Tracker watches. Found by diffing 'build with system'
    vs 'build without system', so it is family-agnostic: it works even on Gemma,
    whose template folds the system text into the user turn (no separate system
    turn, so a system-only render would be empty)."""
    with_sys = _build_ids(tokenizer, system, user)
    no_sys = _build_ids(tokenizer, "", user)
    sm = difflib.SequenceMatcher(a=no_sys, b=with_sys, autojunk=False)
    idx: list[int] = []
    for tag, _i1, _i2, j1, j2 in sm.get_opcodes():
        if tag in ("replace", "insert"):
            idx.extend(range(j1, j2))
    return idx


def attention_rollout(attentions: list[np.ndarray]) -> np.ndarray:
    """Abnar & Zuidema (2020) attention rollout. `attentions` is a per-layer list
    of [heads, seq, seq] matrices. Average heads, add the residual (identity),
    row-normalize, then multiply across layers. Returns [seq, seq]: entry [i, j]
    is the rolled-out attention from query token i to key token j."""
    seq = attentions[0].shape[-1]
    rollout = np.eye(seq)
    eye = np.eye(seq)
    for a in attentions:
        head_avg = a.mean(axis=0)  # [seq, seq]
        aug = head_avg + eye
        aug = aug / aug.sum(axis=-1, keepdims=True)
        rollout = aug @ rollout
    return rollout


def score_case(
    case: Case, model: Any, tokenizer: Any, device: str, want_attn: bool = True
) -> dict[str, np.ndarray]:
    """Forward both transcripts; return per-assistant-token signal arrays + label.

    The assistant continuation is identical in both runs, so its tokens align
    one-for-one and the last len(asst) positions of each run correspond. The
    prediction for assistant token j comes from the logits at the position
    *before* it.
    """
    import torch

    asst_ids, labels = _assistant_token_labels(tokenizer, case.assistant, case.label_spans)
    n_asst = len(asst_ids)
    assert n_asst > 0, f"case {case.id}: empty assistant"

    pre_full = _build_ids(tokenizer, case.system, case.user_full)
    pre_ref = _build_ids(tokenizer, case.system, case.user_ref)
    full_ids = pre_full + asst_ids
    ref_ids = pre_ref + asst_ids
    a_full = len(pre_full)  # index of first assistant token in full_ids
    a_ref = len(pre_ref)

    def forward(ids: list[int], with_attn: bool = False, with_hidden: bool = False):
        t = torch.tensor([ids], device=device)
        with torch.no_grad():
            out = model(t, output_attentions=with_attn, output_hidden_states=with_hidden)
        logits = out.logits[0].float().cpu().numpy()  # [seq, vocab]
        attns = None
        if with_attn and out.attentions is not None:
            attns = [a[0].float().cpu().numpy() for a in out.attentions]  # per layer [h,s,s]
        hidden = out.hidden_states[-1][0].float().cpu().numpy() if with_hidden else None  # [seq,d]
        return logits, attns, hidden

    lg_full, attns_full, hs_full = forward(full_ids, with_attn=want_attn, with_hidden=True)
    lg_ref, _, _ = forward(ref_ids)

    surp = np.empty(n_asst)
    ent = np.empty(n_asst)
    varent = np.empty(n_asst)
    dsurp = np.empty(n_asst)
    kl = np.empty(n_asst)
    temporal_kl = np.empty(n_asst)  # reference-free: KL(pred at tok j || pred at tok j-1)
    resid_jump = np.empty(n_asst)   # reference-free: ||h_t - h_{t-1}|| (state change)
    for j in range(n_asst):
        tok = asst_ids[j]
        p = a_full + j
        row_f = lg_full[p - 1]
        row_r = lg_ref[a_ref + j - 1]
        logp_f = _log_softmax(row_f)
        logp_r = _log_softmax(row_r)
        surp[j] = -logp_f[tok]
        ent[j], varent[j] = entropy_varentropy(row_f)
        dsurp[j] = (-logp_r[tok]) - (-logp_f[tok])
        kl[j] = kl_divergence(row_f, row_r)
        temporal_kl[j] = kl_divergence(row_f, lg_full[p - 2])
        resid_jump[j] = float(np.linalg.norm(hs_full[p] - hs_full[p - 1]))

    # §3 override = high surprisal at low entropy. Continuous, per-case z-scored
    # (the quadrant relative to *this* transcript's own positions).
    def z(a: np.ndarray) -> np.ndarray:
        sd = a.std()
        return (a - a.mean()) / sd if sd > 1e-9 else np.zeros_like(a)

    override = z(surp) - z(ent)

    # Baseline: rolled-out attention from each assistant token onto the planted
    # cause span in the input. NaN if attentions unavailable (e.g. sdpa) or the
    # cause span is empty (negative controls).
    attn_roll = np.full(n_asst, np.nan)
    cause_idx = _cause_token_indices(pre_full, pre_ref)
    if attns_full is not None and cause_idx:
        roll = attention_rollout(attns_full)  # [seq, seq]
        for j in range(n_asst):
            attn_roll[j] = roll[a_full + j, cause_idx].sum()

    # Reference-free attention-ratio signals (single pass, from the same attns).
    #   lookback_ratio  = attention onto context (prompt) / (context + generated)
    #                     (Lookback Lens). High = grounded in the prompt.
    #   attn_distraction = -(attention onto the system/instruction span)
    #                     (Attention Tracker). High = distracted away from the
    #                     original instruction, the injection signature.
    # ponytail: head-mean over all heads; important-head selection is the upgrade
    # path if the all-heads signal is weak.
    lookback = np.full(n_asst, np.nan)
    distraction = np.full(n_asst, np.nan)
    chan = np.full((n_asst, 3), np.nan)  # per-token (I, D, G) attention masses
    snk_mass = np.full(n_asst, np.nan)   # attention onto the sink/template prefix
    hd_arr = np.full(n_asst, np.nan)     # head_disagreement, summed over layers
    instr_idx: list[int] = []
    if attns_full is not None:
        instr_idx = [i for i in _instruction_token_indices(tokenizer, case.system, case.user_full)
                     if i < a_full]
        # Sink span = the scaffolding before the system instruction (BOS + role
        # header), where idle attention parks. Fall back to the first few tokens.
        # ponytail: all layers — on Gemma past 1024 tokens the sliding-window
        # layers can't see token 0, so restrict to global-attn layers there; our
        # transcripts are short, so every layer sees the sink.
        sink_idx = list(range(min(instr_idx))) if instr_idx else list(range(min(4, a_full)))
        n_layers = len(attns_full)
        for j in range(n_asst):
            p = a_full + j
            ctx = gen = sysm = snk = 0.0
            hd = 0.0
            for A in attns_full:  # A: [heads, seq, seq]
                rows = A[:, p, :]            # [heads, seq] attention from token p
                hm = rows.mean(axis=0)       # head-mean attention
                ctx += hm[:a_full].sum()
                gen += hm[a_full:p].sum()
                if instr_idx:
                    sysm += hm[instr_idx].sum()
                if sink_idx:
                    snk += hm[sink_idx].sum()
                # head_disagreement: entropy of the head-mean minus the mean of
                # the heads' own entropies (a Jensen-Shannon gap, always >= 0).
                hd += _prob_entropy(hm) - np.mean([_prob_entropy(rows[h])
                                                   for h in range(rows.shape[0])])
            denom = ctx + gen
            lookback[j] = ctx / denom if denom > 1e-9 else np.nan
            distraction[j] = -(sysm / n_layers) if instr_idx else np.nan
            hd_arr[j] = hd
            if sink_idx:
                snk_mass[j] = snk
            if instr_idx:
                chan[j] = (sysm, ctx - sysm, gen)

    # Combined blind signals (see COMBINED_SIGNALS): fixed injection-oriented
    # combo, unsigned 2D outlier score, and composition divergence over (I,D,G).
    inj_orient = np.full(n_asst, np.nan)
    attn_anomaly = np.full(n_asst, np.nan)
    comp_kl = np.full(n_asst, np.nan)
    if attns_full is not None and instr_idx:
        inj_orient = z(distraction) - z(lookback)
        attn_anomaly = _mahalanobis2(lookback, distraction)
        comp_kl = _composition_kl(chan)

    # Recovered blind attention signals (see BLIND_ATTN2_SIGNALS).
    sink_drain = np.full(n_asst, np.nan)
    head_disagreement = np.full(n_asst, np.nan)
    w = np.full(n_asst, np.nan)
    if attns_full is not None:
        head_disagreement = hd_arr
        if not np.isnan(snk_mass).all():
            sink_drain = -(snk_mass / n_layers)  # negate: draining = interesting
            w = z(sink_drain) - z(lookback)

    return {
        "surprisal": surp,
        "entropy": -ent,  # orient: low entropy = more interesting -> negate
        "varentropy": varent,
        "override": override,
        "temporal_kl": temporal_kl,
        "resid_jump": resid_jump,
        "lookback_ratio": lookback,
        "attn_distraction": distraction,
        "inj_orient": inj_orient,
        "attn_anomaly": attn_anomaly,
        "comp_kl": comp_kl,
        "sink_drain": sink_drain,
        "head_disagreement": head_disagreement,
        "w": w,
        "dsurprisal": dsurp,
        "kl": kl,
        "attn_rollout": attn_roll,
        "label": np.array(labels, dtype=int),
        "tokens": np.array(asst_ids),
    }


# --------------------------------------------------------------------------- #
# Reporting                                                                    #
# --------------------------------------------------------------------------- #
def summarize(
    scored: list[tuple[Case, dict[str, np.ndarray]]], mode: str, n_boot: int
) -> dict[str, Any]:
    rows = [(c, d) for c, d in scored if c.mode == mode]
    n_pos = int(sum(int(d["label"].sum()) for _, d in rows))
    n_tok = int(sum(len(d["label"]) for _, d in rows))
    out: dict[str, Any] = {
        "mode": mode,
        "n_cases": len(rows),
        "n_tokens": n_tok,
        "n_positive": n_pos,
        "signals": {},
    }
    for sig in ALL_SIGNALS:
        per_case = [(d[sig], d["label"]) for _, d in rows]
        point, lo, hi = bootstrap_auroc(per_case, n_boot=n_boot)
        out["signals"][sig] = {"auroc": point, "lo95": lo, "hi95": hi}
    return out


def format_report(summaries: list[dict[str, Any]]) -> str:
    lines: list[str] = []
    for s in summaries:
        lines.append("")
        lines.append(
            f"### mode={s['mode']}  cases={s['n_cases']}  "
            f"tokens={s['n_tokens']}  positives={s['n_positive']}"
        )
        lines.append(f"{'signal':<12} {'needs ref?':<10} {'AUROC':>7}  95% CI")
        lines.append("-" * 48)
        for sig in ALL_SIGNALS:
            v = s["signals"][sig]
            tier = "blind" if sig in REFERENCE_FREE else "needs-ref"
            ci = (
                f"[{v['lo95']:.3f}, {v['hi95']:.3f}]"
                if not np.isnan(v["lo95"])
                else "[n/a]"
            )
            lines.append(f"{sig:<12} {tier:<10} {v['auroc']:>7.3f}  {ci}")
        # Pre-registered hypothesis read-outs.
        sg = s["signals"]
        raw = sg["surprisal"]["auroc"]
        kl = sg["kl"]["auroc"]
        ov = sg["override"]["auroc"]
        lines.append("")
        lines.append(f"  H1 raw surprisal weak?      surprisal AUROC = {raw:.3f}")
        lines.append(
            f"  H2 contrastive >> raw?      kl {kl:.3f} vs surprisal {raw:.3f}  "
            f"(Δ {kl - raw:+.3f})"
        )
        lines.append(
            f"  H3 override > raw?          override {ov:.3f} vs surprisal {raw:.3f}  "
            f"(Δ {ov - raw:+.3f})"
        )
        if s["mode"] == "evalaware":
            vh = sg["varentropy"]["auroc"]
            lines.append(f"  H4 varentropy marks forks?  varentropy AUROC = {vh:.3f}")
        ar = sg["attn_rollout"]["auroc"]
        if not np.isnan(ar):
            lines.append(
                f"  H5 KL beats prior art?      kl {kl:.3f} vs attn_rollout {ar:.3f}  "
                f"(Δ {kl - ar:+.3f})"
            )
        # Blind setting: best reference-free signal vs referenced KL.
        rf = {k: sg[k]["auroc"] for k in REFERENCE_FREE if not np.isnan(sg[k]["auroc"])}
        best_rf = max(rf, key=rf.get)
        lines.append(
            f"  H6 blind detection works?   best reference-free = {best_rf} {rf[best_rf]:.3f}  "
            f"(vs referenced kl {kl:.3f})"
        )
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# Self-test (offline; no torch/model/network needed for the math)             #
# --------------------------------------------------------------------------- #
def selftest() -> int:
    ok = True

    def check(name: str, cond: bool) -> None:
        nonlocal ok
        ok = ok and cond
        print(f"  [{'PASS' if cond else 'FAIL'}] {name}")

    # entropy of uniform = log V; of a delta = 0
    V = 1000
    uniform = np.zeros(V)
    h, vh = entropy_varentropy(uniform)
    check("entropy(uniform)=logV", abs(h - np.log(V)) < 1e-6)
    check("varentropy(uniform)=0", abs(vh) < 1e-9)
    delta = np.full(V, -50.0)
    delta[3] = 50.0
    h2, vh2 = entropy_varentropy(delta)
    check("entropy(delta)~0", h2 < 1e-3)

    # varentropy: two-peak fork >> flat shrug at equal entropy-ish
    fork = np.full(V, -50.0)
    fork[0] = fork[1] = 0.0  # two sharp peaks -> high varentropy
    _, vh_fork = entropy_varentropy(fork)
    shrug = np.zeros(V)  # flat -> zero varentropy
    _, vh_shrug = entropy_varentropy(shrug)
    check("varentropy(fork) > varentropy(shrug)", vh_fork > vh_shrug)

    # KL(p||p)=0; KL(peaked||uniform)>0
    check("KL(p||p)=0", abs(kl_divergence(delta, delta)) < 1e-9)
    check("KL(peaked||uniform)>0", kl_divergence(delta, uniform) > 1.0)

    # AUROC: perfect, anti, chance, ties
    check("auroc perfect=1", abs(auroc([3, 2, 1, 0], [1, 1, 0, 0]) - 1.0) < 1e-9)
    check("auroc anti=0", abs(auroc([0, 1, 2, 3], [1, 1, 0, 0]) - 0.0) < 1e-9)
    check("auroc all-ties=0.5", abs(auroc([1, 1, 1, 1], [1, 0, 1, 0]) - 0.5) < 1e-9)
    check("auroc one-class=nan", np.isnan(auroc([1, 2, 3], [1, 1, 1])))

    # bootstrap returns a CI bracketing the point for a clean signal
    rng = np.random.default_rng(0)
    per_case = []
    for _ in range(8):
        s = np.concatenate([rng.normal(1, 0.3, 5), rng.normal(0, 0.3, 5)])
        lab = np.array([1] * 5 + [0] * 5)
        per_case.append((s, lab))
    pt, lo, hi = bootstrap_auroc(per_case, n_boot=500)
    check("bootstrap point in CI", lo <= pt <= hi)
    check("bootstrap clean signal high", pt > 0.8)

    # marker stripping + char spans
    clean, spans = strip_markers("hello «world» foo «bar baz» end")
    check("strip removes markers", clean == "hello world foo bar baz end")
    check("span 1 maps to 'world'", clean[spans[0][0] : spans[0][1]] == "world")
    check("span 2 maps to 'bar baz'", clean[spans[1][0] : spans[1][1]] == "bar baz")
    clean2, spans2 = strip_markers("no markers here")
    check("no markers -> no spans", clean2 == "no markers here" and spans2 == [])

    # offset->label overlap logic (tokenizer-free): fake offsets over "ABCDEF",
    # label span covers chars [2,4); tokens [0,2),[2,4),[4,6) -> labels 0,1,0
    offsets = [(0, 2), (2, 4), (4, 6)]
    label_spans = [(2, 4)]
    labels = [1 if any(e > ls and s < le for ls, le in label_spans) else 0 for s, e in offsets]
    check("offset overlap labels = [0,1,0]", labels == [0, 1, 0])

    # cause-span diff: full=[A,B,X,Y,C], ref=[A,B,C] -> cause = positions of X,Y
    check("cause diff finds inserted span", _cause_token_indices([1, 2, 9, 8, 3], [1, 2, 3]) == [2, 3])
    check("cause diff empty when ref superset", _cause_token_indices([1, 2, 3], [1, 2, 3, 4]) == [])

    # attention rollout: NaN-free, rows sum ~1 (it's a product of row-stochastic
    # matrices), and a layer attending only to token 0 routes mass to token 0.
    rng2 = np.random.default_rng(1)
    attns = [rng2.random((2, 4, 4)) for _ in range(3)]  # 3 layers, 2 heads, 4x4
    roll = attention_rollout(attns)
    check("rollout rows sum to 1", np.allclose(roll.sum(axis=-1), 1.0))
    check("rollout non-negative", (roll >= 0).all())
    focus = [np.zeros((1, 3, 3)) for _ in range(2)]
    for f in focus:
        f[0, :, 0] = 1.0  # every query attends only to key 0
    rfoc = attention_rollout(focus)
    check("rollout concentrates on attended token", rfoc[:, 0].mean() > rfoc[:, 1].mean())

    # combined signals: Mahalanobis flags a planted 2D outlier and needs >=3
    # points; composition KL is ~0 for constant rows and flags the odd row.
    # NB: the outlier must be rare relative to the cloud — with very few points
    # it drags the covariance toward itself and gets masked (verified: 1-in-5
    # ties with inliers, 1-in-21 is cleanly flagged).
    rng3 = np.random.default_rng(2)
    xs = np.concatenate([rng3.normal(0, 0.1, 20), [3.0]])
    ys = np.concatenate([rng3.normal(0, 0.1, 20), [-3.0]])
    check("mahalanobis flags the outlier", int(np.nanargmax(_mahalanobis2(xs, ys))) == 20)
    check("mahalanobis too-few-points -> NaN",
          np.isnan(_mahalanobis2(np.array([np.nan, 1.0]), np.array([1.0, 2.0]))).all())
    const = np.tile([0.2, 0.3, 0.5], (4, 1))
    check("composition KL of constant rows ~0", float(np.nanmax(_composition_kl(const))) < 1e-6)
    mixed = np.array([[0.2, 0.3, 0.5], [0.2, 0.3, 0.5], [0.2, 0.3, 0.5], [0.9, 0.05, 0.05]])
    check("composition KL flags the odd row", int(np.argmax(_composition_kl(mixed))) == 3)
    check("combined signals are reference-free", set(COMBINED_SIGNALS) <= REFERENCE_FREE)

    # head_disagreement: entropy(head-mean) - mean(head entropies). Zero when the
    # heads look alike, positive when they scatter; never negative (JS gap).
    agree = np.array([[0.5, 0.5], [0.5, 0.5]])  # two heads, identical rows
    hd_agree = _prob_entropy(agree.mean(0)) - np.mean([_prob_entropy(agree[h]) for h in range(2)])
    check("head_disagreement ~0 when heads agree", abs(hd_agree) < 1e-9)
    scatter = np.array([[1.0, 0.0], [0.0, 1.0]])  # heads point opposite ways
    hd_scatter = _prob_entropy(scatter.mean(0)) - np.mean([_prob_entropy(scatter[h]) for h in range(2)])
    check("head_disagreement > 0 when heads scatter", hd_scatter > 0.6)
    check("recovered blind signals are reference-free", set(BLIND_ATTN2_SIGNALS) <= REFERENCE_FREE)

    # reference-free signals: resid_jump of identical states is 0; the reference
    # set excludes the counterfactual signals.
    check("resid jump of identical states = 0", np.linalg.norm(np.ones(8) - np.ones(8)) == 0.0)
    check("reffree set includes state-change + attention-ratio signals",
          {"temporal_kl", "resid_jump", "lookback_ratio", "attn_distraction"} <= REFERENCE_FREE)
    check("kl/attn_rollout are NOT reference-free",
          not ({"kl", "attn_rollout"} & REFERENCE_FREE))

    # cases file loads and every labelled span lands inside the clean assistant
    cases_path = Path(__file__).parent / "data" / "token_selector_cases.json"
    if cases_path.exists():
        cases = load_cases(cases_path)
        modes = {c.mode for c in cases}
        check("cases load, both modes present", {"injection", "evalaware"} <= modes)
        spans_ok = all(
            0 <= ls < le <= len(c.assistant) for c in cases for ls, le in c.label_spans
        )
        check("all label spans in range", spans_ok)
        has_neg_ctrl = any(not c.label_spans for c in cases)
        check("at least one negative control", has_neg_ctrl)

    print(f"\nselftest: {'ALL PASS' if ok else 'FAILURES'}")
    return 0 if ok else 1


# --------------------------------------------------------------------------- #
# Main                                                                         #
# --------------------------------------------------------------------------- #
def run(args: argparse.Namespace) -> int:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    cases = load_cases(Path(args.cases))
    log.info("loaded %d cases (%d injection, %d evalaware)", len(cases),
             sum(c.mode == "injection" for c in cases),
             sum(c.mode == "evalaware" for c in cases))

    model_tag = args.model.split("/")[-1]
    out_dir = Path(args.out) / model_tag
    out_dir.mkdir(parents=True, exist_ok=True)

    # The attention-rollout baseline needs attention weights, which sdpa does not
    # return; force eager when it is enabled. eager is safe on B200 (the broken
    # path is xformers, not eager) and our sequences are short.
    want_attn = not args.no_attn_baseline
    attn_impl = "eager" if want_attn else args.attn
    if want_attn and args.attn != "eager":
        log.info("attn-rollout baseline on -> forcing attn_implementation=eager")

    log.info("loading %s (dtype=%s, attn=%s)", args.model, args.dtype, attn_impl)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16}[args.dtype]
    load_kw = dict(device_map="auto", torch_dtype=dtype, attn_implementation=attn_impl)
    try:
        model = AutoModelForCausalLM.from_pretrained(args.model, **load_kw).eval()
    except (ValueError, KeyError):
        # Gemma-3 is a Gemma3ForConditionalGeneration (multimodal) checkpoint;
        # the text path still runs from input_ids alone.
        from transformers import AutoModelForImageTextToText
        log.info("falling back to AutoModelForImageTextToText (multimodal checkpoint)")
        model = AutoModelForImageTextToText.from_pretrained(args.model, **load_kw).eval()
    device = str(model.get_input_embeddings().weight.device)

    scored: list[tuple[Case, dict[str, np.ndarray]]] = []
    for c in cases:
        d = score_case(c, model, tokenizer, device, want_attn=want_attn)
        scored.append((c, d))
        log.info("scored %-26s tokens=%-3d positives=%d", c.id, len(d["label"]),
                 int(d["label"].sum()))

    # per-token parquet (for later inspection / the qualitative spot-check)
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq

        rows = []
        for c, d in scored:
            toks = tokenizer.convert_ids_to_tokens(d["tokens"].tolist())
            for j in range(len(d["label"])):
                rows.append({
                    "case_id": c.id, "mode": c.mode, "tok_idx": j,
                    "token": toks[j], "label": int(d["label"][j]),
                    **{sig: float(d[sig][j]) for sig in ALL_SIGNALS},
                })
        pq.write_table(pa.Table.from_pylist(rows), out_dir / "tokens.parquet")
        log.info("wrote %s", out_dir / "tokens.parquet")
    except Exception as e:  # parquet is a convenience, not the result
        log.warning("parquet write skipped: %s", e)

    summaries = [summarize(scored, m, args.n_boot) for m in ("injection", "evalaware")]
    report = f"# token selector test — {args.model}\n" + format_report(summaries)
    (out_dir / "summary.json").write_text(json.dumps(summaries, indent=2))
    (out_dir / "report.md").write_text(report)
    print(report)
    log.info("wrote %s and summary.json", out_dir / "report.md")

    # qualitative spot-check: top-5 KL tokens per mode (memory: eyeball samples)
    for mode in ("injection", "evalaware"):
        rows = [(c, d) for c, d in scored if c.mode == mode]
        triples = []
        for c, d in rows:
            toks = tokenizer.convert_ids_to_tokens(d["tokens"].tolist())
            for j in range(len(d["label"])):
                triples.append((float(d["kl"][j]), c.id, toks[j], int(d["label"][j])))
        triples.sort(reverse=True)
        print(f"\n[spot-check] top-5 KL tokens, mode={mode} (label in brackets):")
        for klv, cid, tok, lab in triples[:5]:
            print(f"    kl={klv:6.2f}  [{lab}]  {cid:<22} {tok!r}")

    return 0


def _parse_args(argv: list[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--selftest", action="store_true", help="offline math/alignment test")
    p.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    p.add_argument("--cases", default=str(Path(__file__).parent / "data" / "token_selector_cases.json"))
    p.add_argument("--out", default="results/token_selector")
    p.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16"])
    p.add_argument("--attn", default="sdpa",
                   help="attn_implementation when the attn-rollout baseline is off "
                        "(sdpa is safe on B200; baseline forces eager)")
    p.add_argument("--no-attn-baseline", action="store_true",
                   help="skip the attention-rollout baseline (keeps attn=sdpa, faster)")
    p.add_argument("--n-boot", type=int, default=2000)
    p.add_argument("--log-level", default="INFO")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv if argv is not None else sys.argv[1:])
    logging.basicConfig(level=args.log_level.upper(),
                        format="%(asctime)s %(levelname)s %(name)s %(message)s")
    if args.selftest:
        return selftest()
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
