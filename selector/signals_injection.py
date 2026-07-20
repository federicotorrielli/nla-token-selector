"""scripts/token_selector/signals_injection.py — input-span localization.

The PromptLocate-comparable injection benchmark (the headline). Given an injected
data prompt, localize the *injected span in the input* — the task TracLLM (Wang
et al., USENIX Security 2025) and PromptLocate (Jia et al. 2025) define. Cases
come from build_injection_cases.py (OpenPromptInjection data, free span labels).

This is a different axis from signals.py, which scores the assistant
*continuation*. Here we score INPUT tokens of the data prompt; the label is the
injected span. Our continuation winners do not transfer: contrastive KL has no
per-injected-token counterfactual (the injected tokens are absent from the clean
prompt). So three input-token signal families:

  Cheap, one forward pass over the prompt (the fast pre-NLA probe):
    in_surprisal  -log p(token)        reading surprise as the model ingests it
    in_entropy    H of the next-token dist (negated: low H = more interesting)
    in_attention  rolled-out attention the generation cue sends back to each
                  input token (Abnar-Zuidema rollout, read per input token)
  Strong baseline, O(n_segments) forwards (attribution, not a probe):
    looo          leave-one-segment-out: drop in log p(model's own answer) when a
                  clause is removed. The injected clause is the one the compromised
                  answer most depends on (TracLLM-style perturbation attribution).

Scored two ways: per-token AUROC vs the injected-span label (our internal axis),
and PromptLocate's own axis — word-level Precision/Recall + ROUGE-L between the
predicted injected text (tokens/segments above a threshold) and the true injected
text. Both reported, pooled over cases, case-cluster bootstrap CI on AUROC.

transformers, not sglang, for the same reason as signals.py: we need
full-vocab next-token distributions and teacher-forced answer log-probs. The looo
baseline does one short greedy generation per case, then scores it under ablated
prompts (no generation in the inner loop).

Usage:
    python scripts/token_selector/signals_injection.py --selftest
    python scripts/token_selector/signals_injection.py \
        --model Qwen/Qwen2.5-7B-Instruct \
        --cases scripts/token_selector/data/injection_cases.jsonl \
        --out results/injection_localization --max-cases 200
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

# Reuse the vetted primitives from the sibling selector test (same directory).
sys.path.insert(0, str(Path(__file__).resolve().parent))
from signals import (  # noqa: E402
    _build_ids,
    _log_softmax,
    attention_rollout,
    auroc,
    bootstrap_auroc,
    entropy_varentropy,
)

log = logging.getLogger("injection_localization")

CHEAP_SIGNALS = ("in_surprisal", "in_entropy", "in_attention")
BASELINE_SIGNALS = ("looo",)
ALL_SIGNALS = CHEAP_SIGNALS + BASELINE_SIGNALS


# --------------------------------------------------------------------------- #
# Cases                                                                        #
# --------------------------------------------------------------------------- #
@dataclass
class InjCase:
    id: str
    system: str
    user_full: str
    user_ref: str
    inj_start: int
    inj_end: int
    injected_text: str


def load_cases(path: Path, max_cases: int | None = None) -> list[InjCase]:
    cases: list[InjCase] = []
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        cases.append(InjCase(r["id"], r["system"], r["user_full"], r["user_ref"],
                             r["inj_start"], r["inj_end"], r["injected_text"]))
        if max_cases and len(cases) >= max_cases:
            break
    return cases


# --------------------------------------------------------------------------- #
# Segmentation + localization metrics (dep-free; unit-tested offline)          #
# --------------------------------------------------------------------------- #
def segment_spans(text: str) -> list[tuple[int, int]]:
    """Split into clause/sentence segments at . ! ? and newlines, returning char
    ranges that tile the whole string. Mirrors PromptLocate's regex segmenter
    (no spaCy dependency). Boundaries are kept with the preceding segment."""
    spans: list[tuple[int, int]] = []
    start = 0
    for i, ch in enumerate(text):
        if ch in ".!?\n":
            spans.append((start, i + 1))
            start = i + 1
    if start < len(text):
        spans.append((start, len(text)))
    return [(s, e) for s, e in spans if e > s]


def word_prf(pred: str, gold: str) -> tuple[float, float, float]:
    """Bag-of-words precision/recall/F1 between predicted and gold injected text.
    Multiset intersection (so repeats count), lowercased, whitespace-tokenized."""
    from collections import Counter

    p = Counter(pred.lower().split())
    g = Counter(gold.lower().split())
    overlap = sum((p & g).values())
    prec = overlap / max(sum(p.values()), 1)
    rec = overlap / max(sum(g.values()), 1)
    f1 = 2 * prec * rec / (prec + rec) if prec + rec > 0 else 0.0
    return prec, rec, f1


def rouge_l(pred: str, gold: str) -> float:
    """ROUGE-L F1 (LCS over word sequences) — PromptLocate's recovery metric."""
    a, b = pred.lower().split(), gold.lower().split()
    if not a or not b:
        return 0.0
    # LCS length via rolling DP.
    prev = [0] * (len(b) + 1)
    for x in a:
        cur = [0] * (len(b) + 1)
        for j, y in enumerate(b, 1):
            cur[j] = prev[j - 1] + 1 if x == y else max(prev[j], cur[j - 1])
        prev = cur
    lcs = prev[-1]
    prec, rec = lcs / len(a), lcs / len(b)
    return 2 * prec * rec / (prec + rec) if prec + rec > 0 else 0.0


# --------------------------------------------------------------------------- #
# Scoring one case (needs torch + model; imported lazily)                      #
# --------------------------------------------------------------------------- #
def _user_token_view(tokenizer: Any, system: str, user: str):
    """Tokenize the templated prompt; return (ids, list of (tok_idx, char_start,
    char_end)) for the tokens that fall inside the `user` content. Char offsets
    are in `user`-local coordinates. Found by locating `user` in the rendered
    template and offset-mapping."""
    msgs = [{"role": "system", "content": system}, {"role": "user", "content": user}]
    rendered = tokenizer.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)
    u0 = rendered.rfind(user)
    if u0 < 0:
        # Some templates (Gemma) trim message content, so leading/trailing
        # whitespace of `user` is dropped from the render. Match the stripped
        # text and shift back to `user`-local coordinates by the leading run.
        stripped = user.strip()
        pos = rendered.rfind(stripped)
        assert pos >= 0, "user content not found verbatim in rendered template"
        u0 = pos - (len(user) - len(user.lstrip()))
    enc = tokenizer(rendered, add_special_tokens=False, return_offsets_mapping=True)
    ids = enc["input_ids"]
    view = []
    for ti, (s, e) in enumerate(enc["offset_mapping"]):
        if e <= s:
            continue
        if s >= u0 and e <= u0 + len(user):  # token lies within user content
            view.append((ti, s - u0, e - u0))
    return ids, view


def _labels_for_view(view, inj_start: int, inj_end: int) -> np.ndarray:
    """1 if a viewed token's user-local char span overlaps the injected span."""
    return np.array(
        [1 if (ce > inj_start and cs < inj_end) else 0 for _ti, cs, ce in view],
        dtype=int,
    )


def score_case(case: InjCase, model: Any, tokenizer: Any, device: str,
               do_looo: bool = True) -> dict[str, np.ndarray]:
    import torch

    ids, view = _user_token_view(tokenizer, case.system, case.user_full)
    n = len(view)
    assert n > 0, f"case {case.id}: no user tokens in view"
    labels = _labels_for_view(view, case.inj_start, case.inj_end)

    t = torch.tensor([ids], device=device)
    with torch.no_grad():
        out = model(t, output_attentions=True)
    logits = out.logits[0].float().cpu().numpy()
    attns = [a[0].float().cpu().numpy() for a in out.attentions] if out.attentions else None

    in_surp = np.empty(n)
    in_ent = np.empty(n)
    for k, (ti, _cs, _ce) in enumerate(view):
        row = logits[ti - 1]  # dist that predicted this token
        lp = _log_softmax(row)
        in_surp[k] = -lp[ids[ti]]
        h, _ = entropy_varentropy(row)
        in_ent[k] = h

    # Incoming attention: rolled-out attention the final (generation-cue) position
    # sends back to each input token. High = the model leans on this token.
    in_attn = np.full(n, np.nan)
    if attns is not None:
        roll = attention_rollout(attns)  # [seq, seq]
        q = len(ids) - 1
        for k, (ti, _cs, _ce) in enumerate(view):
            in_attn[k] = roll[q, ti]

    looo = np.full(n, np.nan)
    if do_looo:
        looo = _leave_segment_out(case, model, tokenizer, device, view)

    return {
        "in_surprisal": in_surp,
        "in_entropy": -in_ent,  # low entropy = more interesting -> negate
        "in_attention": in_attn,
        "looo": looo,
        "label": labels,
        "view": view,
        "ids": np.array(ids),
    }


def _leave_segment_out(case: InjCase, model: Any, tokenizer: Any, device: str,
                       view) -> np.ndarray:
    """TracLLM-style perturbation attribution. Greedy-generate the model's answer
    to the full prompt, then for each clause segment of user_full, measure the
    drop in log p(answer) when that segment is removed. The injected clause is the
    one the (compromised) answer most depends on. Score broadcast to its tokens."""

    n = len(view)
    out = np.zeros(n)
    answer = _greedy_answer(case.system, case.user_full, model, tokenizer, device)
    if not answer.strip():
        return np.full(n, np.nan)

    base_lp = _answer_logprob(case.system, case.user_full, answer, model, tokenizer, device)
    segs = segment_spans(case.user_full)
    for s, e in segs:
        ablated = (case.user_full[:s] + case.user_full[e:]).strip()
        if not ablated:
            continue
        lp = _answer_logprob(case.system, ablated, answer, model, tokenizer, device)
        delta = base_lp - lp  # how much the answer needed this segment
        for k, (_ti, cs, ce) in enumerate(view):
            if ce > s and cs < e:  # token in this segment
                out[k] = delta
    return out


def _greedy_answer(system: str, user: str, model, tokenizer, device, max_new: int = 32) -> str:
    import torch

    ids = _build_ids(tokenizer, system, user)
    t = torch.tensor([ids], device=device)
    with torch.no_grad():
        gen = model.generate(t, max_new_tokens=max_new, do_sample=False,
                             pad_token_id=tokenizer.eos_token_id)
    return tokenizer.decode(gen[0][len(ids):], skip_special_tokens=True)


def _answer_logprob(system: str, user: str, answer: str, model, tokenizer, device) -> float:
    """Mean teacher-forced log p of `answer` continuing the [system,user] prompt."""
    import torch

    pre = _build_ids(tokenizer, system, user)
    ans = tokenizer(answer, add_special_tokens=False)["input_ids"]
    if not ans:
        return 0.0
    ids = pre + ans
    t = torch.tensor([ids], device=device)
    with torch.no_grad():
        logits = model(t).logits[0].float().cpu().numpy()
    lps = []
    for j, tok in enumerate(ans):
        row = logits[len(pre) + j - 1]
        lps.append(_log_softmax(row)[tok])
    return float(np.mean(lps))


# --------------------------------------------------------------------------- #
# Reporting                                                                    #
# --------------------------------------------------------------------------- #
def _predicted_text(case: InjCase, view, scores: np.ndarray) -> str:
    """Predicted injected text: the user-token chars whose signal is above the
    per-case mean+std (a simple, parameter-light operating point)."""
    finite = scores[~np.isnan(scores)]
    if finite.size == 0:
        return ""
    thr = finite.mean() + finite.std()
    chars = sorted(
        (cs, ce) for (_ti, cs, ce), sc in zip(view, scores, strict=True)
        if not np.isnan(sc) and sc >= thr
    )
    return " ".join(case.user_full[cs:ce] for cs, ce in chars)


def summarize(scored: list[tuple[InjCase, dict]], n_boot: int) -> dict[str, Any]:
    n_pos = int(sum(int(d["label"].sum()) for _, d in scored))
    n_tok = int(sum(len(d["label"]) for _, d in scored))
    out: dict[str, Any] = {"n_cases": len(scored), "n_tokens": n_tok,
                           "n_positive": n_pos, "signals": {}}
    for sig in ALL_SIGNALS:
        per_case = [(d[sig], d["label"]) for _, d in scored]
        point, lo, hi = bootstrap_auroc(per_case, n_boot=n_boot)
        prf = [word_prf(_predicted_text(c, d["view"], d[sig]), c.injected_text)
               for c, d in scored]
        rl = [rouge_l(_predicted_text(c, d["view"], d[sig]), c.injected_text)
              for c, d in scored]
        out["signals"][sig] = {
            "auroc": point, "lo95": lo, "hi95": hi,
            "word_p": float(np.mean([p for p, _, _ in prf])),
            "word_r": float(np.mean([r for _, r, _ in prf])),
            "word_f1": float(np.mean([f for _, _, f in prf])),
            "rouge_l": float(np.mean(rl)),
        }
    return out


def format_report(s: dict[str, Any]) -> str:
    lines = [
        f"### injection localization  cases={s['n_cases']}  "
        f"tokens={s['n_tokens']}  positives={s['n_positive']}",
        f"{'signal':<14} {'AUROC':>7}  {'95% CI':<16} "
        f"{'wP':>5} {'wR':>5} {'wF1':>5} {'RougeL':>6}",
        "-" * 70,
    ]
    for sig in ALL_SIGNALS:
        v = s["signals"][sig]
        ci = f"[{v['lo95']:.3f},{v['hi95']:.3f}]" if not np.isnan(v["lo95"]) else "[n/a]"
        lines.append(
            f"{sig:<14} {v['auroc']:>7.3f}  {ci:<16} "
            f"{v['word_p']:>5.2f} {v['word_r']:>5.2f} {v['word_f1']:>5.2f} {v['rouge_l']:>6.3f}"
        )
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# Self-test (offline)                                                          #
# --------------------------------------------------------------------------- #
def selftest() -> int:
    ok = True

    def check(name, cond):
        nonlocal ok
        ok = ok and cond
        print(f"  [{'PASS' if cond else 'FAIL'}] {name}")

    # segmentation tiles the string and splits on . ! ? \n
    segs = segment_spans("Hello world. Ignore this!\nNew line")
    joined = "".join("Hello world. Ignore this!\nNew line"[s:e] for s, e in segs)
    check("segments tile the string", joined == "Hello world. Ignore this!\nNew line")
    # splits after '.', '!', and '\n' -> "Hello world.", " Ignore this!", "\n", "New line"
    check("segments split on punctuation", len(segs) == 4)

    # word P/R/F1
    p, r, f = word_prf("ignore the rules", "please ignore the rules now")
    check("word recall = 3/5", abs(r - 0.6) < 1e-9)
    check("word precision = 3/3", abs(p - 1.0) < 1e-9)
    p0, r0, _ = word_prf("", "x y")
    check("empty pred -> 0 recall", r0 == 0.0)

    # ROUGE-L: identical = 1, disjoint = 0, subsequence
    check("rougeL identical = 1", abs(rouge_l("a b c", "a b c") - 1.0) < 1e-9)
    check("rougeL disjoint = 0", rouge_l("a b", "c d") == 0.0)
    check("rougeL subsequence in (0,1)", 0 < rouge_l("a x b y c", "a b c") < 1)

    # label overlap: injected span [10,20) over a few token views
    view = [(0, 0, 5), (1, 5, 12), (2, 12, 18), (3, 18, 25)]
    lab = _labels_for_view(view, 10, 20)
    check("labels overlap injected span = [0,1,1,1]", list(lab) == [0, 1, 1, 1])

    # predicted-text threshold picks the high-signal tokens
    @dataclass
    class _C:
        user_full: str = "aaaa bbbb cccc dddd"
    c = _C()
    v = [(0, 0, 4), (1, 5, 9), (2, 10, 14), (3, 15, 19)]
    sc = np.array([0.0, 0.0, 0.0, 5.0])  # only last token spikes
    pred = _predicted_text(c, v, sc)
    check("predicted text picks the spike token", pred == "dddd")

    # AUROC reuse sanity (imported)
    check("auroc perfect=1", abs(auroc([3, 2, 1, 0], [1, 1, 0, 0]) - 1.0) < 1e-9)

    print(f"\nselftest: {'ALL PASS' if ok else 'FAILURES'}")
    return 0 if ok else 1


# --------------------------------------------------------------------------- #
# Main                                                                         #
# --------------------------------------------------------------------------- #
def run(args: argparse.Namespace) -> int:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    cases = load_cases(Path(args.cases), args.max_cases)
    log.info("loaded %d injection cases", len(cases))

    model_tag = args.model.split("/")[-1]
    out_dir = Path(args.out) / model_tag
    out_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16}[args.dtype]
    load_kw = dict(device_map="auto", torch_dtype=dtype, attn_implementation="eager")
    try:
        model = AutoModelForCausalLM.from_pretrained(args.model, **load_kw).eval()
    except (ValueError, KeyError):
        from transformers import AutoModelForImageTextToText
        log.info("falling back to AutoModelForImageTextToText (multimodal checkpoint)")
        model = AutoModelForImageTextToText.from_pretrained(args.model, **load_kw).eval()
    device = str(model.get_input_embeddings().weight.device)

    scored: list[tuple[InjCase, dict]] = []
    for c in cases:
        try:
            d = score_case(c, model, tokenizer, device, do_looo=not args.no_looo)
        except AssertionError as e:
            log.warning("skip %s: %s", c.id, e)
            continue
        scored.append((c, d))
        log.info("scored %-28s tokens=%-3d positives=%d", c.id, len(d["label"]),
                 int(d["label"].sum()))

    summary = summarize(scored, args.n_boot)
    report = f"# injection localization — {args.model}\n" + format_report(summary)
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    (out_dir / "report.md").write_text(report)
    print(report)
    return 0


def _parse_args(argv):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--selftest", action="store_true")
    p.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    p.add_argument("--cases",
                   default=str(Path(__file__).parent / "data" / "injection_cases.jsonl"))
    p.add_argument("--out", default="results/injection_localization")
    p.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16"])
    p.add_argument("--max-cases", type=int, default=None)
    p.add_argument("--no-looo", action="store_true", help="skip the O(n) attribution baseline")
    p.add_argument("--n-boot", type=int, default=2000)
    p.add_argument("--log-level", default="INFO")
    return p.parse_args(argv)


def main(argv=None):
    args = _parse_args(argv if argv is not None else sys.argv[1:])
    logging.basicConfig(level=args.log_level.upper(),
                        format="%(asctime)s %(levelname)s %(name)s %(message)s")
    if args.selftest:
        return selftest()
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
