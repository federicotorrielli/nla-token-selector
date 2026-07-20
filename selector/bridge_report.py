"""selector/bridge_report.py — assemble the bridge results file.

Reads every model's on-task judgements (results/bridge/<short>_ontask.parquet)
and per-token cheap signals (results/token_selector_v2/<tag>/tokens.parquet),
and writes a standalone markdown report answering the two bridge questions per
model:

  Q1  Is the NLA more on-task inside the flagged span than outside it?
  Q2  Which cheap pre-pass signal best predicts where the NLA is on-task?

Usage:
    python selector/bridge_report.py --out findings/token-selector/bridge-nla-selection.md
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parent))
from signals import auroc  # noqa: E402

_FOUR = [("q7", "Qwen2.5-7B-Instruct", "Qwen-7B"), ("g12", "gemma-3-12b-it", "Gemma-12B"),
         ("g27", "gemma-3-27b-it", "Gemma-27B"), ("l70", "Llama-3.3-70B-Instruct", "Llama-70B")]

# Per experiment kind: which models, where the on-task labels and the cheap signals
# live, the signal list, which are blind (one pass), and the pre-specified primaries.
KINDS = {
    "hand": {  # 27 authored injection/eval cases, signals in tokens.parquet
        "models": _FOUR, "ontask": "results/bridge/{short}_ontask.parquet",
        "signals_from": "tokens", "join": ["case_id", "mode", "tok_idx"],
        "signals": ["surprisal", "entropy", "varentropy", "resid_jump", "temporal_kl",
                    "lookback_ratio", "sink_drain", "head_disagreement", "w", "kl", "attn_rollout"],
        "blind": {"surprisal", "entropy", "varentropy", "resid_jump", "temporal_kl",
                  "lookback_ratio", "sink_drain", "head_disagreement", "w"},
        "primaries": ("head_disagreement", "kl"),
        "title": "The bridge on the 27 authored cases",
    },
    "opi": {  # OpenPromptInjection input tokens, signals in the opi corpus
        "models": _FOUR, "ontask": "results/bridge/opi_{short}_ontask.parquet",
        "signals_from": "results/bridge/opi_{short}_signals.parquet", "join": ["position_id"],
        "signals": ["in_surprisal", "in_entropy", "in_attention"],
        "blind": {"in_surprisal", "in_entropy", "in_attention"},
        "primaries": (),  # input-token setting is new: all exploratory
        "title": "The bridge on real injection (OpenPromptInjection, all 800 cases)",
    },
    "liars": {  # Liars' Bench response tokens, signals in the liars corpus
        "models": [m for m in _FOUR if m[0] in ("l70", "g27")],
        "ontask": "results/bridge/liars_{short}_ontask.parquet",
        "signals_from": "results/bridge/liars_{short}_corpus.parquet", "join": ["position_id"],
        "signals": ["surprisal", "entropy", "varentropy", "resid_jump",
                    "lookback_ratio", "sink_drain", "head_disagreement"],
        "blind": {"surprisal", "entropy", "varentropy", "resid_jump",
                  "lookback_ratio", "sink_drain", "head_disagreement"},
        "primaries": ("head_disagreement",),  # kl needs a reference we lack here
        "title": "The bridge on real lies (Liars' Bench, on-policy)",
    },
    "tt": {  # Tensor Trust hijacking, response tokens (attack vs access_code variant)
        "models": _FOUR, "ontask": "results/bridge/tt_{short}_ontask.parquet",
        "signals_from": "results/bridge/tt_{short}_corpus.parquet", "join": ["position_id"],
        "signals": ["surprisal", "entropy", "varentropy", "resid_jump",
                    "lookback_ratio", "sink_drain", "head_disagreement"],
        "blind": {"surprisal", "entropy", "varentropy", "resid_jump",
                  "lookback_ratio", "sink_drain", "head_disagreement"},
        "primaries": ("head_disagreement",),  # injection primary, pre-specified
        "title": "The bridge on Tensor Trust (prompt hijacking, response side)",
    },
    "taboo": {  # taboo organisms, response tokens; on-task = NLA reveals the secret word
        "models": _FOUR, "ontask": "results/bridge/taboo_{short}_ontask.parquet",
        "signals_from": "results/bridge/taboo_{short}_corpus.parquet", "join": ["position_id"],
        "signals": ["surprisal", "entropy", "varentropy", "resid_jump",
                    "lookback_ratio", "sink_drain", "head_disagreement"],
        "blind": {"surprisal", "entropy", "varentropy", "resid_jump",
                  "lookback_ratio", "sink_drain", "head_disagreement"},
        "primaries": (),  # secret-word setting is new: all exploratory
        "title": "The bridge on taboo organisms (secret word, response side)",
    },
}


def _load(cfg: dict, short: str, tag: str):
    ot = Path(cfg["ontask"].format(short=short))
    if not ot.exists():
        return None
    o = pl.read_parquet(ot)
    if cfg["signals_from"] == "tokens":
        src = Path(f"results/token_selector_v2/{tag}/tokens.parquet")
    else:
        src = Path(cfg["signals_from"].format(short=short))
    if not src.exists():
        return None
    t = pl.read_parquet(src)
    keep = [*cfg["join"], *[c for c in cfg["signals"] if c in t.columns]]
    # position_id is already in the ontask frame; drop it from the right side dup
    if cfg["join"] == ["position_id"]:
        keep = ["position_id", *[c for c in cfg["signals"] if c in t.columns]]
    return o.join(t.select(keep), on=cfg["join"], how="left")


def _q1(df: pl.DataFrame, mode: str, n_boot=2000):
    """On-task rate inside vs outside the span, the lift, and a case-cluster
    bootstrap 95% interval on the lift."""
    m = df.filter(pl.col("mode") == mode)
    y = np.array(m["on_task"].to_list())
    lab = np.array(m["label"].to_list())
    cases = np.array(m["case_id"].to_list())
    inside, outside = y[lab == 1], y[lab == 0]
    if inside.size == 0 or outside.size == 0:
        return None

    def lift(yy, ll):
        i, o = yy[ll == 1], yy[ll == 0]
        return i.mean() - o.mean() if i.size and o.size else np.nan

    rng = np.random.default_rng(0)
    uniq = np.unique(cases)
    idx_by = {c: np.where(cases == c)[0] for c in uniq}
    boots = []
    for _ in range(n_boot):
        sel = np.concatenate([idx_by[c] for c in rng.choice(uniq, len(uniq), replace=True)])
        v = lift(y[sel], lab[sel])
        if not np.isnan(v):
            boots.append(v)
    lo, hi = (np.percentile(boots, [2.5, 97.5]) if boots else (np.nan, np.nan))
    return (inside.mean(), inside.size, outside.mean(), outside.size,
            inside.mean() - outside.mean(), float(lo), float(hi))


def _clustered_boot(cases, scores, y, stat, n_boot=2000, seed=0):
    """Case-cluster bootstrap of `stat(scores_sel, y_sel)`: resample whole cases
    (tokens within a case are correlated). Returns (point, lo95, hi95, p_two_sided)
    where p is the two-sided bootstrap p-value against the null value 0.5."""
    rng = np.random.default_rng(seed)
    uniq = np.unique(cases)
    idx_by = {c: np.where(cases == c)[0] for c in uniq}
    point = stat(scores, y)
    boots = []
    for _ in range(n_boot):
        pick = rng.choice(uniq, size=len(uniq), replace=True)
        sel = np.concatenate([idx_by[c] for c in pick])
        v = stat(scores[sel], y[sel])
        if not np.isnan(v):
            boots.append(v)
    if not boots:
        return point, float("nan"), float("nan"), float("nan")
    b = np.array(boots)
    lo, hi = np.percentile(b, [2.5, 97.5])
    p = 2.0 * min((b <= 0.5).mean(), (b >= 0.5).mean())
    return point, float(lo), float(hi), float(min(p, 1.0))


def _bh_fdr(pvals, q=0.05):
    """Benjamini-Hochberg step-up: boolean 'passes FDR at q' per p-value."""
    p = np.asarray(pvals, dtype=float)
    m = len(p)
    if m == 0:
        return np.zeros(0, bool)
    order = np.argsort(p)
    thresh = q * np.arange(1, m + 1) / m
    passed = np.zeros(m, bool)
    below = np.where(p[order] <= thresh)[0]
    if below.size:
        passed[order[: below.max() + 1]] = True
    return passed


def _q2(df: pl.DataFrame, mode: str, signals, primaries, n_boot=2000):
    """Per-signal AUROC predicting on_task, with case-cluster CI, bootstrap p, and
    BH-FDR flag over the exploratory (non-primary) family. Plus two controls:
    a random-normal score and a label-permutation null (both must sit at ~0.5)."""
    m = df if mode == "pooled" else df.filter(pl.col("mode") == mode)
    y = np.array(m["on_task"].to_list())
    cases = np.array(m["case_id"].to_list())
    if y.sum() == 0 or y.sum() == len(y):
        return None, float("nan"), None
    rows = []
    for c in signals:
        if c not in m.columns:
            continue
        s = np.array(m[c].to_list(), dtype=float)
        pt, lo, hi, p = _clustered_boot(cases, s, y, auroc, n_boot=n_boot)
        if not np.isnan(pt):
            rows.append({"sig": c, "auroc": pt, "lo": lo, "hi": hi, "p": p,
                         "primary": c in primaries})
    rows.sort(key=lambda r: abs(r["auroc"] - 0.5), reverse=True)
    # BH-FDR over the exploratory family only (primaries are confirmatory)
    expl = [r for r in rows if not r["primary"]]
    passed = _bh_fdr([r["p"] for r in expl])
    for r, ok in zip(expl, passed, strict=True):
        r["fdr"] = bool(ok)
    for r in rows:
        r.setdefault("fdr", True)  # primaries: confirmatory, no penalty
    # controls
    rng = np.random.default_rng(0)
    rand_auroc = auroc(rng.standard_normal(len(y)), y)
    perm_devs = [abs(auroc(np.array(m[rows[0]["sig"]].to_list(), dtype=float),
                           rng.permutation(y)) - 0.5) for _ in range(200)]
    controls = {"random": float(rand_auroc), "perm_top_meandev": float(np.mean(perm_devs))}
    return rows, float(y.mean()), controls


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--kind", default="hand", choices=list(KINDS))
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)
    cfg = KINDS[args.kind]
    out_path = args.out or f"findings/token-selector/bridge-{args.kind}.md"

    loaded = [(s, t, disp, _load(cfg, s, t)) for s, t, disp in cfg["models"]]
    have = [(s, t, disp, df) for s, t, disp, df in loaded if df is not None]
    if not have:
        print(f"no {args.kind} bridge outputs found under results/bridge/", file=sys.stderr)
        return 1
    SIGNALS, BLIND, PRIMARIES = cfg["signals"], cfg["blind"], cfg["primaries"]

    L = []
    L.append(f"# {cfg['title']}\n")
    L.append("> Bridge results. At each token we run the NLA, ask the judge "
             "(nvidia/DeepSeek-V4-Flash-NVFP4) whether its explanation is *on-task* for "
             "the threat, and test two things: is the NLA more on-task inside the "
             "flagged span (Q1), and does a cheap one-pass signal predict where it is "
             "on-task (Q2). AUROC: 0.5 is chance; below 0.5 the signal points the other "
             "way. Intervals are case-cluster bootstraps over whole transcripts. "
             f"Pre-specified primary signal(s): {', '.join(PRIMARIES) or '(none — exploratory)'}.\n")

    modes = sorted({m for _, _, _, df in have for m in df["mode"].unique().to_list()})
    mode_disp = {"injection": "injection", "evalaware": "eval-awareness",
                 "injection_input": "injection (input span)", "deception": "deception"}

    L.append("## Q1 — is the NLA more on-task inside the flagged span?\n")
    L.append("On-task rate inside vs outside the planted span, the lift, and its "
             "case-cluster bootstrap 95% interval (2000 resamples over whole "
             "transcripts). A lift whose interval clears 0 is a real localization "
             "of the NLA's on-task-ness to the flagged span.\n")
    L.append("| model | threat | inside | outside | lift [95% CI] |")
    L.append("|---|---|---|---|---|")
    for _s, _t, disp, df in have:
        for mode in modes:
            r = _q1(df, mode)
            if r is None:
                continue
            ins, ni, out, no, lift, lo, hi = r
            star = " ✓" if lo > 0 else ""
            L.append(f"| {disp} | {mode_disp.get(mode, mode)} | {ins:.2f} (n={ni}) | "
                     f"{out:.2f} (n={no}) | **{lift:+.2f}** [{lo:+.2f}, {hi:+.2f}]{star} |")
    L.append("")

    L.append("## Q2 — which cheap pre-pass signal predicts where the NLA is on-task?\n")
    L.append("Full table: AUROC of **every** signal vs the judge's on-task label, one row "
             "per signal, one column per model, with a case-cluster bootstrap 95% interval. "
             "The **best signal for the task is bolded** per model. `*` = blind (one pass, "
             "no counterfactual). `†` = passes Benjamini-Hochberg FDR at q=0.05 over the "
             "exploratory family; the pre-specified primaries are confirmatory and shown "
             "without penalty. The last two rows give the on-task base rate and two controls "
             "(a random-normal score, and the top signal's AUROC under label permutation), "
             "both of which sit at ~0.5 when the machinery is honest.\n")
    q2_modes = [*modes, "pooled"] if len(modes) > 1 else modes
    for mode in q2_modes:
        L.append(f"### {mode_disp.get(mode, mode)}\n")
        # collect per-model results, then emit a signal x model matrix
        per_model = {}
        for _s, _t, disp, df in have:
            per_model[disp] = _q2(df, mode, SIGNALS, PRIMARIES)
        cols = [disp for _s, _t, disp, _df in have if per_model[disp][0] is not None]
        if not cols:
            L.append("_(on-task all one class for every model; no ranking)_\n")
            continue
        # best-for-task signal per model = strongest separation from 0.5
        best = {c: max(per_model[c][0], key=lambda r: abs(r["auroc"] - 0.5))["sig"] for c in cols}
        by_sig = {c: {r["sig"]: r for r in per_model[c][0]} for c in cols}
        L.append("| signal | " + " | ".join(cols) + " |")
        L.append("|" + "---|" * (len(cols) + 1))
        for sig in SIGNALS:
            if not any(sig in by_sig[c] for c in cols):
                continue
            cells = []
            for c in cols:
                r = by_sig[c].get(sig)
                if r is None:
                    cells.append("—")
                    continue
                mark = "*" if sig in BLIND else ""
                fdr = "†" if r.get("fdr") else ""
                txt = f"{r['auroc']:.3f} [{r['lo']:.2f},{r['hi']:.2f}]{mark}{fdr}"
                cells.append(f"**{txt}**" if best[c] == sig else txt)
            L.append(f"| `{sig}` | " + " | ".join(cells) + " |")
        L.append("| _base rate_ | "
                 + " | ".join(f"{per_model[c][1]:.2f}" for c in cols) + " |")
        L.append("| _control rand / perm_ | "
                 + " | ".join(f"{per_model[c][2]['random']:.2f} / "
                              f"0.5±{per_model[c][2]['perm_top_meandev']:.2f}" for c in cols) + " |")
        L.append("")

    L.append("## Headline: the number\n")
    L.append("Best **blind** pre-pass signal for predicting where the NLA is on-task, "
             "pooled over threats, with its 95% interval:\n")
    L.append("| model | best blind signal | AUROC [95% CI] |")
    L.append("|---|---|---|")
    for _s, _t, disp, df in have:
        scored, _, _ = _q2(df, "pooled", SIGNALS, PRIMARIES)
        if scored is None:
            continue
        blind = [r for r in scored if r["sig"] in BLIND]
        if blind:
            r = blind[0]
            L.append(f"| {disp} | `{r['sig']}` | {r['auroc']:.3f} [{r['lo']:.2f}, {r['hi']:.2f}] |")
    L.append("")

    outp = Path(out_path)
    outp.parent.mkdir(parents=True, exist_ok=True)
    outp.write_text("\n".join(L))
    print(f"wrote {outp} ({len(have)}/{len(cfg['models'])} models)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
