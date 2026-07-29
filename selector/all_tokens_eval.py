"""selector/all_tokens_eval.py — evaluations that read a finished all-token run.

No GPU and no server: everything works from the corpus shards (which hold one
activation per token) and the consolidated per-token tables.

    python selector/all_tokens_eval.py spike      # activation-derived columns
    python selector/all_tokens_eval.py pool       # budget analysis of selectors
    python selector/all_tokens_eval.py position   # position-confound check
    python selector/all_tokens_eval.py all        # the full set, in order
    python selector/all_tokens_eval.py --selftest

`spike` writes results/bridge/all_{kind}_{model}_spike/, which
consolidate_all.py joins into the canonical table. See spike_stats for the
columns. Sun et al., arXiv:2603.05498, show that a few channels dominate the
intermediate layers the NLA reads, and that the tokens carrying massive
activations are mostly the first token and delimiters.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import numpy as np
import polars as pl

BR = Path("results/bridge")
CSVS = Path("paper_results") / "bridge"
KINDS = {"hand": ["q7", "g12", "g27", "l70"], "opi": ["q7", "g12", "g27", "l70"],
         "tt": ["q7", "g12", "g27", "l70"], "taboo": ["q7", "g12", "g27", "l70"],
         "liars": ["g27", "l70"]}

SPIKE_FACTOR = 10.0          # spike channels tower this far over the typical one
SAMPLE_ROWS = 20_000
RRF_K = 60
METHODS = ["sel_a", "sel_b", "sel_pool", "sel_rrf", "sel_and"]
BINS = [(0, 256), (256, 512), (512, 1024), (1024, 2048), (2048, 10**9)]
GEMMA = {"g12", "g27"}       # Gemma-3 local layers carry a 1024-token window


# --------------------------------------------------------------- spike columns

def _acts(df: pl.DataFrame) -> np.ndarray:
    # the column is a fixed-size Array, so to_numpy is zero-copy; going via
    # to_list costs seconds per 20k rows and builds the whole thing in Python
    a = df["activation"].to_numpy()
    return a if a.ndim == 2 else np.stack(a).astype(np.float32)


def spike_channels(acts: np.ndarray, factor: float = SPIKE_FACTOR) -> np.ndarray:
    """Channels whose median magnitude exceeds `factor` times the typical one.

    These are the channels that dominate the norm at every position, not the
    ones that spike in a minority of tokens. Spike *tokens* are identified by
    norm_ratio instead, which needs no channel set.
    """
    typical = np.median(np.abs(acts), axis=0)
    return np.flatnonzero(typical > factor * np.median(typical))


def spike_stats(acts: np.ndarray, spike: np.ndarray) -> dict[str, np.ndarray]:
    """Per-token activation statistics. Row order is token order in a transcript.

    dominant_mass    share of the squared norm in the norm-dominating channels
    peak_ratio       largest |channel| over the root mean square
    act_norm         norm of the activation
    norm_ratio       act_norm over the median act_norm of the same shard; a
                     spike token carries a massive activation and so stands
                     orders above its neighbours
    resid_jump_nla   ||h_t - h_{t-1}|| at the NLA layer, dominating channels
                     dropped. Distinct from the corpus `resid_jump`, which is
                     taken at the final layer where the spikes are neutralised.
    """
    sq = acts.astype(np.float64) ** 2
    total = sq.sum(axis=1)
    d = acts.shape[1]
    keep = np.setdiff1d(np.arange(d), spike)
    diff = np.diff(acts[:, keep].astype(np.float64), axis=0)
    norm = np.sqrt(total)
    return {
        "dominant_mass": (sq[:, spike].sum(axis=1) / np.maximum(total, 1e-30)
                          if spike.size else np.zeros(len(acts))),
        "peak_ratio": np.abs(acts).max(axis=1) / np.sqrt(np.maximum(total / d, 1e-30)),
        "act_norm": norm,
        "norm_ratio": norm / max(float(np.median(norm)), 1e-30),
        "resid_jump_nla": np.concatenate([[np.nan], np.linalg.norm(diff, axis=1)]),
    }


def run_spike() -> None:
    for kind, shorts in KINDS.items():
        for short in shorts:
            src = BR / f"all_{kind}_{short}_corpus"
            shards = sorted(src.glob("*.parquet")) if src.is_dir() else []
            if not shards:
                continue
            take = max(1, SAMPLE_ROWS // len(shards))
            sample = np.concatenate([_acts(pl.read_parquet(s).head(take)) for s in shards])
            spike = spike_channels(sample)
            out = BR / f"all_{kind}_{short}_spike"
            out.mkdir(parents=True, exist_ok=True)
            print(f"{kind}/{short}: {len(shards)} shards, "
                  f"spike channels {spike.tolist() or 'none'}")
            for s in shards:
                dst = out / s.name
                if dst.exists():
                    continue
                df = pl.read_parquet(s)
                frame = df.select(pl.col("position_id").cast(pl.Int64), "case_id")
                frame = frame.with_columns(
                    **{k: pl.Series(v) for k, v in spike_stats(_acts(df), spike).items()}
                ).with_columns(
                    pl.when(pl.col("case_id") != pl.col("case_id").shift(1))
                    .then(None).otherwise(pl.col("resid_jump_nla"))
                    .alias("resid_jump_nla")
                ).drop("case_id")
                tmp = dst.with_suffix(".tmp.parquet")
                frame.write_parquet(tmp)
                tmp.replace(dst)


# ------------------------------------------------------------ budget analysis

def _rank(df: pl.DataFrame, sig: str, tail: str) -> pl.DataFrame:
    z = (pl.col(sig) - pl.col(sig).mean().over("case_id")) / (
        pl.col(sig).std().over("case_id") + 1e-12)
    score = z.abs() if tail == "two" else z * pl.lit(df["_sign_" + sig][0])
    score = score.fill_nan(None).fill_null(float("-inf"))
    return df.with_columns(
        score.rank("ordinal", descending=True).over("case_id").alias(f"r_{sig}"))


def _oracle_sign(df: pl.DataFrame, sig: str) -> float:
    hit = df.filter(pl.col("on_task") == 1)[sig].mean()
    miss = df.filter(pl.col("on_task") == 0)[sig].mean()
    if hit is None or miss is None:
        return 1.0
    return 1.0 if hit >= miss else -1.0


def select(df: pl.DataFrame, sig_a: str, sig_b: str, frac: float, tail: str) -> pl.DataFrame:
    """Flag the tokens each method spends its budget on. Every method gets the
    same budget: `pool` is the union, `rrf` a score fusion, `and` the
    intersection."""
    for sig in (sig_a, sig_b):
        df = df.with_columns(pl.lit(_oracle_sign(df, sig)).alias("_sign_" + sig))
        df = _rank(df, sig, tail)
    ra, rb = pl.col(f"r_{sig_a}"), pl.col(f"r_{sig_b}")
    budget = (pl.len().over("case_id") * frac).ceil().clip(1)
    rrf = 1.0 / (RRF_K + ra) + 1.0 / (RRF_K + rb)
    return df.with_columns(
        sel_a=ra <= budget,
        sel_b=rb <= budget,
        sel_pool=pl.min_horizontal(ra, rb).rank("ordinal").over("case_id") <= budget,
        sel_rrf=rrf.rank("ordinal", descending=True).over("case_id") <= budget,
        sel_and=pl.max_horizontal(ra, rb).rank("ordinal").over("case_id") <= budget,
    )


def _per_case(df: pl.DataFrame) -> pl.DataFrame:
    agg = [pl.col("on_task").sum().alias("hits"), pl.len().alias("n")]
    for m in METHODS:
        agg += [(pl.col("on_task") * pl.col(m).cast(pl.Int8)).sum().alias(f"h_{m}"),
                pl.col(m).sum().alias(f"s_{m}")]
    return df.group_by("case_id").agg(agg)


def _precision(cases: pl.DataFrame, m: str) -> float:
    spend = cases[f"s_{m}"].sum()
    return float(cases[f"h_{m}"].sum() / spend) if spend else float("nan")


def _boot_delta(cases, method: str, best: str, n_boot: int, seed: int) -> tuple[float, float]:
    """Case-clustered bootstrap interval on precision(method) - precision(best)."""
    rng = np.random.default_rng(seed)
    hp, sp = cases[f"h_{method}"].to_numpy(), cases[f"s_{method}"].to_numpy()
    hb, sb = cases[f"h_{best}"].to_numpy(), cases[f"s_{best}"].to_numpy()
    idx = rng.integers(0, len(hp), (n_boot, len(hp)))
    d = (hp[idx].sum(1) / np.maximum(sp[idx].sum(1), 1)
         - hb[idx].sum(1) / np.maximum(sb[idx].sum(1), 1))
    return float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))


def _overlap(df: pl.DataFrame) -> float:
    a, b = df["sel_a"].to_numpy(), df["sel_b"].to_numpy()
    union = (a | b).sum()
    return float((a & b).sum() / union) if union else float("nan")


def _regions(df: pl.DataFrame) -> dict[str, float]:
    a, b = df["sel_a"].to_numpy(), df["sel_b"].to_numpy()
    y = df["on_task"].to_numpy()
    out: dict[str, float] = {}
    for name, mask in [("both", a & b), ("a_only", a & ~b), ("b_only", b & ~a)]:
        out[name] = float(y[mask].mean()) if mask.sum() else float("nan")
        out["n_" + name] = int(mask.sum())
    return out


def _sources(all_tokens: bool):
    if all_tokens:
        for p in sorted(BR.glob("all_*.parquet")):
            task, model = p.stem[len("all_"):].rsplit("_", 1)
            df = pl.read_parquet(p)
            if "on_task" in df.columns:
                yield task, model, df.filter(pl.col("on_task").is_not_null())
    else:
        for p in sorted(CSVS.glob("*.csv")):
            task, model = p.stem.rsplit("_", 1)
            yield task, model, pl.read_csv(p)


def run_pool(sig_a: str, sig_b: str, frac: float, tail: str, n_boot: int,
             seed: int, all_tokens: bool) -> str:
    rows = []
    for task, model, df in _sources(all_tokens):
        if sig_a not in df.columns or sig_b not in df.columns or df.is_empty():
            continue
        for mode in sorted(df["mode"].unique()) if task == "hand" else [None]:
            sub = df.filter(pl.col("mode") == mode) if mode else df
            sub = select(sub, sig_a, sig_b, frac, tail)
            cases = _per_case(sub)
            p = {m: _precision(cases, m) for m in METHODS}
            best = "sel_a" if p["sel_a"] >= p["sel_b"] else "sel_b"
            lo, hi = _boot_delta(cases, "sel_pool", best, n_boot, seed)
            alo, ahi = _boot_delta(cases, "sel_and", best, n_boot, seed)
            rows.append(dict(task=mode or task, model=model,
                             base=float(sub["on_task"].mean()), a=p["sel_a"],
                             b=p["sel_b"], pool=p["sel_pool"], rrf=p["sel_rrf"],
                             andd=p["sel_and"], delta=p["sel_pool"] - p[best],
                             lo=lo, hi=hi, adelta=p["sel_and"] - p[best],
                             alo=alo, ahi=ahi, jac=_overlap(sub), **_regions(sub)))

    out = [
        f"# Pooling `{sig_a}` + `{sig_b}` at a {frac:.0%} budget ({tail}-tailed)\n",
        "precision@B = of the B explanations bought, the fraction the judge called",
        "on-task, best per row in bold. Every method spends the same B. `pool`",
        "interleaves the two rankings (the union), `rrf` fuses their scores, `and`",
        "ranks by the worse of the two ranks (the intersection). Each delta is that",
        "combiner minus the better single signal, with a case-clustered 95%",
        "interval; it has to clear 0 for the combiner to be worth anything.",
        "`jaccard` is how much the two signals pick the same tokens (1.0 =",
        "identical).\n",
        f"| task | model | base rate | {sig_a} | {sig_b} | pool | rrf | and | "
        "pool delta | and delta | jaccard |",
        "|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        win = " ✓" if r["lo"] > 0 else ""
        awin = " ✓" if r["alo"] > 0 else ""
        cols = ("a", "b", "pool", "rrf", "andd")
        top = max(r[k] for k in cols)
        c = {k: (f"**{r[k]:.3f}**" if r[k] == top else f"{r[k]:.3f}") for k in cols}
        out.append(
            f"| {r['task']} | {r['model']} | {r['base']:.2f} | {c['a']} | "
            f"{c['b']} | {c['pool']} | {c['rrf']} | {c['andd']} | "
            f"{r['delta']:+.3f} [{r['lo']:+.3f}, {r['hi']:+.3f}]{win} | "
            f"{r['adelta']:+.3f} [{r['alo']:+.3f}, {r['ahi']:+.3f}]{awin} | {r['jac']:.2f} |")

    out += [
        "\n## Where the two methods agree, and where they do not\n",
        "On-task rate among the tokens both methods bought, and among the ones only",
        "one of them bought. A union pays for both exclusive columns, so it can only",
        "pay off when those match the agreement column.\n",
        f"| task | model | base rate | both | {sig_a} only | {sig_b} only |",
        "|---|---|---|---|---|---|",
    ]
    for r in rows:
        out.append(f"| {r['task']} | {r['model']} | {r['base']:.2f} | "
                   f"**{r['both']:.3f}** (n={r['n_both']}) | {r['a_only']:.3f} "
                   f"(n={r['n_a_only']}) | {r['b_only']:.3f} (n={r['n_b_only']}) |")
    return "\n".join(out)


# ----------------------------------------------------------- position confound

def drift(df: pl.DataFrame, col: str) -> tuple[list[float], float]:
    """Mean of `col` per position bin, and the slope of its z-score against
    position, in standard deviations per 1000 tokens."""
    means = []
    for lo, hi in BINS:
        s = df.filter((pl.col("tok_idx") >= lo) & (pl.col("tok_idx") < hi))[col]
        s = s.drop_nulls().drop_nans() if s.len() else s
        means.append(float(s.mean()) if s.len() else float("nan"))
    x = df["tok_idx"].to_numpy().astype(float)
    y = df[col].to_numpy().astype(float)
    ok = np.isfinite(x) & np.isfinite(y)
    if ok.sum() < 2 or np.std(y[ok]) < 1e-12:
        return means, float("nan")
    z = (y[ok] - y[ok].mean()) / y[ok].std()
    return means, float(np.polyfit(x[ok], z, 1)[0] * 1000)


def run_position(signal: str) -> None:
    """Past its 1024-token window a Gemma local layer cannot see token 0, which
    lowers the sink mass and raises sink_drain with position. On-task rate by
    position is printed beside it, since a signal that merely tracks position
    would score against it."""
    hdr = " ".join(f"{f'{lo}-{hi}' if hi < 10**9 else f'{lo}+':>10}" for lo, hi in BINS)
    print(f"\n{signal} by position\n\n{'kind/model':16}{'slope/1k':>10}  {hdr}")
    for p in sorted(BR.glob("all_*.parquet")):
        kind, model = p.stem[len("all_"):].rsplit("_", 1)
        df = pl.read_parquet(p)
        if signal not in df.columns or "tok_idx" not in df.columns:
            continue
        means, slope = drift(df, signal)
        cells = " ".join(f"{m:10.4f}" if np.isfinite(m) else f"{'-':>10}" for m in means)
        print(f"{kind + '/' + model:16}{slope:10.3f}  {cells}"
              f"{' <- gemma' if model in GEMMA else ''}")

    print(f"\non-task rate by position\n\n{'kind/model':16}{'':10}  {hdr}")
    for p in sorted(BR.glob("all_*.parquet")):
        kind, model = p.stem[len("all_"):].rsplit("_", 1)
        df = pl.read_parquet(p)
        if "on_task" not in df.columns:
            continue
        df = df.filter(pl.col("on_task").is_not_null())
        if df.is_empty():
            continue
        means, _ = drift(df, "on_task")
        cells = " ".join(f"{m:10.3f}" if np.isfinite(m) else f"{'-':>10}" for m in means)
        print(f"{kind + '/' + model:16}{'':10}  {cells}")


# --------------------------------------------------------------- full sequence

def run_all(py: str) -> None:
    def step(title: str, *cmd: str) -> None:
        print(f"\n=== {title} ===", flush=True)
        subprocess.run([py, *cmd], check=False)

    run_spike()
    step("consolidate", "selector/consolidate_all.py")
    for k in KINDS:
        step(f"report {k}", "selector/bridge_report.py", "--kind", k, "--all-tokens")
        step(f"report {k} content-only", "selector/bridge_report.py", "--kind", k,
             "--all-tokens", "--content-only")
    step("per-token CSVs", "selector/make_csvs.py", "--all")
    for frac, name in [(0.10, "bridge-pooling-all.md"), (0.01, "bridge-pooling-all-1pct.md")]:
        md = run_pool("entropy", "sink_drain", frac, "oracle", 2000, 0, True)
        Path("findings/token-selector", name).write_text(md + "\n")
        print(f"wrote findings/token-selector/{name}")
    run_position("sink_drain")


# --------------------------------------------------------------------- selftest

def _selftest() -> None:
    rng = np.random.default_rng(0)
    d, n = 64, 200
    acts = rng.normal(0, 1, (n, d)).astype(np.float32)
    acts[:, [7, 31]] *= 500.0
    spike = spike_channels(acts)
    assert spike.tolist() == [7, 31], f"spike channels: {spike.tolist()}"
    st = spike_stats(acts, spike)
    assert st["dominant_mass"].mean() > 0.9, "planted channels must dominate the norm"
    assert np.isnan(st["resid_jump_nla"][0]), "first token has no predecessor"
    plain = np.linalg.norm(np.diff(acts.astype(np.float64), axis=0), axis=1)
    assert plain.mean() > 20 * st["resid_jump_nla"][1:].mean(), "masking must bite"
    flat = rng.normal(0, 1, (n, d)).astype(np.float32)
    assert spike_channels(flat).size == 0, "no spikes in a flat matrix"
    assert spike_stats(flat, np.array([], int))["dominant_mass"].max() == 0.0

    # budget 2 over 10 tokens; each signal finds one of the two on-task tokens
    # and one decoy, so pooling must reach 1.0 where either alone reaches 0.5
    df = pl.DataFrame({"case_id": ["c"] * 10, "on_task": [1, 1] + [0] * 8,
                       "entropy": [9.0, 1.0, -6.0] + [1.0] * 7,
                       "sink_drain": [1.0, 9.0, 1.0, -6.0] + [1.0] * 6})
    sel = select(df, "entropy", "sink_drain", frac=0.2, tail="two")
    cases = _per_case(sel)
    assert _precision(cases, "sel_a") == 0.5, "one-signal precision"
    assert _precision(cases, "sel_b") == 0.5, "one-signal precision"
    assert _precision(cases, "sel_pool") == 1.0, "pool must catch both"
    assert sel["sel_pool"].sum() == 2, "pool must respect the budget"
    assert _overlap(sel) == 0.0, "disjoint picks"
    same = select(df.with_columns(sink_drain=pl.col("entropy")),
                  "entropy", "sink_drain", frac=0.2, tail="two")
    assert _precision(_per_case(same), "sel_pool") == 0.5, "no free lunch"
    assert _overlap(same) == 1.0, "identical picks"

    idx = np.arange(3000)
    assert drift(pl.DataFrame({"tok_idx": idx, "s": idx * 0.001}), "s")[1] > 1.0
    assert abs(drift(pl.DataFrame({"tok_idx": idx, "s": np.sin(idx * 0.5)}), "s")[1]) < 0.1
    assert np.isnan(drift(pl.DataFrame({"tok_idx": idx, "s": np.ones(3000)}), "s")[1])
    print("selftest ok")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", nargs="?", choices=["spike", "pool", "position", "all"])
    ap.add_argument("--signals", nargs=2, default=["entropy", "sink_drain"])
    ap.add_argument("--frac", type=float, default=0.10)
    ap.add_argument("--tail", choices=["two", "oracle"], default="oracle")
    ap.add_argument("--all-tokens", action="store_true")
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--signal", default="sink_drain")
    ap.add_argument("--out", type=Path)
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()
    if args.selftest:
        return _selftest()
    if args.cmd == "spike":
        return run_spike()
    if args.cmd == "position":
        return run_position(args.signal)
    if args.cmd == "pool":
        md = run_pool(*args.signals, args.frac, args.tail, args.n_boot, args.seed,
                      args.all_tokens)
        if args.out:
            args.out.write_text(md + "\n")
        return print(md)
    if args.cmd == "all":
        return run_all(sys.executable)
    ap.error("pick a command: spike, pool, position or all")


if __name__ == "__main__":
    main()
