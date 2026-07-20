"""Emit tidy, reviewer-explorable per-token CSVs for every bridge experiment.
One row per token: identity + label + on_task + every cheap signal. No activations."""
from pathlib import Path

import polars as pl

BR = Path("results/bridge")
OUT = Path("paper_results/bridge")
OUT.mkdir(parents=True, exist_ok=True)

TAG = {"q7": "Qwen2.5-7B-Instruct", "g12": "gemma-3-12b-it",
       "g27": "gemma-3-27b-it", "l70": "Llama-3.3-70B-Instruct"}
HAND_SIG = ["surprisal", "entropy", "varentropy", "resid_jump", "temporal_kl",
            "lookback_ratio", "sink_drain", "head_disagreement", "w", "kl", "attn_rollout"]

SIG = {  # experiment -> (shorts, signals-source template, join key, signal cols)
    "hand": (["q7", "g12", "g27", "l70"], "TOKENS", ["case_id", "mode", "tok_idx"], HAND_SIG),
    "opi":  (["q7", "g12", "g27", "l70"], "opi_{s}_signals.parquet", ["position_id"],
             ["in_surprisal", "in_entropy", "in_attention"]),
    "liars":(["l70", "g27"], "liars_{s}_corpus.parquet", ["position_id"],
             ["surprisal", "entropy", "varentropy", "resid_jump",
              "lookback_ratio", "sink_drain", "head_disagreement"]),
    "tt":   (["q7", "g12", "g27", "l70"], "tt_{s}_corpus.parquet", ["position_id"],
             ["surprisal", "entropy", "varentropy", "resid_jump",
              "lookback_ratio", "sink_drain", "head_disagreement"]),
    "taboo":(["q7", "g12", "g27", "l70"], "taboo_{s}_corpus.parquet", ["position_id"],
             ["surprisal", "entropy", "varentropy", "resid_jump",
              "lookback_ratio", "sink_drain", "head_disagreement"]),
}

for kind, (shorts, tmpl, key, cols) in SIG.items():
    for s in shorts:
        pre = "" if kind == "hand" else f"{kind}_"
        ot = BR / f"{pre}{s}_ontask.parquet"
        if not ot.exists():
            continue
        df = pl.read_parquet(ot)
        src = (Path(f"results/token_selector_v2/{TAG[s]}/tokens.parquet")
               if tmpl == "TOKENS" else BR / tmpl.format(s=s) if tmpl else None)
        if src and src.exists():
            have = [c for c in cols if c in pl.scan_parquet(src).columns]
            right = pl.scan_parquet(src).select([*key, *have]).collect()
            df = df.join(right, on=key, how="left")
        out = OUT / f"{kind}_{s}.csv"
        df.write_csv(out)
        print(f"{out}  ({df.height} rows, {df.width} cols)")
