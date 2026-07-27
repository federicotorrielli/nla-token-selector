"""selector/seed_from_cache.py — reuse cached NLA work for the all-token bridge.

Greedy NLA decode is a deterministic function of the activation, and the tokens
the ORIGINAL bridge probed have unchanged activations in the all-token corpus
(identical prefixes). So the old explanations and judge labels transfer: join
old (case_id, tok_idx) to new (case_id, probe_tok_idx), rewrite position_id to
the new numbering, and drop the rows into a `shard-cache.parquet` inside the
all-token explanations / ontask directories. bridge_run_nla.py and
bridge_judge_ontask.py treat any position present in the output directory as
done, so the seeded rows are never re-decoded or re-judged.

Usage:
    python selector/seed_from_cache.py \
        --new-corpus results/bridge/all_tt_q7_corpus \
        --old-corpus results/bridge/tt_q7_corpus.parquet \
        --old-explanations results/bridge/tt_q7_explanations.parquet \
        --old-ontask results/bridge/tt_q7_ontask.parquet \
        --out-explanations results/bridge/all_tt_q7_explanations \
        --out-ontask results/bridge/all_tt_q7_ontask
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import polars as pl

NEW_META = ["position_id", "case_id", "mode", "tok_idx", "token", "label",
            "probe_tok_idx"]


def _new_probed(new_corpus: Path) -> pl.DataFrame:
    """Probed rows of the all-token corpus, meta columns only (no activations)."""
    lf = pl.scan_parquet(str(new_corpus / "shard-*.parquet"))
    return lf.select(NEW_META).filter(pl.col("probe_tok_idx") >= 0).collect()


def _old_key(old_corpus: Path) -> pl.DataFrame:
    """old position_id -> (case_id, old tok_idx)."""
    return (pl.scan_parquet(str(old_corpus))
            .select(["position_id", "case_id", "tok_idx"]).collect()
            .rename({"position_id": "old_position_id", "tok_idx": "probe_tok_idx"}))


def _seed(new_probed: pl.DataFrame, old_key: pl.DataFrame, old_file: Path,
          meta_cols: list[str], value_cols: list[str],
          out_dir: Path) -> tuple[int, int]:
    old = (pl.scan_parquet(str(old_file))
           .select(["position_id", *value_cols]).collect()
           .rename({"position_id": "old_position_id"})
           .unique(subset=["old_position_id"], keep="first"))
    keyed = old.join(old_key, on="old_position_id", how="inner")
    seeded = new_probed.join(keyed.select(["case_id", "probe_tok_idx", *value_cols]),
                             on=["case_id", "probe_tok_idx"], how="inner")
    out = (seeded.select([*meta_cols, *value_cols])
           # match the Int64 the decode/judge writers produce, so one directory
           # scans with one schema
           .with_columns(pl.col("position_id").cast(pl.Int64)))
    out_dir.mkdir(parents=True, exist_ok=True)
    tmp = out_dir / "shard-cache.tmp.parquet"
    out.write_parquet(tmp)
    tmp.replace(out_dir / "shard-cache.parquet")
    return out.height, old.height


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--new-corpus", required=True, help="all-token corpus DIRECTORY")
    ap.add_argument("--old-corpus", required=True, help="original corpus parquet")
    ap.add_argument("--old-explanations", default=None)
    ap.add_argument("--old-ontask", default=None)
    ap.add_argument("--out-explanations", default=None, help="all-token explanations dir")
    ap.add_argument("--out-ontask", default=None, help="all-token ontask dir")
    args = ap.parse_args(argv)

    new_probed = _new_probed(Path(args.new_corpus))
    old_key = _old_key(Path(args.old_corpus))
    n_total = (pl.scan_parquet(str(Path(args.new_corpus) / "shard-*.parquet"))
               .select(pl.len()).collect().item())
    print(f"new corpus: {n_total} tokens, {new_probed.height} previously probed")

    # Schemas must match what bridge_run_nla.py / bridge_judge_ontask.py write,
    # so every shard in one directory scans with one schema.
    expl_meta = ["position_id", "case_id", "mode", "tok_idx", "token", "label"]
    ontask_meta = ["position_id", "case_id", "mode", "tok_idx", "label"]
    for old_file, out_dir, meta, cols, name in [
        (args.old_explanations, args.out_explanations, expl_meta,
         ["explanation"], "explanations"),
        (args.old_ontask, args.out_ontask, ontask_meta, ["on_task"], "ontask"),
    ]:
        if not old_file or not out_dir:
            continue
        if not Path(old_file).exists():
            print(f"  {name}: {old_file} missing, skipped", file=sys.stderr)
            continue
        n_seed, n_old = _seed(new_probed, old_key, Path(old_file), meta, cols,
                              Path(out_dir))
        print(f"  {name}: seeded {n_seed}/{n_old} cached rows -> {out_dir}")
    print(f"  remaining to decode: {n_total - new_probed.height} new tokens "
          f"(plus any probed rows without a cached row)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
