"""selector/spike_stats.py — find the spike tokens in an all-token corpus.

Sun et al., arXiv:2603.05498, show that a few channels in the intermediate
layers carry values orders of magnitude above the rest, that the affected
channels are fixed per model and essentially input independent, and that after
normalization the tokens carrying them collapse to nearly the same vector. They
call those channels implicit parameters. The NLA reads exactly this layer, so a
token whose activation is dominated by those channels hands the NLA something
close to a constant rather than a representation of the token.

The all-token corpora already store an activation per token, so this needs no
GPU and no forward pass. Per position it writes:

    spike_mass      share of the squared norm carried by the spike channels
    peak_ratio      largest |channel| over the root mean square: how spiky this
                    token is, without reference to which channels are spiky
    act_norm        the plain norm, kept because resid_jump is built from it
    resid_jump_masked
                    ||h_t - h_{t-1}|| with the spike channels dropped, the
                    repair for a signal whose norm the spike channels dominate

Output mirrors the other side tables, one shard per corpus shard, joined on
position_id by consolidate_all.py.

    python selector/spike_stats.py                 # every kind x model found
    python selector/spike_stats.py --kind tt --model q7
    python selector/spike_stats.py --selftest
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import polars as pl

BR = Path("results/bridge")
KINDS = {"hand": ["q7", "g12", "g27", "l70"], "opi": ["q7", "g12", "g27", "l70"],
         "tt": ["q7", "g12", "g27", "l70"], "taboo": ["q7", "g12", "g27", "l70"],
         "liars": ["g27", "l70"]}
# A channel counts as a spike channel when its typical magnitude stands this far
# above the typical channel. The paper reports separations of two to three
# orders of magnitude, so anything in the tens cleanly splits the two families.
SPIKE_FACTOR = 10.0
SAMPLE_ROWS = 20_000


def _acts(df: pl.DataFrame) -> np.ndarray:
    """The activation column as a [n, d] float array."""
    return np.asarray(df["activation"].to_list(), dtype=np.float32)


def find_spike_channels_from(acts: np.ndarray, factor: float = SPIKE_FACTOR) -> np.ndarray:
    """Channels whose typical magnitude towers over the typical channel.

    Uses medians, not means, so that a handful of extreme tokens cannot elect a
    channel on their own. The paper's claim that the set is input independent is
    what makes one sample over one corpus enough.
    """
    typical = np.median(np.abs(acts), axis=0)            # per channel
    return np.flatnonzero(typical > factor * np.median(typical))


def find_spike_channels(shards: list[Path], factor: float = SPIKE_FACTOR) -> np.ndarray:
    """find_spike_channels_from over a sample of the shards."""
    take = max(1, SAMPLE_ROWS // max(len(shards), 1))
    sample = np.concatenate([_acts(pl.read_parquet(s).head(take)) for s in shards])
    return find_spike_channels_from(sample, factor)


def stats(acts: np.ndarray, spike: np.ndarray) -> dict[str, np.ndarray]:
    """Per-token spike statistics for one shard."""
    sq = acts.astype(np.float64) ** 2
    total = sq.sum(axis=1)
    d = acts.shape[1]
    keep = np.setdiff1d(np.arange(d), spike)
    # consecutive rows inside a shard are consecutive tokens of a transcript;
    # the first row of each case has no predecessor, so its jump is undefined
    diff = np.diff(acts[:, keep].astype(np.float64), axis=0)
    jump = np.concatenate([[np.nan], np.linalg.norm(diff, axis=1)])
    return {
        "spike_mass": (sq[:, spike].sum(axis=1) / np.maximum(total, 1e-30)
                       if spike.size else np.zeros(len(acts))),
        "peak_ratio": np.abs(acts).max(axis=1) / np.sqrt(np.maximum(total / d, 1e-30)),
        "act_norm": np.sqrt(total),
        "resid_jump_masked": jump,
    }


def run_one(kind: str, short: str) -> None:
    src = BR / f"all_{kind}_{short}_corpus"
    shards = sorted(src.glob("*.parquet")) if src.is_dir() else []
    if not shards:
        return
    out = BR / f"all_{kind}_{short}_spike"
    out.mkdir(parents=True, exist_ok=True)
    spike = find_spike_channels(shards)
    print(f"{kind}/{short}: {len(shards)} shards, spike channels {spike.tolist() or 'none'}")
    for s in shards:
        dst = out / s.name
        if dst.exists():
            continue
        df = pl.read_parquet(s)
        cols = stats(_acts(df), spike)
        # a case boundary inside the shard also breaks the jump
        frame = df.select(pl.col("position_id").cast(pl.Int64), "case_id").with_columns(
            **{k: pl.Series(v) for k, v in cols.items()}
        )
        frame = frame.with_columns(
            pl.when(pl.col("case_id") != pl.col("case_id").shift(1))
            .then(None)
            .otherwise(pl.col("resid_jump_masked"))
            .alias("resid_jump_masked")
        ).drop("case_id")
        tmp = dst.with_suffix(".tmp.parquet")
        frame.write_parquet(tmp)
        tmp.replace(dst)


def _selftest() -> None:
    rng = np.random.default_rng(0)
    d, n = 64, 200
    acts = rng.normal(0, 1, (n, d)).astype(np.float32)
    acts[:, [7, 31]] *= 500.0                     # two planted spike channels
    spike = find_spike_channels_from(acts)
    assert spike.tolist() == [7, 31], f"spike channels: {spike.tolist()}"

    st = stats(acts, spike)
    assert st["spike_mass"].mean() > 0.9, "planted channels must dominate the norm"
    assert np.isnan(st["resid_jump_masked"][0]), "first token has no predecessor"
    # the repair: the plain jump is dominated by the spike channels, the masked
    # one is not, so they must disagree by a wide margin
    plain = np.linalg.norm(np.diff(acts.astype(np.float64), axis=0), axis=1)
    masked = st["resid_jump_masked"][1:]
    assert plain.mean() > 20 * masked.mean(), "masking must change resid_jump"

    flat = rng.normal(0, 1, (n, d)).astype(np.float32)
    assert find_spike_channels_from(flat).size == 0, "no spikes in a flat matrix"
    assert stats(flat, np.array([], int))["spike_mass"].max() == 0.0
    print("selftest ok")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--kind")
    ap.add_argument("--model")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()
    if args.selftest:
        return _selftest()
    for kind, shorts in KINDS.items():
        if args.kind and kind != args.kind:
            continue
        for s in shorts:
            if args.model and s != args.model:
                continue
            run_one(kind, s)


if __name__ == "__main__":
    main()
