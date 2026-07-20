"""selector/bridge_extract_tt.py — Tensor Trust injection bridge, phase 1.

For each Tensor Trust hijacking case (attack and access_code variants, built by
data/build_tensortrust_cases.py), the base model generates a reply, then we capture
per response-token: the NLA-layer activation, the blind cheap signals, and the
variant label (attack=1 / access_code=0). Output is a corpus parquet in the shape
bridge_run_nla.py reads. Response tokens are capped to bound the NLA decode cost.

Usage:
    python selector/bridge_extract_tt.py \
        --base-model Qwen/Qwen2.5-7B-Instruct --layer 20 --d-model 3584 \
        --cases selector/data/tensortrust_cases.jsonl \
        --out results/bridge/tt_q7_corpus.parquet
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _response_extract import (  # noqa: E402
    build_messages,
    extract_rows,
    generate_reply,
    load_base,
    make_hook,
    schema,
)


def run(args: argparse.Namespace) -> int:
    cases = [json.loads(ln) for ln in Path(args.cases).read_text().splitlines() if ln.strip()]
    if args.limit:
        cases = cases[: args.limit]
    print(f"loaded {len(cases)} Tensor Trust cases", flush=True)

    model, tokenizer, device = load_base(args.base_model, args.dtype)
    handle, captured = make_hook(model, args.layer)

    out_rows = []
    position_id = 0
    for i, c in enumerate(cases):
        msgs = build_messages(tokenizer, c["system"], c["user"])
        try:
            reply = generate_reply(model, tokenizer, device, msgs, args.max_new_tokens)
        except Exception as e:  # noqa: BLE001 — one bad case must not kill the run
            print(f"  gen fail {c['id']}: {e}", flush=True)
            continue
        if not reply:
            continue
        msgs = [*msgs, {"role": "assistant", "content": reply}]
        rows, position_id = extract_rows(
            model, tokenizer, device, captured, msgs,
            case_id=c["id"], label=c["label"], mode="injection",
            d_model=args.d_model, tok_cap=args.tok_cap, attn_max_len=args.attn_max_len,
            position_id=position_id)
        out_rows.extend(rows)
        if (i + 1) % 100 == 0:
            print(f"  {i + 1}/{len(cases)}  ({len(out_rows)} tokens)", flush=True)
    handle.remove()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(out_rows, schema=schema(args.d_model)), out_path)
    n_attack = sum(r["label"] for r in out_rows)
    print(f"wrote {out_path}  ({len(out_rows)} response tokens, {n_attack} from attack variants)")
    return 0


def _parse_args(argv):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--base-model", default="Qwen/Qwen2.5-7B-Instruct")
    p.add_argument("--layer", type=int, default=20)
    p.add_argument("--d-model", type=int, default=3584)
    p.add_argument("--cases", default=str(Path(__file__).parent / "data" / "tensortrust_cases.jsonl"))
    p.add_argument("--max-new-tokens", type=int, default=64)
    p.add_argument("--tok-cap", type=int, default=30)
    p.add_argument("--attn-max-len", type=int, default=2600)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--out", default="results/bridge/tt_q7_corpus.parquet")
    p.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16"])
    return p.parse_args(argv)


if __name__ == "__main__":
    raise SystemExit(run(_parse_args(sys.argv[1:])))
