"""selector/bridge_extract_taboo.py — taboo secret-word bridge, phase 1.

For one park base, load the three taboo LoRA adapters (moon/ship/snow) on the
vanilla base, and for each adapter × elicitation prompt generate a few hint replies,
then capture per response-token: the NLA-layer activation and the blind signals. The
secret word is recorded in the case id (`<word>__p<prompt>__s<sample>`) so the judge
can ask, per token, whether the NLA explanation recovers that word. One corpus per
model covering all three words.

The base is loaded once; adapters are swapped with PEFT `set_adapter`, and the layer
hook is registered on the (persistent) decoder-layer object before wrapping, so it
fires through whichever adapter is active.

Usage:
    python selector/bridge_extract_taboo.py --short q7 --layer 20 --d-model 3584 \
        --samples 8 --out results/bridge/taboo_q7_corpus.parquet
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent / "data"))
from _response_extract import (  # noqa: E402
    build_messages,
    extract_rows,
    generate_reply,
    load_base,
    make_hook,
    schema,
)
from taboo import PROMPTS, WORDS, adapter_id, base_model  # noqa: E402


def run(args: argparse.Namespace) -> int:
    from peft import PeftModel

    words = args.words or list(WORDS)
    base = base_model(args.short)
    print(f"taboo {args.short}: base {base}, words {words}", flush=True)

    # plain base; hook the decoder layer BEFORE wrapping so it fires through any adapter
    model, tokenizer, device = load_base(base, args.dtype)
    handle, captured = make_hook(model, args.layer)

    peft = PeftModel.from_pretrained(model, adapter_id(args.short, words[0]), adapter_name=words[0])
    for w in words[1:]:
        peft.load_adapter(adapter_id(args.short, w), adapter_name=w)

    out_rows = []
    position_id = 0
    for w in words:
        peft.set_adapter(w)
        for pj, prompt in enumerate(PROMPTS):
            for s in range(args.samples):
                msgs = build_messages(tokenizer, None, prompt)
                try:
                    reply = generate_reply(peft, tokenizer, device, msgs, args.max_new_tokens,
                                           do_sample=(s > 0), temperature=args.temperature)
                except Exception as e:  # noqa: BLE001 — one bad case must not kill the run
                    print(f"  gen fail {w} p{pj} s{s}: {e}", flush=True)
                    continue
                if not reply:
                    continue
                msgs = [*msgs, {"role": "assistant", "content": reply}]
                rows, position_id = extract_rows(
                    peft, tokenizer, device, captured, msgs,
                    case_id=f"{w}__p{pj}__s{s}", label=1, mode="taboo",
                    d_model=args.d_model, tok_cap=args.tok_cap, attn_max_len=args.attn_max_len,
                    position_id=position_id)
                out_rows.extend(rows)
        print(f"  {w}: {len(out_rows)} tokens so far", flush=True)
    handle.remove()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(out_rows, schema=schema(args.d_model)), out_path)
    print(f"wrote {out_path}  ({len(out_rows)} response tokens over {len(words)} words)")
    return 0


def _parse_args(argv):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--short", default="q7", choices=["q7", "g12", "g27", "l70"])
    p.add_argument("--layer", type=int, default=20)
    p.add_argument("--d-model", type=int, default=3584)
    p.add_argument("--words", nargs="+", default=None, help="subset of moon ship snow")
    p.add_argument("--samples", type=int, default=8, help="replies per (word, prompt); s0 greedy")
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--max-new-tokens", type=int, default=48)
    p.add_argument("--tok-cap", type=int, default=30)
    p.add_argument("--attn-max-len", type=int, default=2600)
    p.add_argument("--out", default="results/bridge/taboo_q7_corpus.parquet")
    p.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16"])
    return p.parse_args(argv)


if __name__ == "__main__":
    raise SystemExit(run(_parse_args(sys.argv[1:])))
