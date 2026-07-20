"""scripts/token_selector/bridge_extract_activations.py — bridge test, phase 1.

The token selector asks "which token should I point the NLA at?". To answer it
we need, for each token, both (a) the cheap pre-pass signals — already saved by
signals.py in tokens.parquet — and (b) the NLA's explanation of that
token, which needs the base model's hidden state at the NLA's layer.

This script produces (b)'s input: it runs the base model over the same labelled
cases with the same tokenization as signals, captures the residual
hidden state at the NLA's layer for every assistant token, and writes a corpus
parquet in the schema NLAClientLite reads (position_id + activation), plus the
join keys (case_id, mode, tok_idx, label) so the NLA explanations and the cheap
signals line up one-for-one by (case_id, tok_idx).

Same activation convention as scripts/pipeline/build_corpus.py: forward hook on
`model.model.layers[LAYER]`, capture output[0] (the raw, unnormalised residual
stream the AV was trained on), bf16 forward, f32 storage.

Usage:
    python scripts/token_selector/bridge_extract_activations.py \
        --base-model Qwen/Qwen2.5-7B-Instruct --layer 20 --d-model 3584 \
        --cases scripts/token_selector/data/token_selector_cases.json \
        --out results/bridge/q7_corpus.parquet
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parent))
from nla_token_selector.layers import _resolve_layer_module  # noqa: E402
from signals import _assistant_token_labels, _build_ids, load_cases  # noqa: E402


def _schema(d_model: int) -> pa.Schema:
    return pa.schema([
        ("position_id", pa.uint64()),
        ("activation", pa.list_(pa.float32(), d_model)),
        ("case_id", pa.string()),
        ("mode", pa.string()),
        ("tok_idx", pa.int64()),
        ("token", pa.string()),
        ("label", pa.int64()),
    ])


def run(args: argparse.Namespace) -> int:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    cases = load_cases(Path(args.cases))
    print(f"loaded {len(cases)} cases "
          f"({sum(c.mode == 'injection' for c in cases)} injection, "
          f"{sum(c.mode == 'evalaware' for c in cases)} evalaware)")

    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16}[args.dtype]
    load_kw = dict(device_map="auto", dtype=dtype)
    try:
        model = AutoModelForCausalLM.from_pretrained(args.base_model, **load_kw).eval()
    except (ValueError, KeyError):
        from transformers import AutoModelForImageTextToText
        model = AutoModelForImageTextToText.from_pretrained(args.base_model, **load_kw).eval()

    target = _resolve_layer_module(model, args.layer)
    captured: dict[str, torch.Tensor] = {}

    def hook(_m, _i, output):
        captured["h"] = (output[0] if isinstance(output, tuple) else output).detach()

    handle = target.register_forward_hook(hook)

    rows = []
    position_id = 0
    for c in cases:
        asst_ids, labels = _assistant_token_labels(tokenizer, c.assistant, c.label_spans)
        pre = _build_ids(tokenizer, c.system, c.user_full)
        full = pre + asst_ids
        a = len(pre)
        toks = tokenizer.convert_ids_to_tokens(asst_ids)
        with torch.no_grad():
            captured.clear()
            model(torch.tensor([full], device=model.get_input_embeddings().weight.device))
        h = captured["h"][0].to(torch.float32).cpu().numpy()  # [seq, d]
        for j in range(len(asst_ids)):
            vec = h[a + j]
            assert vec.shape == (args.d_model,), (
                f"{c.id}[{j}] activation shape {vec.shape} != ({args.d_model},); "
                f"wrong layer path or d_model."
            )
            rows.append({
                "position_id": np.uint64(position_id),
                "activation": vec.tolist(),
                "case_id": c.id,
                "mode": c.mode,
                "tok_idx": np.int64(j),
                "token": toks[j],
                "label": np.int64(int(labels[j])),
            })
            position_id += 1
    handle.remove()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(rows, schema=_schema(args.d_model)), out)
    n_pos = sum(r["label"] for r in rows)
    print(f"wrote {out}  ({len(rows)} tokens, {n_pos} in a planted span)")
    return 0


def _parse_args(argv):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--base-model", default="Qwen/Qwen2.5-7B-Instruct")
    p.add_argument("--layer", type=int, default=20)
    p.add_argument("--d-model", type=int, default=3584)
    p.add_argument("--cases", default=str(Path(__file__).parent / "data" / "token_selector_cases.json"))
    p.add_argument("--out", default="results/bridge/q7_corpus.parquet")
    p.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16"])
    return p.parse_args(argv)


if __name__ == "__main__":
    raise SystemExit(run(_parse_args(sys.argv[1:])))
