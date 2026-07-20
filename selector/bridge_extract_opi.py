"""selector/bridge_extract_opi.py — bridge test on real injection (OPI).

The 27 hand injection cases were wholly compromised (the whole reply executes the
attack), so selection could not bite. OpenPromptInjection puts the injected
instruction inside a longer input, where the interesting tokens are *sparse* —
the honest test of "which token to point the NLA at".

For a sample of OPI cases this captures, per INPUT token: the NLA-layer activation
(so we can run the NLA on it), the injected-span label, and the cheap input-token
signals (in_surprisal, in_entropy, in_attention) in one forward pass. Output is a
corpus parquet in the same shape bridge_run_nla.py reads, plus the signals for the
analysis join.

Usage:
    python selector/bridge_extract_opi.py \
        --base-model Qwen/Qwen2.5-7B-Instruct --layer 20 --d-model 3584 \
        --cases selector/data/injection_cases.jsonl --max-cases 100 \
        --out results/bridge/opi_q7_corpus.parquet
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from nla_token_selector.layers import _resolve_layer_module

sys.path.insert(0, str(Path(__file__).resolve().parent))
from signals_injection import (  # noqa: E402
    _labels_for_view,
    _log_softmax,
    _user_token_view,
    attention_rollout,
    entropy_varentropy,
    load_cases,
)


def _schema(d_model: int) -> pa.Schema:
    return pa.schema([
        ("position_id", pa.uint64()),
        ("activation", pa.list_(pa.float32(), d_model)),
        ("case_id", pa.string()),
        ("mode", pa.string()),
        ("tok_idx", pa.int64()),
        ("token", pa.string()),
        ("label", pa.int64()),
        ("in_surprisal", pa.float32()),
        ("in_entropy", pa.float32()),
        ("in_attention", pa.float32()),
    ])


def run(args: argparse.Namespace) -> int:
    import random

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    allc = load_cases(Path(args.cases), None)
    if args.max_cases and len(allc) > args.max_cases:
        # sample across the whole file (both targets, all strategies), not the
        # first N (which are all one dense target), so positive density is
        # representative and includes the sparse mrpc-target cases.
        cases = random.Random(0).sample(allc, args.max_cases)
    else:
        cases = allc
    print(f"loaded {len(cases)} OPI cases (of {len(allc)})")

    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16}[args.dtype]
    load_kw = dict(device_map="auto", dtype=dtype, attn_implementation="eager")
    try:
        model = AutoModelForCausalLM.from_pretrained(args.base_model, **load_kw).eval()
    except (ValueError, KeyError):
        from transformers import AutoModelForImageTextToText
        model = AutoModelForImageTextToText.from_pretrained(args.base_model, **load_kw).eval()
    device = str(model.get_input_embeddings().weight.device)

    target = _resolve_layer_module(model, args.layer)
    captured: dict[str, object] = {}

    def hook(_m, _i, output):
        captured["h"] = (output[0] if isinstance(output, tuple) else output).detach()

    handle = target.register_forward_hook(hook)

    rows = []
    position_id = 0
    for c in cases:
        try:
            ids, view = _user_token_view(tokenizer, c.system, c.user_full)
        except (AssertionError, ValueError):
            continue
        if not view:
            continue
        labels = _labels_for_view(view, c.inj_start, c.inj_end)
        t = torch.tensor([ids], device=device)
        with torch.no_grad():
            captured.clear()
            out = model(t, output_attentions=True)
        logits = out.logits[0].float().cpu().numpy()
        h = captured["h"][0].to(torch.float32).cpu().numpy()  # [seq, d]
        attns = [a[0].float().cpu().numpy() for a in out.attentions] if out.attentions else None
        roll = attention_rollout(attns) if attns is not None else None
        q = len(ids) - 1
        toks = tokenizer.convert_ids_to_tokens(ids)
        for k, (ti, _cs, _ce) in enumerate(view):
            vec = h[ti]
            if vec.shape != (args.d_model,):
                raise SystemExit(f"activation shape {vec.shape} != ({args.d_model},)")
            row = logits[ti - 1]
            lp = _log_softmax(row)
            ent, _ = entropy_varentropy(row)
            rows.append({
                "position_id": np.uint64(position_id),
                "activation": vec.tolist(),
                "case_id": c.id, "mode": "injection_input",
                "tok_idx": np.int64(k), "token": toks[ti],
                "label": np.int64(int(labels[k])),
                "in_surprisal": np.float32(-lp[ids[ti]]),
                "in_entropy": np.float32(-ent),  # low entropy = more interesting
                "in_attention": np.float32(roll[q, ti] if roll is not None else np.nan),
            })
            position_id += 1
    handle.remove()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(rows, schema=_schema(args.d_model)), out_path)
    n_pos = sum(r["label"] for r in rows)
    print(f"wrote {out_path}  ({len(rows)} input tokens, {n_pos} in an injected span, "
          f"{n_pos / max(len(rows), 1):.1%} positive)")
    return 0


def _parse_args(argv):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--base-model", default="Qwen/Qwen2.5-7B-Instruct")
    p.add_argument("--layer", type=int, default=20)
    p.add_argument("--d-model", type=int, default=3584)
    p.add_argument("--cases",
                   default=str(Path(__file__).parent / "data" / "injection_cases.jsonl"))
    p.add_argument("--max-cases", type=int, default=100)
    p.add_argument("--out", default="results/bridge/opi_q7_corpus.parquet")
    p.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16"])
    return p.parse_args(argv)


if __name__ == "__main__":
    raise SystemExit(run(_parse_args(sys.argv[1:])))
