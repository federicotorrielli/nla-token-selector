"""selector/bridge_extract_opi.py — bridge signals on real injection (OPI).

OpenPromptInjection puts the injected instruction inside a longer input, where the
interesting tokens are *sparse* — the honest test of "which token to point the NLA
at". For a sample of OPI cases this computes, per INPUT token, the same blind cheap
signals the response-side benchmarks use (surprisal, entropy, varentropy,
resid_jump, lookback_ratio, sink_drain, head_disagreement) plus the expensive
attention-rollout baseline, and the injected-span label — one forward pass, no
activation stored (the NLA explanations and judge labels are cached from the first
run, and the position ids here match them because the case order is deterministic).

For the attention signals the "context" boundary is the first input token (so
lookback = attention to the system/template prefix vs the input itself) and the
sink is the first N tokens, mirroring the response-side definitions.

Usage:
    python selector/bridge_extract_opi.py \
        --base-model Qwen/Qwen2.5-7B-Instruct --layer 20 \
        --cases selector/data/injection_cases.jsonl --max-cases 800 \
        --out results/bridge/opi_q7_signals.parquet
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parent))
from signals_deception import N_SINK, _attn_signals  # noqa: E402
from signals_injection import (  # noqa: E402
    _labels_for_view,
    _user_token_view,
    attention_rollout,
    load_cases,
)

SIGNALS = ["surprisal", "entropy", "varentropy", "resid_jump",
           "lookback_ratio", "sink_drain", "head_disagreement", "attn_rollout"]


def _schema() -> pa.Schema:
    fields = [
        ("position_id", pa.uint64()),
        ("case_id", pa.string()),
        ("mode", pa.string()),
        ("tok_idx", pa.int64()),
        ("token", pa.string()),
        ("label", pa.int64()),
    ]
    fields += [(s, pa.float32()) for s in SIGNALS]
    return pa.schema(fields)


def run(args: argparse.Namespace) -> int:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    allc = load_cases(Path(args.cases), None)
    # Keep the ORIGINAL run's ordering so position_ids line up with the cached
    # explanations/on-task labels: same deterministic sample, same iteration.
    if args.max_cases and len(allc) > args.max_cases:
        import random
        cases = random.Random(0).sample(allc, args.max_cases)
    else:
        cases = allc
    print(f"loaded {len(cases)} OPI cases (of {len(allc)})", flush=True)

    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16}[args.dtype]
    load_kw = dict(device_map="auto", dtype=dtype, attn_implementation="eager")
    try:
        model = AutoModelForCausalLM.from_pretrained(args.base_model, **load_kw).eval()
    except (ValueError, KeyError):
        from transformers import AutoModelForImageTextToText
        model = AutoModelForImageTextToText.from_pretrained(args.base_model, **load_kw).eval()
    device = str(model.get_input_embeddings().weight.device)

    rows = []
    position_id = 0
    for ci, c in enumerate(cases):
        try:
            ids, view = _user_token_view(tokenizer, c.system, c.user_full)
        except (AssertionError, ValueError):
            continue
        if not view:
            continue
        labels = _labels_for_view(view, c.inj_start, c.inj_end)
        want_attn = len(ids) <= args.attn_max_len
        t = torch.tensor([ids], device=device)
        with torch.no_grad():
            out = model(t, output_attentions=want_attn, output_hidden_states=True)
        logits = out.logits[0]
        hs = out.hidden_states[-1][0].float()
        toks = tokenizer.convert_ids_to_tokens(ids)

        idx = [ti for (ti, _cs, _ce) in view]
        r = torch.tensor(idx, device=device)
        rows1 = logits.index_select(0, r - 1).float()           # predict token at ti
        logp = torch.log_softmax(rows1, dim=-1)
        prob = logp.exp()
        tgt = torch.tensor([ids[ti] for ti in idx], device=device)
        surp = (-logp.gather(1, tgt.view(-1, 1)).squeeze(1)).cpu().numpy()
        ent = (-(prob * logp).sum(-1)).cpu().numpy()
        varent = ((prob * (-logp - (-(prob * logp).sum(-1)).unsqueeze(1)) ** 2).sum(-1)).cpu().numpy()
        rjump = (hs.index_select(0, r) - hs.index_select(0, r - 1)).norm(dim=-1).cpu().numpy()

        a_bound = idx[0]                     # first input token = context boundary
        lb = sk = hd = roll_v = np.full(len(idx), np.nan)
        if want_attn and out.attentions is not None:
            sink_n = min(N_SINK, a_bound)
            n_layers = len(out.attentions)
            lb_t, snk_t, hd_t = _attn_signals([a[0] for a in out.attentions], r, a_bound, sink_n)
            lb = lb_t.cpu().numpy()
            sk = (-(snk_t / n_layers)).cpu().numpy() if sink_n > 0 else np.full(len(idx), np.nan)
            hd = hd_t.cpu().numpy()
            attns_np = [a[0].float().cpu().numpy() for a in out.attentions]
            roll = attention_rollout(attns_np)
            q = len(ids) - 1
            roll_v = np.array([roll[q, ti] for ti in idx])
        del out

        for k, ti in enumerate(idx):
            rows.append({
                "position_id": np.uint64(position_id),
                "case_id": c.id, "mode": "injection_input",
                "tok_idx": np.int64(k), "token": toks[ti],
                "label": np.int64(int(labels[k])),
                "surprisal": np.float32(surp[k]), "entropy": np.float32(ent[k]),
                "varentropy": np.float32(varent[k]), "resid_jump": np.float32(rjump[k]),
                "lookback_ratio": np.float32(lb[k]), "sink_drain": np.float32(sk[k]),
                "head_disagreement": np.float32(hd[k]), "attn_rollout": np.float32(roll_v[k]),
            })
            position_id += 1
        if (ci + 1) % 100 == 0:
            print(f"  {ci + 1}/{len(cases)}  ({len(rows)} tokens)", flush=True)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(rows, schema=_schema()), out_path)
    n_pos = sum(int(r["label"]) for r in rows)
    print(f"wrote {out_path}  ({len(rows)} input tokens, {n_pos} in an injected span, "
          f"{n_pos / max(len(rows), 1):.1%} positive)")
    return 0


def _parse_args(argv):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--base-model", default="Qwen/Qwen2.5-7B-Instruct")
    p.add_argument("--layer", type=int, default=20, help="unused; kept for CLI parity")
    p.add_argument("--cases",
                   default=str(Path(__file__).parent / "data" / "injection_cases.jsonl"))
    p.add_argument("--max-cases", type=int, default=800)
    p.add_argument("--attn-max-len", type=int, default=4096)
    p.add_argument("--out", default="results/bridge/opi_q7_signals.parquet")
    p.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16"])
    return p.parse_args(argv)


if __name__ == "__main__":
    raise SystemExit(run(_parse_args(sys.argv[1:])))
