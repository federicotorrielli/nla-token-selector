"""selector/bridge_extract_liars.py — bridge test on real lies.

Liars' Bench deception, response side. For a balanced sample of on-policy
transcripts (the base model probed is the one that generated the text), this
captures, per RESPONSE token of the last assistant message: the NLA-layer
activation, the response-level deceptive label, and the cheap blind signals
(surprisal, entropy, varentropy, resid_jump, lookback_ratio, sink_drain,
head_disagreement) in one forward pass. Output is a corpus parquet in the shape
bridge_run_nla.py reads, plus the signals for the analysis join. Response tokens
are capped per transcript to bound the NLA decode cost.

Sample size per class is set by the 95%-CI calculation in bridge-experiment-design.md
(1000 lying + 1000 honest per model gives a +/-0.03 interval).

Usage:
    python selector/bridge_extract_liars.py \
        --base-model meta-llama/Llama-3.3-70B-Instruct --model-tag llama-v3.3-70b-instruct \
        --layer 53 --d-model 8192 --datasets instructed-deception insider-trading \
        --per-class 1000 --tok-cap 30 --out results/bridge/liars_l70_corpus.parquet
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
from signals_deception import (  # noqa: E402
    HF_REPO,
    N_SINK,
    _attn_signals,
    _response_view,
)

SIGNALS = ["surprisal", "entropy", "varentropy", "resid_jump",
           "lookback_ratio", "sink_drain", "head_disagreement"]


def _schema(d_model: int) -> pa.Schema:
    fields = [
        ("position_id", pa.uint64()),
        ("activation", pa.list_(pa.float32(), d_model)),
        ("case_id", pa.string()),
        ("mode", pa.string()),
        ("tok_idx", pa.int64()),
        ("token", pa.string()),
        ("label", pa.int64()),
    ]
    fields += [(s, pa.float32()) for s in SIGNALS]
    return pa.schema(fields)


def _sample(datasets, model_tag, per_class, seed):
    from datasets import load_dataset
    rng = np.random.default_rng(seed)
    dec, hon = [], []
    for ds in datasets:
        d = load_dataset(HF_REPO, ds, split="test").filter(lambda r: r["model"] == model_tag)
        for r in d:
            (dec if r["deceptive"] else hon).append(
                {"messages": r["messages"], "deceptive": bool(r["deceptive"]),
                 "id": f"{ds}_{r.get('index', len(dec) + len(hon))}"})
    def take(pool):
        if len(pool) <= per_class:
            return pool
        idx = rng.choice(len(pool), per_class, replace=False)
        return [pool[i] for i in idx]
    return take(dec) + take(hon)


def run(args: argparse.Namespace) -> int:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    rows_in = _sample(args.datasets, args.model_tag, args.per_class, args.seed)
    print(f"sampled {len(rows_in)} transcripts "
          f"({sum(r['deceptive'] for r in rows_in)} lying)")

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

    out_rows = []
    position_id = 0
    for c in rows_in:
        try:
            ids, resp, a_resp = _response_view(tokenizer, c["messages"])
        except (ValueError, AssertionError):
            continue
        if args.tok_cap and len(resp) > args.tok_cap:
            resp = resp[: args.tok_cap]
        want_attn = len(ids) <= args.attn_max_len
        t = torch.tensor([ids], device=device)
        with torch.no_grad():
            captured.clear()
            out = model(t, output_attentions=want_attn, output_hidden_states=True)
        logits = out.logits[0]
        hs = out.hidden_states[-1][0].float()
        h_layer = captured["h"][0].to(torch.float32).cpu().numpy()  # NLA-layer [seq, d]
        toks = tokenizer.convert_ids_to_tokens(ids)

        r = torch.tensor(resp, device=device)
        rows1 = logits.index_select(0, r - 1).float()
        logp = torch.log_softmax(rows1, dim=-1)
        prob = logp.exp()
        tgt = torch.tensor([ids[p] for p in resp], device=device)
        surp = (-logp.gather(1, tgt.view(-1, 1)).squeeze(1)).cpu().numpy()
        ent = (-(prob * logp).sum(-1)).cpu().numpy()
        varent = ((prob * (-logp - (-(prob * logp).sum(-1)).unsqueeze(1)) ** 2).sum(-1)).cpu().numpy()
        rjump = (hs.index_select(0, r) - hs.index_select(0, r - 1)).norm(dim=-1).cpu().numpy()

        lb = sk = hd = np.full(len(resp), np.nan)
        if want_attn and out.attentions is not None:
            sink_n = min(N_SINK, a_resp)
            n_layers = len(out.attentions)
            lb_t, snk_t, hd_t = _attn_signals([a[0] for a in out.attentions], r, a_resp, sink_n)
            lb = lb_t.cpu().numpy()
            sk = (-(snk_t / n_layers)).cpu().numpy() if sink_n > 0 else np.full(len(resp), np.nan)
            hd = hd_t.cpu().numpy()
        del out

        for k, p in enumerate(resp):
            vec = h_layer[p]
            if vec.shape != (args.d_model,):
                raise SystemExit(f"activation shape {vec.shape} != ({args.d_model},)")
            out_rows.append({
                "position_id": np.uint64(position_id),
                "activation": vec.tolist(),
                "case_id": c["id"], "mode": "deception",
                "tok_idx": np.int64(k), "token": toks[p],
                "label": np.int64(int(c["deceptive"])),
                "surprisal": np.float32(surp[k]), "entropy": np.float32(ent[k]),
                "varentropy": np.float32(varent[k]), "resid_jump": np.float32(rjump[k]),
                "lookback_ratio": np.float32(lb[k]), "sink_drain": np.float32(sk[k]),
                "head_disagreement": np.float32(hd[k]),
            })
            position_id += 1
    handle.remove()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(out_rows, schema=_schema(args.d_model)), out_path)
    n_lie = sum(r["label"] for r in out_rows)
    print(f"wrote {out_path}  ({len(out_rows)} response tokens, {n_lie} from lying responses)")
    return 0


def _parse_args(argv):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--base-model", default="meta-llama/Llama-3.3-70B-Instruct")
    p.add_argument("--model-tag", default="llama-v3.3-70b-instruct")
    p.add_argument("--layer", type=int, default=53)
    p.add_argument("--d-model", type=int, default=8192)
    p.add_argument("--datasets", nargs="+",
                   default=["instructed-deception", "insider-trading", "convincing-game"])
    p.add_argument("--per-class", type=int, default=1000)
    p.add_argument("--tok-cap", type=int, default=30)
    p.add_argument("--attn-max-len", type=int, default=2600)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default="results/bridge/liars_l70_corpus.parquet")
    p.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16"])
    return p.parse_args(argv)


if __name__ == "__main__":
    raise SystemExit(run(_parse_args(sys.argv[1:])))
