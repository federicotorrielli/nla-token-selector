"""selector/_response_extract.py — shared response-side corpus extraction.

Given a chat transcript whose LAST message is an assistant reply, forward the base
model once and emit, per response token: the NLA-layer activation, the blind cheap
signals, and a per-case label. The Tensor Trust and taboo extractors GENERATE that
reply first (this module has the generate helper); the Liars' extractor uses replies
that already exist in its dataset. One place for the forward + signal maths so the
three benchmarks cannot drift apart.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pyarrow as pa

sys.path.insert(0, str(Path(__file__).resolve().parent))
from signals_deception import N_SINK, _attn_signals, _response_view  # noqa: E402

SIGNALS = ["surprisal", "entropy", "varentropy", "resid_jump",
           "lookback_ratio", "sink_drain", "head_disagreement"]


def load_resume(out_path):
    """Resume support: (existing_rows, done_case_ids, next_position_id). A killed
    extract restarts from the last checkpoint instead of from zero."""
    import polars as pl

    p = Path(out_path)
    if not p.exists():
        return [], set(), 0
    prev = pl.read_parquet(p)
    rows = prev.to_dicts()
    done = set(prev["case_id"].to_list())
    nxt = (max(int(r["position_id"]) for r in rows) + 1) if rows else 0
    return rows, done, nxt


def write_corpus(out_path, rows, d_model):
    """Atomic parquet write (tmp then replace), so a checkpoint can't truncate."""
    import os

    import pyarrow.parquet as pq

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_path.with_suffix(".tmp.parquet")
    pq.write_table(pa.Table.from_pylist(rows, schema=schema(d_model)), tmp)
    os.replace(tmp, out_path)


def schema(d_model: int) -> pa.Schema:
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


def load_base(base_model: str, dtype_name: str = "bfloat16", adapter: str | None = None):
    """Load the base model (+ optional LoRA adapter) in eager attention for signals."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(base_model)
    dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16}[dtype_name]
    load_kw = dict(device_map="auto", dtype=dtype, attn_implementation="eager")
    try:
        model = AutoModelForCausalLM.from_pretrained(base_model, **load_kw).eval()
    except (ValueError, KeyError):
        from transformers import AutoModelForImageTextToText
        model = AutoModelForImageTextToText.from_pretrained(base_model, **load_kw).eval()
    if adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, adapter).eval()
    device = str(model.get_input_embeddings().weight.device)
    return model, tokenizer, device


def build_messages(tokenizer, system: str | None, user: str) -> list[dict]:
    """Message list the tokenizer's chat template accepts. Folds `system` into the
    user turn for templates that reject a system role (e.g. Gemma)."""
    if not system:
        return [{"role": "user", "content": user}]
    msgs = [{"role": "system", "content": system}, {"role": "user", "content": user}]
    try:
        tokenizer.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)
        return msgs
    except Exception:
        return [{"role": "user", "content": f"{system}\n\n{user}"}]


def generate_reply(model, tokenizer, device, messages: list[dict], max_new_tokens: int = 64,
                   do_sample: bool = False, temperature: float = 1.0) -> str:
    """Base-model reply to `messages` (which must end at the user turn). Greedy by
    default; set do_sample for a temperature sample (used to get several transcripts
    per taboo prompt)."""
    import torch

    # transformers 5.x apply_chat_template returns a BatchEncoding, not a tensor;
    # generate(**enc) also passes the attention mask.
    enc = tokenizer.apply_chat_template(messages, add_generation_prompt=True,
                                        return_tensors="pt", return_dict=True).to(device)
    n_in = enc["input_ids"].shape[1]
    kw = dict(max_new_tokens=max_new_tokens, pad_token_id=tokenizer.eos_token_id)
    kw.update(dict(do_sample=True, temperature=temperature) if do_sample else dict(do_sample=False))
    with torch.no_grad():
        gen = model.generate(**enc, **kw)
    return tokenizer.decode(gen[0][n_in:], skip_special_tokens=True).strip()


def make_hook(model, layer: int):
    """Register a forward hook on the NLA layer; returns (handle, captured-dict)."""
    from nla_token_selector.layers import _resolve_layer_module

    target = _resolve_layer_module(model, layer)
    captured: dict[str, object] = {}

    def hook(_m, _i, output):
        captured["h"] = (output[0] if isinstance(output, tuple) else output).detach()

    return target.register_forward_hook(hook), captured


def extract_rows(model, tokenizer, device, captured, messages, *, case_id, label, mode,
                 d_model, tok_cap, attn_max_len, position_id):
    """One teacher-forced forward over `messages` (last = assistant reply). Emits a row
    per response token. Returns (rows, next_position_id); ([], position_id) if the
    response view fails."""
    import torch

    try:
        ids, resp, a_resp = _response_view(tokenizer, messages)
    except (ValueError, AssertionError):
        return [], position_id
    if tok_cap and len(resp) > tok_cap:
        resp = resp[:tok_cap]
    want_attn = len(ids) <= attn_max_len
    t = torch.tensor([ids], device=device)
    with torch.no_grad():
        captured.clear()
        out = model(t, output_attentions=want_attn, output_hidden_states=True)
    logits = out.logits[0]
    hs = out.hidden_states[-1][0].float()
    h_layer = captured["h"][0].to(torch.float32).cpu().numpy()
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

    rows = []
    for k, p in enumerate(resp):
        vec = h_layer[p]
        if vec.shape != (d_model,):
            raise SystemExit(f"activation shape {vec.shape} != ({d_model},)")
        rows.append({
            "position_id": np.uint64(position_id), "activation": vec.tolist(),
            "case_id": case_id, "mode": mode, "tok_idx": np.int64(k), "token": toks[p],
            "label": np.int64(int(label)),
            "surprisal": np.float32(surp[k]), "entropy": np.float32(ent[k]),
            "varentropy": np.float32(varent[k]), "resid_jump": np.float32(rjump[k]),
            "lookback_ratio": np.float32(lb[k]), "sink_drain": np.float32(sk[k]),
            "head_disagreement": np.float32(hd[k]),
        })
        position_id += 1
    return rows, position_id
