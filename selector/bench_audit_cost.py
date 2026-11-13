"""selector/bench_audit_cost.py — what a selector adds to the cost of an audit (E7).

Three costs per model, on the same OPI transcripts:

  (a) activation pass  one forward with the default (SDPA) attention and a hook at
                       the verbalizer layer; any audit pays this to get activations.
  (b) signal pass      the forward the signals need: eager attention with every
                       attention map, the full next-token distribution, all hidden
                       states (`bridge_extract_all.score_all`).
  (c) explanation      one NLA explanation; measured separately by
                       `bench_engine.py --continuous` (positions per second).

A signal ranker adds (b) - (a) when activations are captured after the fact, and
(b) when the audit reuses a forward the deployment already ran for generation
(then (a) is free and the signal pass is extra). *structure* adds nothing. The
script also reports floating point operations: 2 x parameters x tokens per forward.

Usage (one GPU):
    python selector/bench_audit_cost.py --short q7 --n 50 --out results/review/e7_q7.json
"""

from __future__ import annotations

import argparse
import gc
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _response_extract import make_hook  # noqa: E402
from bridge_extract_all import score_all  # noqa: E402

BASES = {"q7": ("Qwen/Qwen2.5-7B-Instruct", 20), "g12": ("google/gemma-3-12b-it", 32),
         "g27": ("google/gemma-3-27b-it", 41), "l70": ("meta-llama/Llama-3.3-70B-Instruct", 53)}
CASES = Path(__file__).resolve().parent / "data" / "injection_cases.jsonl"


def transcripts(tokenizer, n: int) -> list[list[int]]:
    rows = [json.loads(ln) for ln in CASES.read_text().splitlines()[:n]]
    out = []
    for r in rows:
        msgs = [{"role": "system", "content": r["system"]},
                {"role": "user", "content": r["user_full"]}]
        try:
            ids = tokenizer.apply_chat_template(msgs, add_generation_prompt=True,
                                                tokenize=True)
        except Exception:  # templates without a system role (Gemma folds it)
            msgs = [{"role": "user", "content": f"{r['system']}\n\n{r['user_full']}"}]
            ids = tokenizer.apply_chat_template(msgs, add_generation_prompt=True,
                                                tokenize=True)
        out.append(list(ids["input_ids"] if hasattr(ids, "keys") else ids))
    return out


def load(name: str, attn: str):
    import torch
    from transformers import AutoModelForCausalLM, AutoModelForImageTextToText

    kw = dict(device_map="cuda", dtype=torch.bfloat16, attn_implementation=attn)
    try:
        return AutoModelForCausalLM.from_pretrained(name, **kw).eval()
    except (ValueError, KeyError):
        return AutoModelForImageTextToText.from_pretrained(name, **kw).eval()


def timed(fn, reps: int) -> float:
    import torch

    fn()  # warm-up
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(reps):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / reps


def main(argv=None) -> int:
    import torch
    from transformers import AutoTokenizer

    ap =argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--short", required=True, choices=list(BASES))
    ap.add_argument("--n", type=int, default=50, help="transcripts")
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args(argv)

    name, layer = BASES[args.short]
    tok = AutoTokenizer.from_pretrained(name)
    seqs = transcripts(tok, args.n)
    n_tok = sum(map(len, seqs))
    result = {"model": args.short, "transcripts": len(seqs), "tokens": n_tok}

    for label, attn in (("activation_pass", "sdpa"), ("signal_pass", "eager")):
        model = load(name, attn)
        handle, captured = make_hook(model, layer)
        dev = str(model.get_input_embeddings().weight.device)
        if label == "activation_pass":
            def run(model=model, dev=dev):
                with torch.no_grad():
                    for ids in seqs:
                        model(torch.tensor([ids], device=dev))
        else:
            def run(model=model, dev=dev, captured=captured):
                for ids in seqs:
                    score_all(model, tok, dev, captured, ids, a_bound=4,
                              attn_max_len=10**9)
        sec = timed(run, args.reps)
        result[f"{label}_s_per_transcript"] = sec / len(seqs)
        result[f"{label}_ms_per_token"] = 1000 * sec / n_tok
        result["parameters"] = sum(p.numel() for p in model.parameters())
        handle.remove()
        # `run` holds the model as a default argument: drop both before the next
        # load, or a 70B model is resident twice and the second load runs out of memory.
        del run, model, captured
        gc.collect()
        torch.cuda.empty_cache()

    result["forward_flops_per_token"] = 2 * result["parameters"]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
