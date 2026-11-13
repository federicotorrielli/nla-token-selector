"""selector/bridge_judge_second.py — relabel every explanation with a second evaluator.

The first evaluator (DeepSeek-V4-Flash, `bridge_judge_ontask.py`) answered one yes/no
question per explanation. This script asks a model from another family the SAME
system prompt and question, over an OpenAI-compatible API, so that every headline
number can be recomputed under a second set of labels.

Differences from the first evaluator, all deliberate:
  * strict parsing: only "yes"/"no" (case and surrounding punctuation stripped) is a
    label; anything else is stored as invalid (label null), never as "no".
  * the raw reply is kept, so the parser can be audited.
  * one call per unique (question, explanation); temperature 0 makes repeats redundant.

Output: results/bridge/second_eval/{kind}_{model}.parquet with position_id, on_task2
(Int8, null = invalid). The cache of raw replies lives in
results/bridge/second_eval/cache/*.parquet, keyed by sha1(question + explanation).

The key is read from CNLA_EVAL_API_KEY; it never touches the repository.

Usage:
    CNLA_EVAL_API_KEY=... uv run python selector/bridge_judge_second.py \
        --kinds opi tt liars taboo --concurrency 128
    uv run python selector/bridge_judge_second.py --selftest
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import os
import sys
import time
from pathlib import Path

import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parent))
from bridge_judge_ontask import SYS_PROMPT, _question  # noqa: E402

BR = Path("results/bridge")
OUT = BR / "second_eval"
CACHE = OUT / "cache"
KINDS = {"opi": ["q7", "g12", "g27", "l70"], "tt": ["q7", "g12", "g27", "l70"],
         "liars": ["g27", "l70"], "taboo": ["q7", "g12", "g27", "l70"]}


def user_prompt(explanation: str, question: str) -> str:
    """Byte-identical to bridge_judge_ontask._judge_one."""
    return (f'Description: "{explanation}"\n\nQuestion: For this description, '
            f'{question}\nAnswer Yes or No.')


def key_of(question: str, explanation: str) -> str:
    return hashlib.sha1(f"{question}\x00{explanation}".encode()).hexdigest()


def parse(reply: str | None) -> int | None:
    """Strict: the reply must be one word, yes or no."""
    if reply is None:
        return None
    word = reply.strip().strip(".!\"'`*").strip().lower()
    return {"yes": 1, "no": 0}.get(word)


def load_cache() -> dict[str, str | None]:
    files = sorted(CACHE.glob("*.parquet"))
    if not files:
        return {}
    df = pl.read_parquet(files).unique("key", keep="last")
    return dict(zip(df["key"].to_list(), df["reply"].to_list(), strict=True))


def todo_frame(kind: str, model: str) -> pl.DataFrame:
    """One row per position with its question and cache key."""
    df = pl.read_parquet(BR / f"all_{kind}_{model}.parquet",
                         columns=["position_id", "case_id", "mode", "explanation"])
    df = df.filter(pl.col("explanation").is_not_null())
    q = [_question(m, c)
         for m, c in zip(df["mode"].to_list(), df["case_id"].to_list(), strict=True)]
    keys = [key_of(a, b) for a, b in zip(q, df["explanation"].to_list(), strict=True)]
    return df.with_columns(pl.Series("question", q), pl.Series("key", keys))


async def _call(client, sem, args, question, explanation) -> str | None:
    body = {"model": args.model, "temperature": 0.0, "max_tokens": args.max_tokens,
            "reasoning_effort": args.reasoning_effort,
            "messages": [{"role": "system", "content": SYS_PROMPT},
                         {"role": "user", "content": user_prompt(explanation, question)}]}
    async with sem:
        for attempt in range(8):
            try:
                r = await client.post(args.url, json=body)
                if r.status_code in (429, 500, 502, 503, 504):
                    await asyncio.sleep(min(60, 2 ** attempt))
                    continue
                r.raise_for_status()
                return r.json()["choices"][0]["message"].get("content")
            except Exception:  # network errors: back off and retry
                await asyncio.sleep(min(60, 2 ** attempt))
    return None  # stays out of the cache, retried on the next run


async def run_calls(items: list[tuple[str, str, str]], args) -> None:
    """items: (key, question, explanation). Writes cache chunks as it goes."""
    import httpx

    CACHE.mkdir(parents=True, exist_ok=True)
    sem = asyncio.Semaphore(args.concurrency)
    headers = {"Authorization": f"Bearer {os.environ['CNLA_EVAL_API_KEY']}"}
    limits = httpx.Limits(max_connections=args.concurrency + 8)
    async with httpx.AsyncClient(headers=headers, timeout=180.0, limits=limits) as client:
        t0 = time.time()
        for start in range(0, len(items), args.chunk):
            chunk = items[start:start + args.chunk]
            replies = await asyncio.gather(*(_call(client, sem, args, q, e)
                                             for _, q, e in chunk))
            done = [(k, r) for (k, _, _), r in zip(chunk, replies, strict=True)
                    if r is not None]
            if done:
                keys, reps = zip(*done, strict=True)
                pl.DataFrame({"key": list(keys), "reply": list(reps)}).write_parquet(
                    CACHE / f"{time.time_ns()}.parquet")
            n = start + len(chunk)
            rate = n / (time.time() - t0)
            print(f"  {n}/{len(items)}  {rate:.1f}/s  eta {(len(items) - n) / rate / 3600:.1f} h"
                  f"  failed {len(chunk) - len(done)}", flush=True)


def do_run(args) -> int:
    cache = load_cache()
    frames = {(k, m): todo_frame(k, m) for k in args.kinds for m in KINDS[k]
              if args.models is None or m in args.models}
    items, seen = [], set(cache)
    for df in frames.values():
        for k, q, e in df.select("key", "question", "explanation").iter_rows():
            if k not in seen:
                seen.add(k)
                items.append((k, q, e))
    if args.limit:
        items = items[:args.limit]
    print(f"{len(cache)} cached, {len(items)} to call", flush=True)
    if items:
        asyncio.run(run_calls(items, args))
        cache = load_cache()
    OUT.mkdir(parents=True, exist_ok=True)
    for (kind, model), df in frames.items():
        rep = pl.DataFrame({"key": list(cache), "reply": list(cache.values())})
        out = df.join(rep, on="key", how="left").with_columns(
            pl.col("reply").map_elements(parse, return_dtype=pl.Int8).alias("on_task2"))
        out.select("position_id", "reply", "on_task2").write_parquet(
            OUT / f"{kind}_{model}.parquet")
        n, cov = out.height, out["reply"].is_not_null().mean()
        inv = (out["reply"].is_not_null() & out["on_task2"].is_null()).sum()
        print(f"{kind}_{model}: {n} positions, replied {cov:.1%}, invalid {inv}")
    return 0


def selftest() -> int:
    ok = True

    def check(name, cond):
        nonlocal ok
        ok &= bool(cond)
        print(("PASS " if cond else "FAIL ") + name)

    check("yes parses", parse("Yes") == 1 and parse(" yes.\n") == 1 and parse("**No**") == 0)
    check("prefix is invalid", parse("Yes, because") is None and parse("Y") is None)
    check("none is invalid", parse(None) is None and parse("") is None)
    check("key depends on question", key_of("a", "x") != key_of("b", "x"))
    check("prompt matches first evaluator", user_prompt("E", "Q?").endswith("Q?\nAnswer Yes or No."))
    check("taboo question filled", "'moon'" in _question("taboo", "moon__p0__s1"))
    return 0 if ok else 1


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--kinds", nargs="+", default=list(KINDS))
    ap.add_argument("--models", nargs="+", default=None)
    ap.add_argument("--url", default="https://ai.cloud.sdu.dk/v1/chat/completions")
    ap.add_argument("--model", default="zai-org/GLM-5.3")
    ap.add_argument("--reasoning-effort", default="low")
    ap.add_argument("--max-tokens", type=int, default=256,
                    help="covers the reasoning tokens a reasoning model spends first")
    ap.add_argument("--concurrency", type=int, default=64)
    ap.add_argument("--chunk", type=int, default=4096, help="calls per cache checkpoint")
    ap.add_argument("--limit", type=int, default=0, help="call at most N (pilot runs)")
    args = ap.parse_args(argv)
    return selftest() if args.selftest else do_run(args)


if __name__ == "__main__":
    raise SystemExit(main())
