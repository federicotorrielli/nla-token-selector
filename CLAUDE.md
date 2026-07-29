# CLAUDE.md

Guidance for Claude Code working in this repository.

## What this is

A study of **which token positions deserve an (expensive) NLA explanation**. A
Natural Language Autoencoder (NLA) reads a language model's residual-stream
activation at one token and writes a paragraph explaining it, at roughly 500
generated tokens per explanation. A long transcript has hundreds of tokens, so
you cannot explain them all. This project asks whether a cheap number, computed
in one ordinary forward pass before any NLA call, picks the positions where the
NLA will produce a task-relevant explanation.

**Inference-only.** Nothing here trains a model or launches an inference server;
it drives externally launched SGLang servers over HTTP and reads cached
activations from disk. Spun out of the calibrated-nla project (the per-claim
calibration paper); this repo carries only the selector slice.

Four open NLA targets by short id: `q7` (Qwen2.5-7B), `g12` (Gemma-3-12B), `g27`
(Gemma-3-27B), `l70` (Llama-3.3-70B); registry in `configs/base.yaml`.

## Four benchmarks

| benchmark | threat | tokens | models |
|---|---|---|---|
| OpenPromptInjection | injection | input span | all four |
| Tensor Trust | injection (hijack) | response | all four |
| Liars' Bench | deception | response | l70, g27 (on-policy only) |
| Taboo organisms | secret word | response | all four × {moon, ship, snow} |

## Architecture

- `src/nla_token_selector/client.py::NLAClientLite` — the single shim for every AV
  decode / AR critic score, wrapping the vendored `_nla_inference.py`. Methods
  never POST to SGLang directly. Cache persists to Parquet on `flush()`/`close()`.
- `selector/signals*.py` — the cheap "number" methods: `signals.py` (assistant
  side + shared `auroc`, case loader), `signals_injection.py` (input span, OPI /
  Tensor Trust), `signals_deception.py` (response side, Liars' / taboo). These use
  a teacher-forced `transformers` forward because the signals need the FULL-vocab
  next-token distribution and eager attention, which SGLang does not expose. This
  is the one HF code path besides the vendored inference primitives.
- **Bridge pipeline** (per benchmark, GPU-phased, resumable): `bridge_extract_*`
  (base model → activation + signals + label) → `bridge_run_nla.py` (NLA decode per
  token via the AV server) → `bridge_judge_ontask.py` (judge: is the explanation
  on-task?) → `bridge_report.py` (Q1/Q2 tables) → `make_csvs.py` (per-token CSVs).
- `configs/base.yaml` + `settings.py` (pydantic-settings, `CNLA_` env prefix).

## Commands

Run everything from the repository root, because the scripts hold paths under
`results/` that are relative to it.

```bash
uv sync
uv run ruff check .    # line length 100; the vendored _nla_inference.py is excluded
```

The project lists pytest as a development dependency and has no test files. The
checks that exist are offline `--selftest` flags on six scripts. Each runs the maths
on fixed inputs and needs no GPU and no network:

```bash
python selector/signals.py --selftest
python selector/signals_injection.py --selftest
python selector/signals_deception.py --selftest
python selector/data/build_injection_cases.py --selftest
python selector/data/build_tensortrust_cases.py --selftest
python selector/data/taboo.py --selftest
```

The headline results are the **all-token** run: every token of the full rendered
transcript, chat template included, 4.7M positions over five benchmarks and four
models. It needs a GPU, the park models cached, and a launched SGLang server;
`selector/README.md` has the detail.

```bash
bash selector/run_bridge_all_tokens.sh   # MODELS/KINDS/LIMIT/NLIMIT env knobs
python selector/all_tokens_eval.py all   # spike cols, reports, CSVs, checks
python selector/bridge_extract_all.py --selftest   # offline check, no GPU
```

The original per-benchmark scripts (`run_bridge_{opi,tt,liars,taboo}.sh`) probed
a subset of tokens and are superseded: every signal they produced is in the
all-token tables. Only the 27-case pilot keeps a subset output, because it alone
carries the referenced signals `kl` and `attn_rollout`, which need a second pass
over a counterfactual transcript:

```bash
bash selector/run_bridge_all.sh
python selector/bridge_report.py --kind hand
python selector/make_csvs.py
```

## Traps

- The files in `selector/` are standalone scripts. They import each other by adding
  their own directory to `sys.path`, so run them as `python selector/x.py`. The
  `python -m` form fails.
- `bridge_report.py --kind hand` and `make_csvs.py` read the signals for the 27
  authored cases from `results/token_selector_v2/<full model tag>/tokens.parquet`.
  `signals.py` writes to `results/token_selector` by default, so pass it
  `--out results/token_selector_v2` or the join comes back empty.
- Entropy carries a different sign in the two code paths. `signals.py` and
  `signals_injection.py` negate it before storing, so a larger value means a more
  interesting token. The bridge extractors store it raw. An AUROC below 0.5 in a
  bridge table therefore means the signal runs backward, and the two families of
  tables read in opposite directions.
- The model registry appears in six places: `configs/base.yaml`, the `declare -A`
  blocks in each `run_bridge_*.sh` (including `run_bridge_all_tokens.sh`), `_FOUR`
  in `bridge_report.py`, `TAG` in `make_csvs.py`, and `BASES` in
  `selector/data/taboo.py`. Adding a model means editing all six.
- All-token outputs are DIRECTORIES of parquet shards
  (`results/bridge/all_{kind}_{model}_{corpus,explanations,ontask}/`), not single
  files — the corpora hold an activation per token and no longer fit in memory.
  A `.done` sentinel marks a complete extraction; `shard-cache.parquet` inside
  the explanations/ontask dirs holds the rows transplanted from the original run
  by `seed_from_cache.py` (join on `case_id` + `probe_tok_idx`). Tensor Trust and
  taboo replies are rebuilt from the stored corpus tokens (taboo samples were
  unseeded temperature draws, so full regeneration cannot reproduce them), then
  greedily continued past the old 30-token cap up to the original generation
  budget. `consolidate_all.py` collapses each benchmark x model into ONE
  canonical `results/bridge/all_{kind}_{model}.parquet` (everything but the
  activations) — further experiments should read that, not the shard dirs. It
  also derives `w` from `sink_drain` and `lookback_ratio`, and joins the
  activation-derived columns written by `all_tokens_eval.py spike`.
- Configuration arrives as environment variables with the `CNLA_` prefix, which
  dates from the calibrated-nla project this repository was spun out of. They have
  to be set before the package is imported. The top of `bridge_run_nla.py` shows
  the pattern.
- `selector/data/injection_cases.jsonl` is committed. `tensortrust_cases.jsonl` is
  built on the first run by `run_bridge_tt.sh`. Everything under `results/` and
  every `*.parquet` file is ignored by git.
- The AV servers listen on ports 30000, 30001, 30002 and 30003 for q7, g12, g27 and
  l70; the judge `nvidia/DeepSeek-V4-Flash-NVFP4` listens on 31000. One heavy model
  sits on the GPU at a time, and each phase skips an output that already exists, so
  an interrupted run picks up where it stopped.

## Reporting rule

Every benchmark result doc shows the **full per-signal metric table** (one row per
token-selector signal, best-for-task bolded), never just the top-k, and runs **all
four models** except Liars' Bench (on-policy, so l70 + g27 only).

The statistics behind those tables are fixed in advance. Two signals were chosen
before the benchmark runs and are reported without a penalty for multiple
comparisons: `head_disagreement` for the blind case (one forward pass) and `kl` for
the referenced case (a second pass over a counterfactual transcript). Every other
signal is exploratory and passes Benjamini-Hochberg control of the false discovery
rate at q=0.05. Intervals are bootstraps that resample whole transcripts rather than
individual tokens, 2000 resamples, seed 0, because the tokens inside one transcript
are correlated. Two controls, a random score and a permutation of the on-task
labels, sit at 0.5 in every cell. A run where they drift is broken.

The **all-token** tables are the headline; the 27-case pilot is a controlled
methodology check with wide intervals, not evidence.
`findings/token-selector/README.md` holds the current results and
`findings/token-selector/bridge-experiment-design.md` holds the hypotheses, the
statistics and the controls. Both outrank anything restated here.

## Conventions

- `results/` gitignored; `paper_results/bridge/` holds the committed per-token CSVs;
  `findings/token-selector/` holds the result docs.
- A "position_id" is a global flat token index into the extraction corpus.
- Runs need a machine with a GPU, the park models cached, and an externally
  launched SGLang server (see `selector/launch_sglang.sh`).

## Style

British quotation punctuation, no em-dashes, no jargon in chat (explain terms),
concise commits without Co-Authored-By trailers.
