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
activations from disk. Spun out of `~/LocalProjects/calibrated_nla` (the per-claim
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

## Reporting rule

Every benchmark result doc shows the **full per-signal metric table** (one row per
token-selector signal, best-for-task bolded), never just the top-k, and runs **all
four models** except Liars' Bench (on-policy, so l70 + g27 only).

## Conventions

- `results/` gitignored; `paper_results/bridge/` holds the committed per-token CSVs;
  `findings/token-selector/` holds the result docs; `paper/` is the manuscript.
- A "position_id" is a global flat token index into the extraction corpus.
- Run on the SDU B200: `ssh ucloud@ssh.cloud.sdu.dk` (ephemeral port), env
  `~/miniconda3/envs/pao`, workdir `/work/nla_token_selector`.

## Style

British quotation punctuation, no em-dashes, no jargon in chat (explain terms),
concise commits without Co-Authored-By trailers.
