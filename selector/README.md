# Token selector

Code for the experiment that opens the NLA-calibration paper: *which token
positions deserve an expensive NLA explanation, and can a cheap one-pass number
tell you before you spend the budget?* The science, results, and validity
argument live next door in `findings/token-selector/` (start with `README.md`
there, then `bridge-experiment-design.md`). This file is the code map.

Everything here talks to externally launched SGLang servers over HTTP; nothing
trains or serves a model. Data lives in `data/`, raw result tables in
`paper_results/bridge/`.

## What each file is

**Signal engines** (a forward pass computes the cheap signals; also runnable as
the localization studies that produced the paper's headline AUROCs):

| file | threat / setting | run it to get |
|---|---|---|
| `signals.py` | assistant-side (injection, eval-awareness) on the 27 authored cases | the per-token signals + the localization AUROC table |
| `signals_injection.py` | injected *input* span, OpenPromptInjection | input-token signals + localization on real injections |
| `signals_deception.py` | deception, Liars' Bench responses | response-token signals + lie-detection AUROCs |

`signals.py` also holds shared helpers (`auroc`, the case loader, token labels)
that the other engines and the bridge scripts import.

**Bridge pipeline** (does a cheap signal predict where the NLA lands on-task?),
run in this order:

1. `bridge_extract_*` — one per benchmark: forward the base model, capture the
   NLA-layer activation, the ground-truth label, and the cheap signals into a
   corpus parquet. `bridge_extract_activations.py` (27 hand cases),
   `bridge_extract_opi.py` (OPI input span), `bridge_extract_liars.py` (Liars'
   responses), `bridge_extract_tt.py` (Tensor Trust: generate a base-model reply,
   probe the response), `bridge_extract_taboo.py` (taboo: base + LoRA adapter,
   generate hints, probe the response). The three response-side extractors share
   `_response_extract.py` (generate → teacher-forced forward → per-token signals).
2. `bridge_run_nla.py` — greedy NLA decode at each token, via its AV SGLang server.
3. `bridge_judge_ontask.py --judge` — ask the judge (DeepSeek-V4-Flash) whether
   each explanation is on-task (taboo: whether it reveals the secret word).
4. `bridge_report.py --kind {hand,opi,liars,tt,taboo}` — full per-signal Q1/Q2
   tables (best-for-task bolded) with clustered-bootstrap intervals, controls, and
   FDR, written to `findings/token-selector/bridge-*.md`.
5. `make_csvs.py` — export the tidy per-token CSVs to `paper_results/bridge/`.

**Orchestration** (GPU-phased, resumable; one heavy model on the GPU at a time):

- `run_bridge_all.sh` (hand), `run_bridge_opi.sh` (OPI), `run_bridge_liars.sh`
  (Liars'), `run_bridge_tt.sh` (Tensor Trust), `run_bridge_taboo.sh` (taboo) —
  drive extract → NLA → judge across models.
- `launch_sglang.sh` — the SGLang launcher they call. Set `PENV` to a conda env
  prefix or activate one first.

**Data** (`data/`): `token_selector_cases.json` (27 authored cases),
`injection_cases.jsonl` + `build_injection_cases.py` (OPI),
`build_tensortrust_cases.py` (Tensor Trust hijacking, 776×2 variants),
`taboo.py` (taboo adapter registry + the four elicitation prompts).

## Reproduce

```bash
# activate the env (needs torch + sglang + the park models cached), then:
PENV=/path/to/conda/env   # or activate it and skip this

bash selector/run_bridge_opi.sh      # injection (OPI), all models
bash selector/run_bridge_tt.sh       # injection (Tensor Trust), all models
bash selector/run_bridge_liars.sh    # deception (Liars'), on-policy models
bash selector/run_bridge_taboo.sh    # deception (taboo), all models × moon/ship/snow

for k in opi tt liars taboo; do python selector/bridge_report.py --kind $k; done
python selector/make_csvs.py
```

Each `.sh` skips any phase whose parquet already exists, so an interrupted run
resumes where it stopped. The localization-only studies run directly, e.g.
`python selector/signals_injection.py --help`.
