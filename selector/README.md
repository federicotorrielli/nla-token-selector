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

1. `bridge_extract_activations.py` / `bridge_extract_opi.py` /
   `bridge_extract_liars.py` — one per benchmark: forward the base model, capture
   the NLA-layer activation, the ground-truth span label, and the cheap signals
   into a corpus parquet.
2. `bridge_run_nla.py` — greedy NLA decode at each token, via its AV SGLang server.
3. `bridge_judge_ontask.py --judge` — ask the judge (DeepSeek-V4-Flash) whether
   each explanation is on-task; `--analyze` for a quick look.
4. `bridge_report.py --kind {hand,opi,liars}` — the effect-size tables
   (Q1 localization, Q2 prediction) with clustered-bootstrap intervals, controls,
   and FDR, written to `findings/token-selector/bridge-*.md`.
5. `make_csvs.py` — export the tidy per-token CSVs to `paper_results/bridge/`.

**Orchestration** (GPU-phased, resumable; one heavy model on the GPU at a time):

- `run_bridge_all.sh` (hand cases), `run_bridge_opi.sh` (injection),
  `run_bridge_liars.sh` (deception) — drive extract → NLA → judge across models.
- `launch_sglang.sh` — the SGLang launcher they call. Set `PENV` to a conda env
  prefix or activate one first.

**Data** (`data/`): `token_selector_cases.json` (the 27 authored cases),
`injection_cases.jsonl` (built OPI cases), `build_injection_cases.py` (rebuilds
the latter from an OpenPromptInjection clone).

## Reproduce

```bash
# activate the env (needs torch + sglang + the park models cached), then:
PENV=/path/to/conda/env   # or activate it and skip this

bash selector/run_bridge_opi.sh      # injection, all models
bash selector/run_bridge_liars.sh    # deception, on-policy models
bash selector/run_bridge_all.sh      # 27 authored cases

python selector/bridge_report.py --kind opi
python selector/bridge_report.py --kind liars
python selector/make_csvs.py
```

Each `.sh` skips any phase whose parquet already exists, so an interrupted run
resumes where it stopped. The localization-only studies run directly, e.g.
`python selector/signals_injection.py --help`.
