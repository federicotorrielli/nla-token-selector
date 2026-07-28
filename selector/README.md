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

**All-token variant** (every position of the full rendered transcript, chat
template included — the deployment-realistic pool a selector must rank):

- `bridge_extract_all.py --kind {opi,tt,taboo,liars,hand}` — full-sequence
  activations + the blind signals per token, plus `region`
  (template/system/user/assistant/assistant_prior) and `probe_tok_idx` (the
  token's index in the OLD probed subset, -1 if new). Output is a DIRECTORY of
  parquet shards (`results/bridge/all_{kind}_{model}_corpus/`, `.done` sentinel)
  because these corpora no longer fit in one file. Tensor Trust and taboo
  replies are rebuilt from the tokens stored in the original corpora (taboo
  samples were unseeded temperature draws, so regeneration would not reproduce
  them); replies stored at the old 30-token cap are then greedily continued up
  to the original generation budget (64 tt / 48 taboo, stopping at end of
  sequence), so every reply runs to its natural end. For the sampled taboo
  replies that tail is a deterministic reconstruction conditioned on the
  sampled prefix; the true sampled tail was never saved.
- `seed_from_cache.py` — greedy NLA decode is deterministic in the activation
  and the probed tokens' activations are unchanged, so the old explanations and
  judge labels are transplanted onto the new position ids
  (join on `case_id` + `probe_tok_idx`) as a `shard-cache.parquet`.
- `bridge_run_nla.py` / `bridge_judge_ontask.py` accept a shard DIRECTORY for
  `--corpus` / `--explanations` and skip every position already present in the
  output directory (which is how the seeded cache short-circuits them).
- `run_bridge_all_tokens.sh` — drives extract → seed → decode → judge across
  models, printing per-kind token counts and the projected NLA decode volume
  after extraction (the cost checkpoint). Env knobs: `MODELS`, `KINDS`,
  `LIMIT`/`NLIMIT` (smoke).
- `consolidate_all.py` — joins each corpus (minus activations), its
  explanations, and its judge labels into ONE canonical parquet per benchmark
  x model, `results/bridge/all_{kind}_{model}.parquet` — the tidy, reusable
  per-token table for further experiments. The shard directories are pipeline
  internals; downstream work should read the consolidated files.
- `bridge_run_nla.py --engine` — the fast decode path, and what the all-token
  orchestration uses. It drives SGLang **in-process** (no AV server to launch)
  with **continuous batching**: a fixed number of requests stay in flight and
  each slot refills as it frees. Measured on q7, per position:

  | path | pos/s |
  |---|---|
  | HTTP server, batch 32, serial (the original) | 3.9 |
  | HTTP server, batch 32, 4 concurrent | 12.4 |
  | in-process engine, lockstep batches | 16.0 |
  | in-process engine, continuous batching | 21.9 (28.4 in production) |

  The win is continuous batching: explanations average ~145 tokens against a
  600 cap, so a lockstep batch idles at the pace of its longest sequence.
  Bigger batches and CUDA graphs were both measured and do **nothing** here
  (batch 64→512 flat; CUDA graphs 5%, i.e. noise), so graphs stay disabled as
  the NLA inference code intends. `bench_decode.py` / `bench_engine.py` are the
  measurement scripts; re-run them if the hardware or SGLang version changes.
- **Multi-GPU**: set `NGPU` (defaults to every visible GPU). Decode and judge
  then run one worker per GPU over disjoint shards
  (`--shard-stride N --shard-offset K`, `CUDA_VISIBLE_DEVICES` per worker).
  This is *data* parallelism: every AV in the registry, l70 included, fits on
  one B200, so splitting a model across GPUs would only add communication.
  Each worker writes explanation shards named after the corpus shards it read,
  so the output directory reassembles itself with no merge step, and a dead
  worker is restarted with the same offset to resume just its share.
  `python selector/bridge_run_nla.py --selftest` checks the partition is
  disjoint and complete. Scaling is near-linear in GPUs; watch host CPU, since
  one decode worker carried a load of ~70 on a 384-core node, so ~4-5 workers
  per machine is the comfortable ceiling.
- `start_all_tokens.sh` — starts the run in a detached **tmux** session
  (`tmux attach -t nla` to watch). Plain nohup over a closing SSH channel did
  not survive reliably.
- `bridge_report.py --kind K --all-tokens` and `make_csvs.py --all` read the
  `all_*` outputs (reports gain a region-composition table; CSVs are gzipped).

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
