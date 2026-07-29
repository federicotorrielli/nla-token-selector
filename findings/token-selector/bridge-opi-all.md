# The bridge on real injection (OpenPromptInjection, all 800 cases) — ALL tokens (chat template + input + response)

> Bridge results. At each token we run the NLA, ask the judge (nvidia/DeepSeek-V4-Flash-NVFP4) whether its explanation is *on-task* for the threat, and test two things: is the NLA more on-task inside the flagged span (Q1), and does a cheap one-pass signal predict where it is on-task (Q2). AUROC: 0.5 is chance; below 0.5 the signal points the other way. Intervals are case-cluster bootstraps over whole transcripts. Pre-specified primary signal(s): (none — exploratory).

## Q1 — is the NLA more on-task inside the flagged span?

On-task rate inside vs outside the planted span, the lift, and its case-cluster bootstrap 95% interval (2000 resamples over whole transcripts). A lift whose interval clears 0 is a real localization of the NLA's on-task-ness to the flagged span.

| model | threat | inside | outside | lift [95% CI] |
|---|---|---|---|---|
| Qwen-7B | injection (input span) | 0.34 (n=50520) | 0.18 (n=60850) | **+0.16** [+0.16, +0.17] ✓ |
| Gemma-12B | injection (input span) | 0.22 (n=50160) | 0.12 (n=57590) | **+0.10** [+0.09, +0.10] ✓ |
| Gemma-27B | injection (input span) | 0.24 (n=50160) | 0.13 (n=57590) | **+0.11** [+0.11, +0.12] ✓ |
| Llama-70B | injection (input span) | 0.23 (n=48780) | 0.06 (n=77560) | **+0.17** [+0.16, +0.17] ✓ |

## Region composition

Share of all tokens and NLA on-task rate per region (template = chat-template scaffolding outside any message content).

| model | region | share | on-task |
|---|---|---|---|
| Qwen-7B | user | 0.77 | 0.21 |
| Qwen-7B | system | 0.14 | 0.47 |
| Qwen-7B | template | 0.09 | 0.31 |
| Gemma-12B | user | 0.79 | 0.14 |
| Gemma-12B | system | 0.14 | 0.32 |
| Gemma-12B | template | 0.07 | 0.22 |
| Gemma-27B | user | 0.79 | 0.15 |
| Gemma-27B | system | 0.14 | 0.34 |
| Gemma-27B | template | 0.07 | 0.30 |
| Llama-70B | user | 0.66 | 0.14 |
| Llama-70B | template | 0.22 | 0.02 |
| Llama-70B | system | 0.12 | 0.28 |

## Q2 — which cheap pre-pass signal predicts where the NLA is on-task?

Full table: AUROC of **every** signal vs the judge's on-task label, one row per signal, one column per model, with a case-cluster bootstrap 95% interval. The **best signal for the task is bolded** per model. `*` = blind (one pass, no counterfactual). `†` = passes Benjamini-Hochberg FDR at q=0.05 over the exploratory family; the pre-specified primaries are confirmatory and shown without penalty. The last two rows give the on-task base rate and two controls (a random-normal score, and the top signal's AUROC under label permutation), both of which sit at ~0.5 when the machinery is honest.

### injection (input span)

| signal | Qwen-7B | Gemma-12B | Gemma-27B | Llama-70B |
|---|---|---|---|---|
| `surprisal` | 0.511 [0.51,0.51]*† | 0.506 [0.50,0.51]*† | 0.495 [0.49,0.50]*† | 0.513 [0.51,0.52]*† |
| `entropy` | 0.372 [0.37,0.38]*† | 0.397 [0.39,0.40]*† | 0.452 [0.45,0.46]*† | 0.451 [0.45,0.46]*† |
| `varentropy` | 0.366 [0.36,0.37]*† | 0.391 [0.39,0.40]*† | 0.442 [0.44,0.45]*† | 0.424 [0.42,0.43]*† |
| `temporal_kl` | 0.606 [0.60,0.61]*† | 0.508 [0.50,0.51]*† | 0.509 [0.51,0.51]*† | 0.482 [0.48,0.49]*† |
| `resid_jump` | 0.554 [0.55,0.56]*† | 0.541 [0.54,0.55]*† | 0.515 [0.51,0.52]*† | 0.501 [0.50,0.51]* |
| `lookback_ratio` | 0.602 [0.59,0.61]*† | 0.603 [0.60,0.61]*† | 0.582 [0.58,0.59]*† | 0.469 [0.46,0.47]*† |
| `sink_drain` | 0.531 [0.53,0.54]*† | 0.568 [0.56,0.57]*† | 0.596 [0.59,0.60]*† | **0.673 [0.67,0.68]*†** |
| `head_disagreement` | 0.497 [0.49,0.50]* | 0.545 [0.54,0.55]*† | 0.574 [0.57,0.58]*† | 0.641 [0.64,0.65]*† |
| `norm_ratio` | 0.359 [0.35,0.36]*† | 0.603 [0.60,0.61]*† | 0.618 [0.61,0.62]*† | 0.345 [0.34,0.35]*† |
| `peak_ratio` | **0.239 [0.24,0.24]*†** | 0.414 [0.41,0.42]*† | 0.384 [0.38,0.39]*† | 0.361 [0.36,0.36]*† |
| `dominant_mass` | 0.261 [0.26,0.27]*† | 0.412 [0.41,0.42]*† | 0.427 [0.42,0.43]*† | 0.380 [0.38,0.38]*† |
| `resid_jump_nla` | 0.596 [0.59,0.60]*† | **0.685 [0.68,0.69]*†** | **0.635 [0.63,0.64]*†** | 0.463 [0.46,0.47]*† |
| `attn_rollout` | 0.603 [0.60,0.61]† | 0.565 [0.56,0.57]† | 0.559 [0.55,0.56]† | 0.502 [0.50,0.51] |
| _base rate_ | 0.25 | 0.17 | 0.18 | 0.13 |
| _control rand / perm_ | 0.50 / 0.5±0.00 | 0.50 / 0.5±0.00 | 0.50 / 0.5±0.00 | 0.50 / 0.5±0.00 |

## Headline: the number

Best **blind** pre-pass signal for predicting where the NLA is on-task, pooled over threats, with its 95% interval:

| model | best blind signal | AUROC [95% CI] |
|---|---|---|
| Qwen-7B | `peak_ratio` | 0.239 [0.24, 0.24] |
| Gemma-12B | `resid_jump_nla` | 0.685 [0.68, 0.69] |
| Gemma-27B | `resid_jump_nla` | 0.635 [0.63, 0.64] |
| Llama-70B | `sink_drain` | 0.673 [0.67, 0.68] |
