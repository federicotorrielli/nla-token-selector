# The bridge on the 27 authored cases — ALL tokens (chat template + input + response)

> Bridge results. At each token we run the NLA, ask the judge (nvidia/DeepSeek-V4-Flash-NVFP4) whether its explanation is *on-task* for the threat, and test two things: is the NLA more on-task inside the flagged span (Q1), and does a cheap one-pass signal predict where it is on-task (Q2). AUROC: 0.5 is chance; below 0.5 the signal points the other way. Intervals are case-cluster bootstraps over whole transcripts. Pre-specified primary signal(s): head_disagreement, kl.

## Q1 — is the NLA more on-task inside the flagged span?

On-task rate inside vs outside the planted span, the lift, and its case-cluster bootstrap 95% interval (2000 resamples over whole transcripts). A lift whose interval clears 0 is a real localization of the NLA's on-task-ness to the flagged span.

| model | threat | inside | outside | lift [95% CI] |
|---|---|---|---|---|
| Qwen-7B | eval-awareness | 0.19 (n=98) | 0.07 (n=614) | **+0.12** [+0.05, +0.22] ✓ |
| Qwen-7B | injection | 0.88 (n=129) | 0.91 (n=803) | **-0.02** [-0.11, +0.06] |
| Gemma-12B | eval-awareness | 0.42 (n=99) | 0.11 (n=579) | **+0.32** [+0.21, +0.44] ✓ |
| Gemma-12B | injection | 0.92 (n=130) | 0.93 (n=766) | **-0.01** [-0.09, +0.05] |
| Gemma-27B | eval-awareness | 0.62 (n=99) | 0.15 (n=579) | **+0.47** [+0.36, +0.59] ✓ |
| Gemma-27B | injection | 0.91 (n=130) | 0.95 (n=766) | **-0.04** [-0.11, +0.01] |
| Llama-70B | eval-awareness | 0.61 (n=98) | 0.10 (n=895) | **+0.51** [+0.35, +0.66] ✓ |
| Llama-70B | injection | 0.80 (n=128) | 0.76 (n=1111) | **+0.04** [-0.06, +0.14] |

## Region composition

Share of all tokens and NLA on-task rate per region (template = chat-template scaffolding outside any message content).

| model | region | share | on-task |
|---|---|---|---|
| Qwen-7B | user | 0.35 | 0.56 |
| Qwen-7B | assistant | 0.27 | 0.50 |
| Qwen-7B | template | 0.21 | 0.47 |
| Qwen-7B | system | 0.17 | 0.70 |
| Gemma-12B | user | 0.37 | 0.64 |
| Gemma-12B | assistant | 0.28 | 0.55 |
| Gemma-12B | system | 0.18 | 0.70 |
| Gemma-12B | template | 0.17 | 0.45 |
| Gemma-27B | user | 0.37 | 0.65 |
| Gemma-27B | assistant | 0.28 | 0.63 |
| Gemma-27B | system | 0.18 | 0.69 |
| Gemma-27B | template | 0.17 | 0.54 |
| Llama-70B | template | 0.42 | 0.32 |
| Llama-70B | user | 0.26 | 0.57 |
| Llama-70B | assistant | 0.19 | 0.62 |
| Llama-70B | system | 0.13 | 0.69 |

## Q2 — which cheap pre-pass signal predicts where the NLA is on-task?

Full table: AUROC of **every** signal vs the judge's on-task label, one row per signal, one column per model, with a case-cluster bootstrap 95% interval. The **best signal for the task is bolded** per model. `*` = blind (one pass, no counterfactual). `†` = passes Benjamini-Hochberg FDR at q=0.05 over the exploratory family; the pre-specified primaries are confirmatory and shown without penalty. The last two rows give the on-task base rate and two controls (a random-normal score, and the top signal's AUROC under label permutation), both of which sit at ~0.5 when the machinery is honest.

### eval-awareness

| signal | Qwen-7B | Gemma-12B | Gemma-27B | Llama-70B |
|---|---|---|---|---|
| `surprisal` | 0.523 [0.46,0.58]* | 0.434 [0.37,0.49]*† | 0.503 [0.43,0.57]* | 0.528 [0.50,0.56]* |
| `entropy` | 0.487 [0.41,0.54]* | 0.440 [0.38,0.49]*† | 0.406 [0.36,0.45]*† | 0.532 [0.47,0.59]* |
| `varentropy` | 0.494 [0.44,0.54]* | 0.426 [0.36,0.47]*† | 0.407 [0.36,0.45]*† | 0.499 [0.44,0.56]* |
| `temporal_kl` | 0.484 [0.45,0.52]* | 0.559 [0.52,0.61]*† | 0.645 [0.60,0.70]*† | 0.488 [0.41,0.57]* |
| `resid_jump` | 0.486 [0.41,0.54]* | 0.461 [0.41,0.53]* | 0.627 [0.58,0.67]*† | 0.368 [0.32,0.42]*† |
| `lookback_ratio` | 0.510 [0.45,0.56]* | 0.457 [0.40,0.50]* | 0.415 [0.37,0.45]*† | 0.326 [0.26,0.38]*† |
| `sink_drain` | 0.615 [0.56,0.69]*† | **0.695 [0.65,0.75]*†** | **0.712 [0.68,0.75]*†** | **0.827 [0.78,0.87]*†** |
| `head_disagreement` | 0.614 [0.56,0.68]*† | 0.672 [0.62,0.74]*† | 0.674 [0.63,0.72]*† | 0.795 [0.75,0.84]*† |
| `norm_ratio` | **0.301 [0.22,0.38]*†** | 0.586 [0.55,0.62]*† | 0.593 [0.55,0.65]*† | 0.679 [0.65,0.72]*† |
| `peak_ratio` | 0.373 [0.31,0.45]*† | 0.608 [0.54,0.66]*† | 0.300 [0.27,0.33]*† | 0.513 [0.45,0.58]* |
| `dominant_mass` | 0.401 [0.33,0.49]*† | 0.610 [0.54,0.67]*† | 0.555 [0.50,0.61]* | 0.441 [0.36,0.52]* |
| `resid_jump_nla` | 0.515 [0.45,0.57]* | 0.425 [0.36,0.50]* | 0.463 [0.43,0.50]* | 0.578 [0.53,0.62]*† |
| _base rate_ | 0.09 | 0.15 | 0.22 | 0.15 |
| _control rand / perm_ | 0.46 / 0.5±0.03 | 0.51 / 0.5±0.02 | 0.50 / 0.5±0.02 | 0.51 / 0.5±0.02 |

### injection

| signal | Qwen-7B | Gemma-12B | Gemma-27B | Llama-70B |
|---|---|---|---|---|
| `surprisal` | 0.475 [0.41,0.54]* | 0.491 [0.39,0.60]* | 0.557 [0.48,0.69]* | 0.411 [0.38,0.44]*† |
| `entropy` | 0.431 [0.34,0.49]* | 0.498 [0.36,0.60]* | 0.533 [0.47,0.63]* | 0.361 [0.32,0.39]*† |
| `varentropy` | 0.430 [0.34,0.50]* | 0.506 [0.38,0.60]* | 0.527 [0.46,0.62]* | 0.362 [0.33,0.39]*† |
| `temporal_kl` | 0.517 [0.44,0.56]* | 0.473 [0.40,0.58]* | 0.477 [0.38,0.57]* | 0.612 [0.58,0.65]*† |
| `resid_jump` | **0.402 [0.33,0.48]*** | 0.396 [0.30,0.48]* | 0.411 [0.35,0.48]* | 0.405 [0.37,0.44]*† |
| `lookback_ratio` | 0.513 [0.47,0.56]* | 0.569 [0.50,0.63]* | 0.551 [0.47,0.64]* | 0.455 [0.43,0.48]*† |
| `sink_drain` | 0.533 [0.47,0.64]* | 0.579 [0.51,0.67]* | 0.442 [0.36,0.51]* | **0.730 [0.66,0.80]*†** |
| `head_disagreement` | 0.503 [0.44,0.61]*† | 0.593 [0.51,0.69]*† | 0.461 [0.38,0.53]*† | 0.710 [0.65,0.78]*† |
| `norm_ratio` | 0.468 [0.35,0.57]* | 0.432 [0.35,0.51]* | **0.606 [0.53,0.71]*†** | 0.395 [0.34,0.44]*† |
| `peak_ratio` | 0.554 [0.48,0.62]* | **0.388 [0.31,0.45]*†** | 0.457 [0.32,0.53]* | 0.562 [0.52,0.60]*† |
| `dominant_mass` | 0.537 [0.46,0.60]* | 0.392 [0.32,0.45]*† | 0.477 [0.40,0.57]* | 0.525 [0.49,0.56]* |
| `resid_jump_nla` | 0.446 [0.37,0.49]*† | 0.437 [0.37,0.50]* | 0.537 [0.45,0.62]* | 0.400 [0.37,0.43]*† |
| _base rate_ | 0.90 | 0.93 | 0.94 | 0.76 |
| _control rand / perm_ | 0.47 / 0.5±0.02 | 0.49 / 0.5±0.03 | 0.53 / 0.5±0.03 | 0.47 / 0.5±0.01 |

### pooled

| signal | Qwen-7B | Gemma-12B | Gemma-27B | Llama-70B |
|---|---|---|---|---|
| `surprisal` | 0.478 [0.45,0.51]* | 0.435 [0.41,0.46]*† | 0.457 [0.42,0.49]*† | 0.457 [0.43,0.48]*† |
| `entropy` | 0.415 [0.39,0.44]*† | 0.475 [0.44,0.51]* | 0.472 [0.43,0.51]* | 0.430 [0.40,0.46]*† |
| `varentropy` | 0.422 [0.39,0.45]*† | 0.483 [0.45,0.52]* | 0.473 [0.44,0.51]* | 0.424 [0.39,0.45]*† |
| `temporal_kl` | 0.551 [0.52,0.59]*† | 0.463 [0.43,0.50]* | 0.518 [0.48,0.55]* | 0.543 [0.51,0.58]*† |
| `resid_jump` | 0.481 [0.45,0.51]* | **0.369 [0.34,0.40]*†** | 0.417 [0.38,0.47]*† | 0.419 [0.40,0.44]*† |
| `lookback_ratio` | 0.531 [0.50,0.56]*† | 0.531 [0.51,0.56]*† | 0.509 [0.47,0.54]* | 0.453 [0.42,0.48]*† |
| `sink_drain` | 0.590 [0.56,0.62]*† | 0.599 [0.57,0.63]*† | **0.598 [0.56,0.63]*†** | 0.703 [0.67,0.73]*† |
| `head_disagreement` | 0.593 [0.56,0.62]*† | 0.608 [0.57,0.64]*† | 0.598 [0.56,0.63]*† | **0.704 [0.68,0.73]*†** |
| `norm_ratio` | **0.348 [0.31,0.39]*†** | 0.427 [0.38,0.48]*† | 0.486 [0.45,0.53]* | 0.478 [0.43,0.52]* |
| `peak_ratio` | 0.406 [0.36,0.46]*† | 0.513 [0.47,0.55]* | 0.434 [0.40,0.47]*† | 0.542 [0.51,0.57]*† |
| `dominant_mass` | 0.409 [0.37,0.46]*† | 0.518 [0.48,0.56]* | 0.545 [0.51,0.59]*† | 0.508 [0.48,0.54]* |
| `resid_jump_nla` | 0.484 [0.46,0.51]* | 0.442 [0.41,0.47]*† | 0.471 [0.44,0.51]* | 0.463 [0.43,0.49]*† |
| _base rate_ | 0.55 | 0.59 | 0.63 | 0.49 |
| _control rand / perm_ | 0.48 / 0.5±0.01 | 0.50 / 0.5±0.01 | 0.49 / 0.5±0.01 | 0.52 / 0.5±0.01 |

## Headline: the number

Best **blind** pre-pass signal for predicting where the NLA is on-task, pooled over threats, with its 95% interval:

| model | best blind signal | AUROC [95% CI] |
|---|---|---|
| Qwen-7B | `norm_ratio` | 0.348 [0.31, 0.39] |
| Gemma-12B | `resid_jump` | 0.369 [0.34, 0.40] |
| Gemma-27B | `sink_drain` | 0.598 [0.56, 0.63] |
| Llama-70B | `head_disagreement` | 0.704 [0.68, 0.73] |
