# The bridge on the 27 authored cases — ALL tokens (chat template + input + response)

> Bridge results. At each token we run the NLA, ask the judge (nvidia/DeepSeek-V4-Flash-NVFP4) whether its explanation is *on-task* for the threat, and test two things: is the NLA more on-task inside the flagged span (Q1), and does a cheap one-pass signal predict where it is on-task (Q2). AUROC: 0.5 is chance; below 0.5 the signal points the other way. Intervals are case-cluster bootstraps over whole transcripts. Pre-specified primary signal(s): head_disagreement, kl.

## Q1 — is the NLA more on-task inside the flagged span?

On-task rate inside vs outside the planted span, the lift, and its case-cluster bootstrap 95% interval (2000 resamples over whole transcripts). A lift whose interval clears 0 is a real localization of the NLA's on-task-ness to the flagged span.

| model | threat | inside | outside | lift [95% CI] |
|---|---|---|---|---|
| Qwen-7B | eval-awareness | 0.19 (n=98) | 0.09 (n=445) | **+0.10** [+0.03, +0.19] ✓ |
| Qwen-7B | injection | 0.88 (n=129) | 0.91 (n=621) | **-0.03** [-0.12, +0.06] |
| Gemma-12B | eval-awareness | 0.42 (n=99) | 0.13 (n=449) | **+0.29** [+0.19, +0.41] ✓ |
| Gemma-12B | injection | 0.92 (n=130) | 0.95 (n=626) | **-0.03** [-0.11, +0.04] |
| Gemma-27B | eval-awareness | 0.62 (n=99) | 0.17 (n=449) | **+0.44** [+0.34, +0.56] ✓ |
| Gemma-27B | injection | 0.91 (n=130) | 0.94 (n=626) | **-0.04** [-0.11, +0.02] |
| Llama-70B | eval-awareness | 0.61 (n=98) | 0.19 (n=440) | **+0.42** [+0.26, +0.57] ✓ |
| Llama-70B | injection | 0.80 (n=128) | 0.87 (n=621) | **-0.08** [-0.17, +0.02] |

## Region composition

Share of all tokens and NLA on-task rate per region (template = chat-template scaffolding outside any message content).

| model | region | share | on-task |
|---|---|---|---|
| Qwen-7B | user | 0.44 | 0.56 |
| Qwen-7B | assistant | 0.34 | 0.50 |
| Qwen-7B | system | 0.22 | 0.70 |
| Gemma-12B | user | 0.45 | 0.64 |
| Gemma-12B | assistant | 0.34 | 0.55 |
| Gemma-12B | system | 0.22 | 0.70 |
| Gemma-27B | user | 0.45 | 0.65 |
| Gemma-27B | assistant | 0.34 | 0.63 |
| Gemma-27B | system | 0.22 | 0.69 |
| Llama-70B | user | 0.44 | 0.57 |
| Llama-70B | assistant | 0.34 | 0.62 |
| Llama-70B | system | 0.22 | 0.69 |

## Q2 — which cheap pre-pass signal predicts where the NLA is on-task?

Full table: AUROC of **every** signal vs the judge's on-task label, one row per signal, one column per model, with a case-cluster bootstrap 95% interval. The **best signal for the task is bolded** per model. `*` = blind (one pass, no counterfactual). `†` = passes Benjamini-Hochberg FDR at q=0.05 over the exploratory family; the pre-specified primaries are confirmatory and shown without penalty. The last two rows give the on-task base rate and two controls (a random-normal score, and the top signal's AUROC under label permutation), both of which sit at ~0.5 when the machinery is honest.

### eval-awareness

| signal | Qwen-7B | Gemma-12B | Gemma-27B | Llama-70B |
|---|---|---|---|---|
| `surprisal` | 0.562 [0.51,0.62]* | 0.497 [0.43,0.56]* | 0.537 [0.46,0.61]* | 0.504 [0.46,0.55]* |
| `entropy` | 0.535 [0.45,0.59]* | 0.446 [0.38,0.50]* | 0.416 [0.36,0.47]*† | 0.426 [0.34,0.49]* |
| `varentropy` | 0.552 [0.50,0.59]* | 0.423 [0.35,0.47]*† | 0.415 [0.36,0.47]*† | 0.403 [0.32,0.48]*† |
| `temporal_kl` | 0.506 [0.46,0.54]* | 0.556 [0.51,0.62]*† | 0.632 [0.57,0.70]*† | 0.511 [0.43,0.60]* |
| `resid_jump` | 0.468 [0.39,0.53]* | 0.519 [0.47,0.59]* | 0.643 [0.60,0.69]*† | 0.400 [0.34,0.46]*† |
| `lookback_ratio` | 0.562 [0.50,0.62]* | 0.500 [0.43,0.55]* | 0.446 [0.39,0.49]*† | 0.417 [0.35,0.48]*† |
| `sink_drain` | 0.571 [0.51,0.65]* | **0.669 [0.62,0.72]*†** | 0.678 [0.64,0.72]*† | **0.715 [0.66,0.78]*†** |
| `head_disagreement` | 0.565 [0.51,0.63]*† | 0.656 [0.60,0.73]*† | 0.648 [0.60,0.70]*† | 0.693 [0.64,0.76]*† |
| `w` | 0.520 [0.47,0.58]* | 0.595 [0.56,0.64]*† | 0.615 [0.57,0.66]*† | 0.650 [0.60,0.70]*† |
| `norm_ratio` | **0.322 [0.24,0.41]*†** | 0.588 [0.54,0.64]*† | 0.618 [0.57,0.67]*† | 0.646 [0.60,0.69]*† |
| `peak_ratio` | 0.390 [0.33,0.46]*† | 0.590 [0.51,0.65]*† | **0.309 [0.27,0.34]*†** | 0.435 [0.35,0.52]* |
| `dominant_mass` | 0.432 [0.36,0.51]* | 0.593 [0.52,0.65]*† | 0.566 [0.51,0.64]*† | 0.422 [0.33,0.51]* |
| `resid_jump_nla` | 0.542 [0.48,0.60]* | 0.489 [0.41,0.58]* | 0.490 [0.45,0.53]* | 0.621 [0.56,0.69]*† |
| _base rate_ | 0.11 | 0.18 | 0.25 | 0.27 |
| _control rand / perm_ | 0.51 / 0.5±0.03 | 0.50 / 0.5±0.03 | 0.53 / 0.5±0.02 | 0.49 / 0.5±0.02 |

### injection

| signal | Qwen-7B | Gemma-12B | Gemma-27B | Llama-70B |
|---|---|---|---|---|
| `surprisal` | 0.530 [0.45,0.62]* | 0.526 [0.41,0.65]* | 0.527 [0.43,0.68]* | 0.504 [0.45,0.57]* |
| `entropy` | 0.492 [0.42,0.56]* | 0.598 [0.48,0.71]* | 0.526 [0.45,0.63]* | 0.450 [0.41,0.50]* |
| `varentropy` | 0.493 [0.41,0.58]* | 0.614 [0.50,0.71]* | 0.510 [0.43,0.61]* | 0.442 [0.39,0.52]* |
| `temporal_kl` | 0.508 [0.43,0.56]* | 0.379 [0.32,0.46]*† | 0.485 [0.39,0.58]* | 0.581 [0.51,0.65]* |
| `resid_jump` | 0.423 [0.34,0.51]* | 0.391 [0.27,0.51]* | **0.395 [0.33,0.48]*** | 0.496 [0.43,0.59]* |
| `lookback_ratio` | 0.524 [0.46,0.59]* | **0.643 [0.55,0.71]*†** | 0.543 [0.46,0.63]* | 0.514 [0.45,0.56]* |
| `sink_drain` | 0.424 [0.36,0.51]* | 0.413 [0.33,0.48]*† | 0.485 [0.39,0.58]* | 0.484 [0.42,0.59]* |
| `head_disagreement` | **0.398 [0.32,0.49]*†** | 0.430 [0.33,0.51]*† | 0.502 [0.41,0.60]*† | 0.488 [0.43,0.58]*† |
| `w` | 0.451 [0.39,0.52]* | 0.381 [0.31,0.47]*† | 0.475 [0.38,0.58]* | 0.488 [0.42,0.58]* |
| `norm_ratio` | 0.569 [0.46,0.68]* | 0.539 [0.43,0.67]* | 0.598 [0.52,0.72]* | 0.460 [0.38,0.55]* |
| `peak_ratio` | 0.516 [0.42,0.61]* | 0.552 [0.47,0.66]* | 0.438 [0.30,0.50]* | 0.489 [0.38,0.56]* |
| `dominant_mass` | 0.530 [0.43,0.61]* | 0.561 [0.48,0.67]* | 0.477 [0.37,0.59]* | 0.509 [0.39,0.58]* |
| `resid_jump_nla` | 0.534 [0.48,0.57]* | 0.475 [0.41,0.55]* | 0.533 [0.47,0.60]* | **0.402 [0.34,0.50]*** |
| _base rate_ | 0.91 | 0.94 | 0.94 | 0.86 |
| _control rand / perm_ | 0.51 / 0.5±0.03 | 0.62 / 0.5±0.03 | 0.55 / 0.5±0.03 | 0.51 / 0.5±0.02 |

### pooled

| signal | Qwen-7B | Gemma-12B | Gemma-27B | Llama-70B |
|---|---|---|---|---|
| `surprisal` | 0.488 [0.45,0.53]* | 0.460 [0.43,0.50]* | 0.466 [0.42,0.51]* | 0.477 [0.45,0.51]* |
| `entropy` | 0.423 [0.39,0.46]*† | 0.490 [0.45,0.53]* | 0.468 [0.42,0.51]* | 0.413 [0.38,0.45]*† |
| `varentropy` | 0.431 [0.39,0.47]*† | 0.500 [0.46,0.54]* | 0.467 [0.43,0.51]* | 0.406 [0.37,0.44]*† |
| `temporal_kl` | 0.565 [0.52,0.60]*† | 0.454 [0.42,0.49]* | 0.518 [0.47,0.56]* | 0.537 [0.50,0.58]* |
| `resid_jump` | 0.468 [0.43,0.50]* | **0.366 [0.33,0.40]*†** | 0.408 [0.36,0.47]*† | 0.457 [0.42,0.50]* |
| `lookback_ratio` | 0.558 [0.52,0.59]*† | 0.563 [0.53,0.59]*† | 0.527 [0.49,0.57]* | 0.505 [0.47,0.54]* |
| `sink_drain` | 0.552 [0.51,0.59]*† | 0.570 [0.53,0.60]*† | 0.591 [0.55,0.63]*† | 0.594 [0.54,0.64]*† |
| `head_disagreement` | 0.560 [0.52,0.60]*† | 0.584 [0.54,0.62]*† | **0.597 [0.55,0.64]*†** | **0.610 [0.56,0.66]*†** |
| `w` | 0.487 [0.47,0.51]* | 0.512 [0.48,0.54]* | 0.538 [0.51,0.57]*† | 0.530 [0.49,0.58]* |
| `norm_ratio` | **0.360 [0.31,0.41]*†** | 0.436 [0.39,0.49]* | 0.501 [0.45,0.55]* | 0.500 [0.44,0.56]* |
| `peak_ratio` | 0.393 [0.35,0.45]*† | 0.527 [0.48,0.57]* | 0.445 [0.40,0.48]*† | 0.494 [0.44,0.54]* |
| `dominant_mass` | 0.408 [0.36,0.46]*† | 0.533 [0.49,0.58]* | 0.553 [0.50,0.60]* | 0.490 [0.44,0.53]* |
| `resid_jump_nla` | 0.497 [0.46,0.53]* | 0.466 [0.43,0.50]* | 0.482 [0.45,0.52]* | 0.501 [0.45,0.55]* |
| _base rate_ | 0.57 | 0.62 | 0.65 | 0.61 |
| _control rand / perm_ | 0.50 / 0.5±0.01 | 0.50 / 0.5±0.01 | 0.50 / 0.5±0.01 | 0.51 / 0.5±0.01 |

## Headline: the number

Best **blind** pre-pass signal for predicting where the NLA is on-task, pooled over threats, with its 95% interval:

| model | best blind signal | AUROC [95% CI] |
|---|---|---|
| Qwen-7B | `norm_ratio` | 0.360 [0.31, 0.41] |
| Gemma-12B | `resid_jump` | 0.366 [0.33, 0.40] |
| Gemma-27B | `head_disagreement` | 0.597 [0.55, 0.64] |
| Llama-70B | `head_disagreement` | 0.610 [0.56, 0.66] |
