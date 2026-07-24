# The bridge on the 27 authored cases

> Bridge results. At each token we run the NLA, ask the judge (nvidia/DeepSeek-V4-Flash-NVFP4) whether its explanation is *on-task* for the threat, and test two things: is the NLA more on-task inside the flagged span (Q1), and does a cheap one-pass signal predict where it is on-task (Q2). AUROC: 0.5 is chance; below 0.5 the signal points the other way. Intervals are case-cluster bootstraps over whole transcripts. Pre-specified primary signal(s): head_disagreement, kl.

## Q1 — is the NLA more on-task inside the flagged span?

On-task rate inside vs outside the planted span, the lift, and its case-cluster bootstrap 95% interval (2000 resamples over whole transcripts). A lift whose interval clears 0 is a real localization of the NLA's on-task-ness to the flagged span.

| model | threat | inside | outside | lift [95% CI] |
|---|---|---|---|---|
| Qwen-7B | eval-awareness | 0.19 (n=98) | 0.00 (n=116) | **+0.19** [+0.11, +0.29] ✓ |
| Qwen-7B | injection | 0.88 (n=129) | 0.91 (n=94) | **-0.03** [-0.16, +0.12] |
| Gemma-12B | eval-awareness | 0.42 (n=99) | 0.02 (n=116) | **+0.41** [+0.29, +0.52] ✓ |
| Gemma-12B | injection | 0.92 (n=130) | 0.86 (n=93) | **+0.06** [-0.11, +0.24] |
| Gemma-27B | eval-awareness | 0.62 (n=99) | 0.08 (n=116) | **+0.54** [+0.42, +0.65] ✓ |
| Gemma-27B | injection | 0.91 (n=130) | 0.95 (n=93) | **-0.04** [-0.10, +0.03] |
| Llama-70B | eval-awareness | 0.61 (n=98) | 0.16 (n=114) | **+0.45** [+0.25, +0.64] ✓ |
| Llama-70B | injection | 0.80 (n=128) | 0.95 (n=94) | **-0.15** [-0.29, -0.04] |

## Q2 — which cheap pre-pass signal predicts where the NLA is on-task?

Full table: AUROC of **every** signal vs the judge's on-task label, one row per signal, one column per model, with a case-cluster bootstrap 95% interval. The **best signal for the task is bolded** per model. `*` = blind (one pass, no counterfactual). `†` = passes Benjamini-Hochberg FDR at q=0.05 over the exploratory family; the pre-specified primaries are confirmatory and shown without penalty. The last two rows give the on-task base rate and two controls (a random-normal score, and the top signal's AUROC under label permutation), both of which sit at ~0.5 when the machinery is honest.

### eval-awareness

| signal | Qwen-7B | Gemma-12B | Gemma-27B | Llama-70B |
|---|---|---|---|---|
| `surprisal` | 0.475 [0.34,0.62]* | 0.414 [0.32,0.51]* | 0.491 [0.37,0.60]* | 0.500 [0.42,0.59]* |
| `entropy` | 0.419 [0.27,0.58]* | 0.581 [0.47,0.70]* | 0.531 [0.43,0.61]* | 0.489 [0.38,0.62]* |
| `varentropy` | 0.635 [0.50,0.76]* | 0.413 [0.28,0.54]* | 0.482 [0.40,0.58]* | 0.549 [0.43,0.65]* |
| `resid_jump` | 0.380 [0.32,0.43]*† | 0.366 [0.28,0.43]*† | 0.498 [0.43,0.57]* | 0.438 [0.39,0.48]*† |
| `temporal_kl` | 0.418 [0.32,0.52]* | 0.566 [0.46,0.67]* | 0.578 [0.49,0.67]* | 0.509 [0.41,0.62]* |
| `lookback_ratio` | 0.640 [0.54,0.73]*† | **0.660 [0.57,0.74]*†** | 0.628 [0.52,0.73]* | 0.574 [0.47,0.67]* |
| `sink_drain` | 0.707 [0.61,0.81]*† | 0.653 [0.58,0.73]*† | 0.602 [0.50,0.70]* | 0.632 [0.55,0.72]*† |
| `head_disagreement` | 0.750 [0.67,0.85]*† | 0.646 [0.57,0.74]*† | 0.579 [0.46,0.68]*† | 0.656 [0.56,0.74]*† |
| `w` | 0.538 [0.42,0.64]* | 0.485 [0.42,0.55]* | 0.462 [0.36,0.57]* | 0.486 [0.42,0.54]* |
| `kl` | **0.751 [0.66,0.85]†** | 0.636 [0.55,0.72]† | **0.630 [0.56,0.70]†** | **0.753 [0.69,0.81]†** |
| `attn_rollout` | 0.456 [0.30,0.63] | 0.385 [0.25,0.52] | 0.475 [0.31,0.60] | 0.462 [0.36,0.57] |
| _base rate_ | 0.09 | 0.20 | 0.33 | 0.37 |
| _control rand / perm_ | 0.54 / 0.5±0.05 | 0.54 / 0.5±0.04 | 0.53 / 0.5±0.03 | 0.49 / 0.5±0.04 |

### injection

| signal | Qwen-7B | Gemma-12B | Gemma-27B | Llama-70B |
|---|---|---|---|---|
| `surprisal` | 0.580 [0.50,0.67]* | 0.462 [0.25,0.69]* | 0.625 [0.44,0.76]* | 0.550 [0.43,0.69]* |
| `entropy` | 0.448 [0.35,0.53]* | 0.509 [0.33,0.68]* | 0.282 [0.17,0.43]* | 0.532 [0.35,0.72]* |
| `varentropy` | 0.568 [0.49,0.68]* | 0.485 [0.32,0.67]* | **0.718 [0.57,0.83]*** | 0.485 [0.34,0.65]* |
| `resid_jump` | **0.330 [0.19,0.47]*** | 0.414 [0.22,0.64]* | 0.348 [0.20,0.49]* | 0.497 [0.39,0.67]* |
| `temporal_kl` | 0.447 [0.31,0.54]* | 0.422 [0.33,0.57]* | 0.480 [0.36,0.68]* | 0.584 [0.46,0.69]* |
| `lookback_ratio` | 0.603 [0.47,0.70]* | 0.601 [0.46,0.76]* | 0.577 [0.38,0.70]* | 0.645 [0.50,0.80]* |
| `sink_drain` | 0.542 [0.40,0.69]* | **0.624 [0.40,0.82]*** | 0.468 [0.33,0.63]* | 0.391 [0.26,0.51]* |
| `head_disagreement` | 0.481 [0.33,0.68]*† | 0.606 [0.32,0.82]*† | 0.505 [0.34,0.71]*† | 0.402 [0.23,0.56]*† |
| `w` | 0.508 [0.39,0.62]* | 0.543 [0.37,0.73]* | 0.483 [0.35,0.65]* | 0.378 [0.26,0.52]* |
| `kl` | 0.411 [0.27,0.57]† | 0.412 [0.30,0.50]† | 0.403 [0.22,0.59]† | 0.401 [0.30,0.52]† |
| `attn_rollout` | 0.397 [0.24,0.56] | 0.397 [0.20,0.60] | 0.386 [0.20,0.59] | **0.276 [0.10,0.44]** |
| _base rate_ | 0.90 | 0.89 | 0.92 | 0.86 |
| _control rand / perm_ | 0.37 / 0.5±0.05 | 0.48 / 0.5±0.05 | 0.54 / 0.5±0.06 | 0.60 / 0.5±0.05 |

### pooled

| signal | Qwen-7B | Gemma-12B | Gemma-27B | Llama-70B |
|---|---|---|---|---|
| `surprisal` | 0.354 [0.27,0.43]*† | 0.349 [0.29,0.42]*† | 0.402 [0.33,0.48]*† | 0.426 [0.37,0.49]* |
| `entropy` | 0.629 [0.53,0.73]*† | 0.621 [0.54,0.70]*† | 0.551 [0.45,0.63]* | 0.615 [0.53,0.68]*† |
| `varentropy` | 0.387 [0.27,0.50]* | 0.378 [0.29,0.46]*† | 0.452 [0.36,0.55]* | 0.406 [0.33,0.49]*† |
| `resid_jump` | 0.507 [0.45,0.56]* | 0.304 [0.24,0.39]*† | 0.335 [0.26,0.43]*† | 0.490 [0.43,0.55]* |
| `temporal_kl` | 0.590 [0.49,0.69]* | 0.495 [0.41,0.59]* | 0.539 [0.45,0.64]* | 0.593 [0.51,0.67]*† |
| `lookback_ratio` | 0.575 [0.50,0.67]* | 0.602 [0.52,0.69]*† | 0.600 [0.51,0.69]* | 0.581 [0.50,0.67]* |
| `sink_drain` | 0.672 [0.59,0.74]*† | 0.684 [0.61,0.75]*† | 0.635 [0.55,0.70]*† | 0.604 [0.52,0.68]*† |
| `head_disagreement` | 0.761 [0.68,0.82]*† | 0.750 [0.67,0.81]*† | 0.689 [0.59,0.76]*† | **0.652 [0.57,0.72]*†** |
| `w` | 0.505 [0.48,0.53]* | 0.504 [0.46,0.56]* | 0.482 [0.43,0.53]* | 0.461 [0.40,0.51]* |
| `kl` | 0.544 [0.47,0.62]† | 0.510 [0.45,0.58]† | 0.514 [0.45,0.59]† | 0.593 [0.52,0.67]† |
| `attn_rollout` | **0.185 [0.10,0.30]†** | **0.203 [0.12,0.32]†** | **0.258 [0.16,0.38]†** | 0.383 [0.26,0.50] |
| _base rate_ | 0.50 | 0.55 | 0.63 | 0.62 |
| _control rand / perm_ | 0.49 / 0.5±0.02 | 0.50 / 0.5±0.02 | 0.54 / 0.5±0.02 | 0.54 / 0.5±0.02 |

## Headline: the number

Best **blind** pre-pass signal for predicting where the NLA is on-task, pooled over threats, with its 95% interval:

| model | best blind signal | AUROC [95% CI] |
|---|---|---|
| Qwen-7B | `head_disagreement` | 0.761 [0.68, 0.82] |
| Gemma-12B | `head_disagreement` | 0.750 [0.67, 0.81] |
| Gemma-27B | `head_disagreement` | 0.689 [0.59, 0.76] |
| Llama-70B | `head_disagreement` | 0.652 [0.57, 0.72] |
