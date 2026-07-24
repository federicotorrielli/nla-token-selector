# The bridge on real injection (OpenPromptInjection, all 800 cases)

> Bridge results. At each token we run the NLA, ask the judge (nvidia/DeepSeek-V4-Flash-NVFP4) whether its explanation is *on-task* for the threat, and test two things: is the NLA more on-task inside the flagged span (Q1), and does a cheap one-pass signal predict where it is on-task (Q2). AUROC: 0.5 is chance; below 0.5 the signal points the other way. Intervals are case-cluster bootstraps over whole transcripts. Pre-specified primary signal(s): (none — exploratory).

## Q1 — is the NLA more on-task inside the flagged span?

On-task rate inside vs outside the planted span, the lift, and its case-cluster bootstrap 95% interval (2000 resamples over whole transcripts). A lift whose interval clears 0 is a real localization of the NLA's on-task-ness to the flagged span.

| model | threat | inside | outside | lift [95% CI] |
|---|---|---|---|---|
| Qwen-7B | injection (input span) | 0.34 (n=50520) | 0.02 (n=35250) | **+0.33** [+0.32, +0.33] ✓ |
| Gemma-12B | injection (input span) | 0.22 (n=50160) | 0.02 (n=34790) | **+0.21** [+0.20, +0.21] ✓ |
| Gemma-27B | injection (input span) | 0.24 (n=50160) | 0.00 (n=34790) | **+0.24** [+0.24, +0.24] ✓ |
| Llama-70B | injection (input span) | 0.23 (n=48780) | 0.01 (n=34360) | **+0.23** [+0.22, +0.23] ✓ |

## Q2 — which cheap pre-pass signal predicts where the NLA is on-task?

Full table: AUROC of **every** signal vs the judge's on-task label, one row per signal, one column per model, with a case-cluster bootstrap 95% interval. The **best signal for the task is bolded** per model. `*` = blind (one pass, no counterfactual). `†` = passes Benjamini-Hochberg FDR at q=0.05 over the exploratory family; the pre-specified primaries are confirmatory and shown without penalty. The last two rows give the on-task base rate and two controls (a random-normal score, and the top signal's AUROC under label permutation), both of which sit at ~0.5 when the machinery is honest.

### injection (input span)

| signal | Qwen-7B | Gemma-12B | Gemma-27B | Llama-70B |
|---|---|---|---|---|
| `surprisal` | 0.496 [0.49,0.50]* | 0.501 [0.49,0.51]* | 0.474 [0.47,0.48]*† | 0.526 [0.52,0.53]*† |
| `entropy` | 0.327 [0.32,0.33]*† | 0.405 [0.40,0.41]*† | 0.460 [0.45,0.47]*† | 0.416 [0.41,0.42]*† |
| `varentropy` | **0.314 [0.31,0.32]*†** | 0.406 [0.40,0.41]*† | 0.447 [0.44,0.45]*† | 0.399 [0.39,0.41]*† |
| `resid_jump` | 0.549 [0.54,0.55]*† | 0.531 [0.52,0.54]*† | 0.505 [0.50,0.51]* | 0.573 [0.57,0.58]*† |
| `lookback_ratio` | 0.530 [0.52,0.54]*† | 0.543 [0.53,0.55]*† | 0.499 [0.49,0.51]* | 0.481 [0.47,0.49]*† |
| `sink_drain` | 0.631 [0.62,0.64]*† | **0.669 [0.66,0.68]*†** | **0.715 [0.71,0.72]*†** | **0.678 [0.67,0.68]*†** |
| `head_disagreement` | 0.582 [0.57,0.59]*† | 0.624 [0.62,0.63]*† | 0.669 [0.66,0.68]*† | 0.635 [0.63,0.64]*† |
| `attn_rollout` | 0.575 [0.57,0.58]† | 0.509 [0.50,0.52]† | 0.501 [0.49,0.51] | 0.561 [0.56,0.57]† |
| _base rate_ | 0.21 | 0.14 | 0.15 | 0.14 |
| _control rand / perm_ | 0.50 / 0.5±0.00 | 0.50 / 0.5±0.00 | 0.50 / 0.5±0.00 | 0.50 / 0.5±0.00 |

## Headline: the number

Best **blind** pre-pass signal for predicting where the NLA is on-task, pooled over threats, with its 95% interval:

| model | best blind signal | AUROC [95% CI] |
|---|---|---|
| Qwen-7B | `varentropy` | 0.314 [0.31, 0.32] |
| Gemma-12B | `sink_drain` | 0.669 [0.66, 0.68] |
| Gemma-27B | `sink_drain` | 0.715 [0.71, 0.72] |
| Llama-70B | `sink_drain` | 0.678 [0.67, 0.68] |
