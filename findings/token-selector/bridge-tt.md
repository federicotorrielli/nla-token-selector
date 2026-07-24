# The bridge on Tensor Trust (prompt hijacking, response side)

> Bridge results. At each token we run the NLA, ask the judge (nvidia/DeepSeek-V4-Flash-NVFP4) whether its explanation is *on-task* for the threat, and test two things: is the NLA more on-task inside the flagged span (Q1), and does a cheap one-pass signal predict where it is on-task (Q2). AUROC: 0.5 is chance; below 0.5 the signal points the other way. Intervals are case-cluster bootstraps over whole transcripts. Pre-specified primary signal(s): head_disagreement.

## Q1 — is the NLA more on-task inside the flagged span?

On-task rate inside vs outside the planted span, the lift, and its case-cluster bootstrap 95% interval (2000 resamples over whole transcripts). A lift whose interval clears 0 is a real localization of the NLA's on-task-ness to the flagged span.

| model | threat | inside | outside | lift [95% CI] |
|---|---|---|---|---|
| Qwen-7B | injection | 0.90 (n=6103) | 0.91 (n=3651) | **-0.01** [-0.04, +0.02] |
| Gemma-12B | injection | 0.89 (n=7677) | 0.91 (n=4543) | **-0.02** [-0.04, -0.00] |
| Gemma-27B | injection | 0.91 (n=9129) | 0.94 (n=6250) | **-0.02** [-0.04, -0.00] |
| Llama-70B | injection | 0.87 (n=8306) | 0.91 (n=4251) | **-0.04** [-0.06, -0.01] |

## Q2 — which cheap pre-pass signal predicts where the NLA is on-task?

Full table: AUROC of **every** signal vs the judge's on-task label, one row per signal, one column per model, with a case-cluster bootstrap 95% interval. The **best signal for the task is bolded** per model. `*` = blind (one pass, no counterfactual). `†` = passes Benjamini-Hochberg FDR at q=0.05 over the exploratory family; the pre-specified primaries are confirmatory and shown without penalty. The last two rows give the on-task base rate and two controls (a random-normal score, and the top signal's AUROC under label permutation), both of which sit at ~0.5 when the machinery is honest.

### injection

| signal | Qwen-7B | Gemma-12B | Gemma-27B | Llama-70B |
|---|---|---|---|---|
| `surprisal` | 0.337 [0.31,0.37]*† | 0.468 [0.44,0.50]* | 0.443 [0.42,0.47]*† | 0.595 [0.57,0.62]*† |
| `entropy` | 0.326 [0.29,0.36]*† | 0.463 [0.43,0.50]* | 0.449 [0.42,0.48]*† | **0.598 [0.57,0.63]*†** |
| `varentropy` | 0.314 [0.28,0.35]*† | 0.461 [0.43,0.50]* | 0.447 [0.42,0.48]*† | 0.595 [0.57,0.62]*† |
| `resid_jump` | 0.654 [0.63,0.68]*† | 0.523 [0.50,0.55]* | 0.459 [0.43,0.49]*† | 0.510 [0.49,0.53]* |
| `lookback_ratio` | 0.727 [0.70,0.75]*† | **0.660 [0.64,0.68]*†** | **0.639 [0.62,0.66]*†** | 0.596 [0.57,0.62]*† |
| `sink_drain` | **0.735 [0.70,0.77]*†** | 0.515 [0.47,0.56]* | 0.515 [0.47,0.55]* | 0.549 [0.52,0.58]*† |
| `head_disagreement` | 0.695 [0.66,0.73]*† | 0.538 [0.50,0.58]*† | 0.540 [0.51,0.57]*† | 0.477 [0.44,0.51]*† |
| _base rate_ | 0.90 | 0.90 | 0.92 | 0.88 |
| _control rand / perm_ | 0.51 / 0.5±0.01 | 0.51 / 0.5±0.01 | 0.51 / 0.5±0.01 | 0.51 / 0.5±0.01 |

## Headline: the number

Best **blind** pre-pass signal for predicting where the NLA is on-task, pooled over threats, with its 95% interval:

| model | best blind signal | AUROC [95% CI] |
|---|---|---|
| Qwen-7B | `sink_drain` | 0.735 [0.70, 0.77] |
| Gemma-12B | `lookback_ratio` | 0.660 [0.64, 0.68] |
| Gemma-27B | `lookback_ratio` | 0.639 [0.62, 0.66] |
| Llama-70B | `entropy` | 0.598 [0.57, 0.63] |
