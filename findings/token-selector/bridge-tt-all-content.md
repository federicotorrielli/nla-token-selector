# The bridge on Tensor Trust (prompt hijacking, response side) — ALL tokens (chat template + input + response)

> Bridge results. At each token we run the NLA, ask the judge (nvidia/DeepSeek-V4-Flash-NVFP4) whether its explanation is *on-task* for the threat, and test two things: is the NLA more on-task inside the flagged span (Q1), and does a cheap one-pass signal predict where it is on-task (Q2). AUROC: 0.5 is chance; below 0.5 the signal points the other way. Intervals are case-cluster bootstraps over whole transcripts. Pre-specified primary signal(s): head_disagreement.

## Q1 — is the NLA more on-task inside the flagged span?

On-task rate inside vs outside the planted span, the lift, and its case-cluster bootstrap 95% interval (2000 resamples over whole transcripts). A lift whose interval clears 0 is a real localization of the NLA's on-task-ness to the flagged span.

| model | threat | inside | outside | lift [95% CI] |
|---|---|---|---|---|
| Qwen-7B | injection | 0.81 (n=342572) | 0.89 (n=165446) | **-0.08** [-0.09, -0.07] |
| Gemma-12B | injection | 0.82 (n=357411) | 0.92 (n=170941) | **-0.10** [-0.11, -0.09] |
| Gemma-27B | injection | 0.84 (n=360536) | 0.91 (n=174416) | **-0.07** [-0.08, -0.06] |
| Llama-70B | injection | 0.63 (n=345330) | 0.82 (n=165000) | **-0.19** [-0.21, -0.17] |

## Region composition

Share of all tokens and NLA on-task rate per region (template = chat-template scaffolding outside any message content).

| model | region | share | on-task |
|---|---|---|---|
| Qwen-7B | user | 0.55 | 0.80 |
| Qwen-7B | system | 0.42 | 0.88 |
| Qwen-7B | assistant | 0.03 | 0.88 |
| Gemma-12B | user | 0.55 | 0.79 |
| Gemma-12B | system | 0.42 | 0.92 |
| Gemma-12B | assistant | 0.04 | 0.89 |
| Gemma-27B | user | 0.54 | 0.82 |
| Gemma-27B | system | 0.41 | 0.90 |
| Gemma-27B | assistant | 0.05 | 0.91 |
| Llama-70B | user | 0.55 | 0.56 |
| Llama-70B | system | 0.42 | 0.83 |
| Llama-70B | assistant | 0.04 | 0.87 |

## Q2 — which cheap pre-pass signal predicts where the NLA is on-task?

Full table: AUROC of **every** signal vs the judge's on-task label, one row per signal, one column per model, with a case-cluster bootstrap 95% interval. The **best signal for the task is bolded** per model. `*` = blind (one pass, no counterfactual). `†` = passes Benjamini-Hochberg FDR at q=0.05 over the exploratory family; the pre-specified primaries are confirmatory and shown without penalty. The last two rows give the on-task base rate and two controls (a random-normal score, and the top signal's AUROC under label permutation), both of which sit at ~0.5 when the machinery is honest.

### injection

| signal | Qwen-7B | Gemma-12B | Gemma-27B | Llama-70B |
|---|---|---|---|---|
| `surprisal` | 0.529 [0.52,0.54]*† | 0.629 [0.62,0.64]*† | 0.592 [0.58,0.60]*† | 0.661 [0.65,0.67]*† |
| `entropy` | 0.526 [0.52,0.54]*† | 0.605 [0.60,0.61]*† | 0.587 [0.58,0.60]*† | 0.661 [0.65,0.67]*† |
| `varentropy` | 0.512 [0.50,0.52]*† | 0.594 [0.59,0.60]*† | 0.581 [0.57,0.59]*† | 0.630 [0.62,0.64]*† |
| `temporal_kl` | 0.622 [0.61,0.63]*† | 0.646 [0.63,0.66]*† | 0.603 [0.59,0.62]*† | 0.658 [0.64,0.67]*† |
| `resid_jump` | **0.648 [0.64,0.66]*†** | 0.693 [0.68,0.70]*† | 0.653 [0.64,0.66]*† | 0.721 [0.71,0.73]*† |
| `lookback_ratio` | 0.496 [0.49,0.50]*† | 0.495 [0.49,0.50]*† | 0.490 [0.49,0.49]*† | 0.484 [0.48,0.49]*† |
| `sink_drain` | 0.584 [0.58,0.59]*† | 0.511 [0.50,0.52]* | 0.560 [0.55,0.57]*† | 0.527 [0.52,0.54]*† |
| `head_disagreement` | 0.519 [0.51,0.53]*† | 0.330 [0.32,0.34]*† | 0.454 [0.44,0.46]*† | 0.385 [0.38,0.39]*† |
| `w` | 0.596 [0.59,0.60]*† | 0.542 [0.53,0.55]*† | 0.572 [0.56,0.58]*† | 0.572 [0.57,0.58]*† |
| `norm_ratio` | 0.533 [0.52,0.54]*† | **0.740 [0.73,0.75]*†** | **0.670 [0.66,0.68]*†** | 0.682 [0.67,0.69]*† |
| `peak_ratio` | 0.443 [0.44,0.45]*† | 0.697 [0.69,0.70]*† | 0.381 [0.37,0.39]*† | 0.508 [0.50,0.51]*† |
| `dominant_mass` | 0.426 [0.42,0.43]*† | 0.698 [0.69,0.70]*† | 0.452 [0.45,0.46]*† | 0.316 [0.31,0.33]*† |
| `resid_jump_nla` | 0.637 [0.63,0.65]*† | 0.711 [0.70,0.72]*† | 0.666 [0.66,0.68]*† | **0.729 [0.72,0.74]*†** |
| _base rate_ | 0.84 | 0.85 | 0.86 | 0.69 |
| _control rand / perm_ | 0.50 / 0.5±0.00 | 0.50 / 0.5±0.00 | 0.50 / 0.5±0.00 | 0.50 / 0.5±0.00 |

## Headline: the number

Best **blind** pre-pass signal for predicting where the NLA is on-task, pooled over threats, with its 95% interval:

| model | best blind signal | AUROC [95% CI] |
|---|---|---|
| Qwen-7B | `resid_jump` | 0.648 [0.64, 0.66] |
| Gemma-12B | `norm_ratio` | 0.740 [0.73, 0.75] |
| Gemma-27B | `norm_ratio` | 0.670 [0.66, 0.68] |
| Llama-70B | `resid_jump_nla` | 0.729 [0.72, 0.74] |
