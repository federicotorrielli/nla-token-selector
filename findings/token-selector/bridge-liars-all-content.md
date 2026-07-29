# The bridge on real lies (Liars' Bench, on-policy) — ALL tokens (chat template + input + response)

> Bridge results. At each token we run the NLA, ask the judge (nvidia/DeepSeek-V4-Flash-NVFP4) whether its explanation is *on-task* for the threat, and test two things: is the NLA more on-task inside the flagged span (Q1), and does a cheap one-pass signal predict where it is on-task (Q2). AUROC: 0.5 is chance; below 0.5 the signal points the other way. Intervals are case-cluster bootstraps over whole transcripts. Pre-specified primary signal(s): head_disagreement.

## Q1 — is the NLA more on-task inside the flagged span?

On-task rate inside vs outside the planted span, the lift, and its case-cluster bootstrap 95% interval (2000 resamples over whole transcripts). A lift whose interval clears 0 is a real localization of the NLA's on-task-ness to the flagged span.

| model | threat | inside | outside | lift [95% CI] |
|---|---|---|---|---|
| Gemma-27B | deception | 0.02 (n=666839) | 0.00 (n=690645) | **+0.01** [+0.01, +0.01] ✓ |
| Llama-70B | deception | 0.05 (n=257759) | 0.01 (n=248567) | **+0.04** [+0.04, +0.04] ✓ |

## Region composition

Share of all tokens and NLA on-task rate per region (template = chat-template scaffolding outside any message content).

| model | region | share | on-task |
|---|---|---|---|
| Gemma-27B | assistant_prior | 0.35 | 0.01 |
| Gemma-27B | user | 0.35 | 0.00 |
| Gemma-27B | assistant | 0.15 | 0.02 |
| Gemma-27B | system | 0.15 | 0.03 |
| Llama-70B | user | 0.52 | 0.00 |
| Llama-70B | assistant_prior | 0.18 | 0.00 |
| Llama-70B | system | 0.16 | 0.11 |
| Llama-70B | assistant | 0.14 | 0.03 |

## Q2 — which cheap pre-pass signal predicts where the NLA is on-task?

Full table: AUROC of **every** signal vs the judge's on-task label, one row per signal, one column per model, with a case-cluster bootstrap 95% interval. The **best signal for the task is bolded** per model. `*` = blind (one pass, no counterfactual). `†` = passes Benjamini-Hochberg FDR at q=0.05 over the exploratory family; the pre-specified primaries are confirmatory and shown without penalty. The last two rows give the on-task base rate and two controls (a random-normal score, and the top signal's AUROC under label permutation), both of which sit at ~0.5 when the machinery is honest.

### deception

| signal | Gemma-27B | Llama-70B |
|---|---|---|
| `surprisal` | 0.601 [0.60,0.61]*† | 0.578 [0.57,0.58]*† |
| `entropy` | 0.700 [0.70,0.71]*† | 0.631 [0.62,0.64]*† |
| `varentropy` | 0.711 [0.71,0.72]*† | 0.623 [0.62,0.63]*† |
| `temporal_kl` | 0.387 [0.38,0.39]*† | 0.366 [0.36,0.38]*† |
| `resid_jump` | 0.313 [0.30,0.32]*† | 0.509 [0.50,0.51]*† |
| `lookback_ratio` | 0.457 [0.45,0.47]*† | 0.483 [0.47,0.49]*† |
| `sink_drain` | 0.361 [0.34,0.38]*† | 0.270 [0.26,0.28]*† |
| `head_disagreement` | **0.287 [0.27,0.30]*†** | **0.252 [0.24,0.26]*†** |
| `norm_ratio` | 0.549 [0.54,0.56]*† | 0.350 [0.34,0.36]*† |
| `peak_ratio` | 0.526 [0.52,0.54]*† | 0.571 [0.56,0.58]*† |
| `dominant_mass` | 0.683 [0.67,0.69]*† | 0.570 [0.56,0.58]*† |
| `resid_jump_nla` | 0.502 [0.50,0.51]* | 0.417 [0.41,0.43]*† |
| _base rate_ | 0.01 | 0.03 |
| _control rand / perm_ | 0.50 / 0.5±0.00 | 0.50 / 0.5±0.00 |

## Headline: the number

Best **blind** pre-pass signal for predicting where the NLA is on-task, pooled over threats, with its 95% interval:

| model | best blind signal | AUROC [95% CI] |
|---|---|---|
| Gemma-27B | `head_disagreement` | 0.287 [0.27, 0.30] |
| Llama-70B | `head_disagreement` | 0.252 [0.24, 0.26] |
