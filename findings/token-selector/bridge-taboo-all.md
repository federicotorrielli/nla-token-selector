# The bridge on taboo organisms (secret word, response side) — ALL tokens (chat template + input + response)

> Bridge results. At each token we run the NLA, ask the judge (nvidia/DeepSeek-V4-Flash-NVFP4) whether its explanation is *on-task* for the threat, and test two things: is the NLA more on-task inside the flagged span (Q1), and does a cheap one-pass signal predict where it is on-task (Q2). AUROC: 0.5 is chance; below 0.5 the signal points the other way. Intervals are case-cluster bootstraps over whole transcripts. Pre-specified primary signal(s): (none — exploratory).

## Q1 — is the NLA more on-task inside the flagged span?

On-task rate inside vs outside the planted span, the lift, and its case-cluster bootstrap 95% interval (2000 resamples over whole transcripts). A lift whose interval clears 0 is a real localization of the NLA's on-task-ness to the flagged span.

| model | threat | inside | outside | lift [95% CI] |
|---|---|---|---|---|

## Region composition

Share of all tokens and NLA on-task rate per region (template = chat-template scaffolding outside any message content).

| model | region | share | on-task |
|---|---|---|---|
| Qwen-7B | assistant | 0.42 | 0.31 |
| Qwen-7B | template | 0.38 | 0.12 |
| Qwen-7B | user | 0.20 | 0.18 |
| Gemma-12B | assistant | 0.56 | 0.30 |
| Gemma-12B | user | 0.28 | 0.20 |
| Gemma-12B | template | 0.16 | 0.33 |
| Gemma-27B | assistant | 0.58 | 0.29 |
| Gemma-27B | user | 0.26 | 0.32 |
| Gemma-27B | template | 0.16 | 0.31 |
| Llama-70B | assistant | 0.44 | 0.17 |
| Llama-70B | template | 0.39 | 0.11 |
| Llama-70B | user | 0.17 | 0.26 |

## Q2 — which cheap pre-pass signal predicts where the NLA is on-task?

Full table: AUROC of **every** signal vs the judge's on-task label, one row per signal, one column per model, with a case-cluster bootstrap 95% interval. The **best signal for the task is bolded** per model. `*` = blind (one pass, no counterfactual). `†` = passes Benjamini-Hochberg FDR at q=0.05 over the exploratory family; the pre-specified primaries are confirmatory and shown without penalty. The last two rows give the on-task base rate and two controls (a random-normal score, and the top signal's AUROC under label permutation), both of which sit at ~0.5 when the machinery is honest.

### taboo

| signal | Qwen-7B | Gemma-12B | Gemma-27B | Llama-70B |
|---|---|---|---|---|
| `surprisal` | 0.522 [0.50,0.54]*† | 0.504 [0.48,0.53]* | 0.571 [0.55,0.59]*† | 0.491 [0.47,0.51]* |
| `entropy` | 0.566 [0.55,0.58]*† | 0.557 [0.53,0.58]*† | 0.604 [0.58,0.62]*† | 0.522 [0.51,0.53]*† |
| `varentropy` | 0.551 [0.53,0.57]*† | 0.601 [0.57,0.63]*† | 0.622 [0.60,0.65]*† | 0.535 [0.52,0.55]*† |
| `temporal_kl` | 0.499 [0.48,0.52]* | 0.500 [0.48,0.52]* | 0.402 [0.38,0.42]*† | 0.515 [0.50,0.53]* |
| `resid_jump` | 0.527 [0.50,0.55]*† | 0.579 [0.56,0.60]*† | 0.519 [0.50,0.54]*† | 0.549 [0.53,0.57]*† |
| `lookback_ratio` | 0.397 [0.35,0.45]*† | 0.485 [0.44,0.54]* | 0.545 [0.49,0.60]* | 0.545 [0.50,0.59]* |
| `sink_drain` | 0.598 [0.57,0.63]*† | 0.498 [0.46,0.54]* | 0.428 [0.40,0.46]*† | 0.552 [0.53,0.57]*† |
| `head_disagreement` | 0.557 [0.53,0.59]*† | 0.514 [0.48,0.55]* | 0.440 [0.41,0.47]*† | 0.568 [0.54,0.60]*† |
| `norm_ratio` | 0.381 [0.35,0.41]*† | 0.483 [0.44,0.52]* | 0.416 [0.38,0.45]*† | 0.690 [0.64,0.73]*† |
| `peak_ratio` | 0.767 [0.76,0.78]*† | 0.674 [0.66,0.69]*† | 0.572 [0.56,0.59]*† | 0.681 [0.65,0.72]*† |
| `dominant_mass` | **0.796 [0.78,0.81]*†** | **0.681 [0.67,0.69]*†** | **0.651 [0.62,0.68]*†** | 0.703 [0.67,0.74]*† |
| `resid_jump_nla` | 0.326 [0.31,0.34]*† | 0.374 [0.36,0.39]*† | 0.354 [0.34,0.37]*† | **0.723 [0.70,0.74]*†** |
| _base rate_ | 0.21 | 0.28 | 0.30 | 0.16 |
| _control rand / perm_ | 0.51 / 0.5±0.01 | 0.49 / 0.5±0.01 | 0.51 / 0.5±0.01 | 0.50 / 0.5±0.01 |

## Headline: the number

Best **blind** pre-pass signal for predicting where the NLA is on-task, pooled over threats, with its 95% interval:

| model | best blind signal | AUROC [95% CI] |
|---|---|---|
| Qwen-7B | `dominant_mass` | 0.796 [0.78, 0.81] |
| Gemma-12B | `dominant_mass` | 0.681 [0.67, 0.69] |
| Gemma-27B | `dominant_mass` | 0.651 [0.62, 0.68] |
| Llama-70B | `resid_jump_nla` | 0.723 [0.70, 0.74] |
