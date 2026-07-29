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
| Qwen-7B | assistant | 0.68 | 0.31 |
| Qwen-7B | user | 0.32 | 0.18 |
| Gemma-12B | assistant | 0.67 | 0.30 |
| Gemma-12B | user | 0.33 | 0.20 |
| Gemma-27B | assistant | 0.69 | 0.29 |
| Gemma-27B | user | 0.31 | 0.32 |
| Llama-70B | assistant | 0.72 | 0.17 |
| Llama-70B | user | 0.28 | 0.26 |

## Q2 — which cheap pre-pass signal predicts where the NLA is on-task?

Full table: AUROC of **every** signal vs the judge's on-task label, one row per signal, one column per model, with a case-cluster bootstrap 95% interval. The **best signal for the task is bolded** per model. `*` = blind (one pass, no counterfactual). `†` = passes Benjamini-Hochberg FDR at q=0.05 over the exploratory family; the pre-specified primaries are confirmatory and shown without penalty. The last two rows give the on-task base rate and two controls (a random-normal score, and the top signal's AUROC under label permutation), both of which sit at ~0.5 when the machinery is honest.

### taboo

| signal | Qwen-7B | Gemma-12B | Gemma-27B | Llama-70B |
|---|---|---|---|---|
| `surprisal` | 0.621 [0.60,0.64]*† | 0.540 [0.52,0.56]*† | 0.610 [0.59,0.63]*† | 0.666 [0.64,0.69]*† |
| `entropy` | 0.696 [0.67,0.72]*† | 0.605 [0.59,0.62]*† | 0.672 [0.65,0.69]*† | 0.709 [0.69,0.73]*† |
| `varentropy` | 0.679 [0.65,0.71]*† | 0.626 [0.60,0.65]*† | 0.656 [0.63,0.68]*† | 0.720 [0.69,0.75]*† |
| `temporal_kl` | 0.453 [0.44,0.47]*† | 0.496 [0.48,0.52]* | 0.416 [0.40,0.44]*† | 0.399 [0.37,0.42]*† |
| `resid_jump` | 0.548 [0.52,0.57]*† | 0.618 [0.60,0.63]*† | 0.596 [0.58,0.62]*† | 0.640 [0.62,0.66]*† |
| `lookback_ratio` | 0.468 [0.42,0.52]* | 0.469 [0.42,0.52]* | 0.558 [0.51,0.61]*† | 0.684 [0.65,0.72]*† |
| `sink_drain` | 0.430 [0.40,0.46]*† | 0.439 [0.40,0.48]*† | 0.354 [0.32,0.39]*† | 0.270 [0.25,0.29]*† |
| `head_disagreement` | 0.441 [0.41,0.47]*† | 0.451 [0.42,0.49]*† | 0.362 [0.33,0.39]*† | 0.339 [0.32,0.36]*† |
| `norm_ratio` | 0.469 [0.44,0.49]*† | 0.530 [0.49,0.57]* | 0.430 [0.39,0.47]*† | **0.735 [0.69,0.77]*†** |
| `peak_ratio` | 0.705 [0.68,0.73]*† | 0.688 [0.67,0.71]*† | 0.618 [0.60,0.64]*† | 0.703 [0.66,0.75]*† |
| `dominant_mass` | **0.760 [0.74,0.78]*†** | **0.696 [0.68,0.72]*†** | **0.715 [0.68,0.74]*†** | 0.699 [0.65,0.75]*† |
| `resid_jump_nla` | 0.401 [0.38,0.42]*† | 0.421 [0.41,0.44]*† | 0.365 [0.35,0.39]*† | 0.711 [0.68,0.74]*† |
| _base rate_ | 0.27 | 0.27 | 0.30 | 0.19 |
| _control rand / perm_ | 0.48 / 0.5±0.01 | 0.49 / 0.5±0.01 | 0.49 / 0.5±0.01 | 0.49 / 0.5±0.01 |

## Headline: the number

Best **blind** pre-pass signal for predicting where the NLA is on-task, pooled over threats, with its 95% interval:

| model | best blind signal | AUROC [95% CI] |
|---|---|---|
| Qwen-7B | `dominant_mass` | 0.760 [0.74, 0.78] |
| Gemma-12B | `dominant_mass` | 0.696 [0.68, 0.72] |
| Gemma-27B | `dominant_mass` | 0.715 [0.68, 0.74] |
| Llama-70B | `norm_ratio` | 0.735 [0.69, 0.77] |
