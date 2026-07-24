# The bridge on taboo organisms (secret word, response side)

> Bridge results. At each token we run the NLA, ask the judge (nvidia/DeepSeek-V4-Flash-NVFP4) whether its explanation is *on-task* for the threat, and test two things: is the NLA more on-task inside the flagged span (Q1), and does a cheap one-pass signal predict where it is on-task (Q2). AUROC: 0.5 is chance; below 0.5 the signal points the other way. Intervals are case-cluster bootstraps over whole transcripts. Pre-specified primary signal(s): (none — exploratory).

## Q1 — is the NLA more on-task inside the flagged span?

On-task rate inside vs outside the planted span, the lift, and its case-cluster bootstrap 95% interval (2000 resamples over whole transcripts). A lift whose interval clears 0 is a real localization of the NLA's on-task-ness to the flagged span.

| model | threat | inside | outside | lift [95% CI] |
|---|---|---|---|---|

## Q2 — which cheap pre-pass signal predicts where the NLA is on-task?

Full table: AUROC of **every** signal vs the judge's on-task label, one row per signal, one column per model, with a case-cluster bootstrap 95% interval. The **best signal for the task is bolded** per model. `*` = blind (one pass, no counterfactual). `†` = passes Benjamini-Hochberg FDR at q=0.05 over the exploratory family; the pre-specified primaries are confirmatory and shown without penalty. The last two rows give the on-task base rate and two controls (a random-normal score, and the top signal's AUROC under label permutation), both of which sit at ~0.5 when the machinery is honest.

### taboo

| signal | Qwen-7B | Gemma-12B | Gemma-27B | Llama-70B |
|---|---|---|---|---|
| `surprisal` | 0.745 [0.72,0.77]*† | 0.679 [0.65,0.70]*† | 0.699 [0.68,0.72]*† | 0.752 [0.73,0.77]*† |
| `entropy` | **0.776 [0.75,0.80]*†** | 0.734 [0.71,0.76]*† | 0.741 [0.72,0.76]*† | 0.794 [0.77,0.82]*† |
| `varentropy` | 0.768 [0.74,0.80]*† | 0.729 [0.71,0.75]*† | 0.727 [0.70,0.75]*† | 0.799 [0.78,0.82]*† |
| `resid_jump` | 0.544 [0.52,0.57]*† | 0.681 [0.66,0.71]*† | 0.656 [0.63,0.68]*† | 0.717 [0.70,0.74]*† |
| `lookback_ratio` | 0.474 [0.45,0.51]* | 0.479 [0.45,0.52]* | 0.514 [0.48,0.56]* | 0.716 [0.69,0.74]*† |
| `sink_drain` | 0.283 [0.26,0.31]*† | **0.252 [0.23,0.27]*†** | **0.228 [0.21,0.25]*†** | **0.093 [0.08,0.12]*†** |
| `head_disagreement` | 0.312 [0.29,0.33]*† | 0.275 [0.25,0.30]*† | 0.250 [0.22,0.29]*† | 0.226 [0.21,0.25]*† |
| _base rate_ | 0.37 | 0.34 | 0.33 | 0.21 |
| _control rand / perm_ | 0.51 / 0.5±0.01 | 0.51 / 0.5±0.01 | 0.51 / 0.5±0.01 | 0.49 / 0.5±0.01 |

## Headline: the number

Best **blind** pre-pass signal for predicting where the NLA is on-task, pooled over threats, with its 95% interval:

| model | best blind signal | AUROC [95% CI] |
|---|---|---|
| Qwen-7B | `entropy` | 0.776 [0.75, 0.80] |
| Gemma-12B | `sink_drain` | 0.252 [0.23, 0.27] |
| Gemma-27B | `sink_drain` | 0.228 [0.21, 0.25] |
| Llama-70B | `sink_drain` | 0.093 [0.08, 0.12] |
