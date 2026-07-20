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

AUROC of each signal vs the judge's on-task label, with a case-cluster bootstrap 95% interval. `*` = blind (one pass, no counterfactual). `†` = passes Benjamini-Hochberg FDR at q=0.05 over the exploratory family; the pre-specified primaries (`head_disagreement`, `kl`) are confirmatory and shown without penalty. Base rate = share of on-task tokens. Two controls per cell: a random-normal score and the top-signal AUROC under label permutation both sit at ~0.5, confirming the machinery invents no signal.

### injection (input span)

| model | base rate | top signals: AUROC [95% CI] | controls (rand / perm) |
|---|---|---|---|
| Qwen-7B | 0.21 | in_entropy*†=0.673 [0.67,0.68]  in_attention*†=0.575 [0.57,0.58]  in_surprisal*=0.496 [0.49,0.50] | 0.50 / 0.5±0.00 |
| Gemma-12B | 0.14 | in_entropy*†=0.595 [0.59,0.60]  in_attention*†=0.509 [0.50,0.52]  in_surprisal*=0.501 [0.49,0.51] | 0.50 / 0.5±0.00 |
| Gemma-27B | 0.15 | in_entropy*†=0.540 [0.53,0.55]  in_surprisal*†=0.474 [0.47,0.48]  in_attention*=0.501 [0.49,0.51] | 0.50 / 0.5±0.00 |
| Llama-70B | 0.14 | in_entropy*†=0.584 [0.58,0.59]  in_attention*†=0.561 [0.56,0.57]  in_surprisal*†=0.526 [0.52,0.53] | 0.50 / 0.5±0.00 |

## Headline: the number

Best **blind** pre-pass signal for predicting where the NLA is on-task, pooled over threats, with its 95% interval:

| model | best blind signal | AUROC [95% CI] |
|---|---|---|
| Qwen-7B | `in_entropy` | 0.673 [0.67, 0.68] |
| Gemma-12B | `in_entropy` | 0.595 [0.59, 0.60] |
| Gemma-27B | `in_entropy` | 0.540 [0.53, 0.55] |
| Llama-70B | `in_entropy` | 0.584 [0.58, 0.59] |
