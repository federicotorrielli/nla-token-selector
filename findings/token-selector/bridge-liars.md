# The bridge on real lies (Liars' Bench, on-policy)

> Bridge results. At each token we run the NLA, ask the judge (nvidia/DeepSeek-V4-Flash-NVFP4) whether its explanation is *on-task* for the threat, and test two things: is the NLA more on-task inside the flagged span (Q1), and does a cheap one-pass signal predict where it is on-task (Q2). AUROC: 0.5 is chance; below 0.5 the signal points the other way. Intervals are case-cluster bootstraps over whole transcripts. Pre-specified primary signal(s): head_disagreement.

## Q1 — is the NLA more on-task inside the flagged span?

On-task rate inside vs outside the planted span, the lift, and its case-cluster bootstrap 95% interval (2000 resamples over whole transcripts). A lift whose interval clears 0 is a real localization of the NLA's on-task-ness to the flagged span.

| model | threat | inside | outside | lift [95% CI] |
|---|---|---|---|---|
| Gemma-27B | deception | 0.05 (n=27691) | 0.01 (n=26678) | **+0.04** [+0.04, +0.05] ✓ |
| Llama-70B | deception | 0.06 (n=25836) | 0.01 (n=13179) | **+0.05** [+0.05, +0.05] ✓ |

## Q2 — which cheap pre-pass signal predicts where the NLA is on-task?

AUROC of each signal vs the judge's on-task label, with a case-cluster bootstrap 95% interval. `*` = blind (one pass, no counterfactual). `†` = passes Benjamini-Hochberg FDR at q=0.05 over the exploratory family; the pre-specified primaries (`head_disagreement`, `kl`) are confirmatory and shown without penalty. Base rate = share of on-task tokens. Two controls per cell: a random-normal score and the top-signal AUROC under label permutation both sit at ~0.5, confirming the machinery invents no signal.

### deception

| model | base rate | top signals: AUROC [95% CI] | controls (rand / perm) |
|---|---|---|---|
| Gemma-27B | 0.03 | sink_drain*†=0.632 [0.61,0.65]  varentropy*†=0.589 [0.57,0.60]  entropy*†=0.587 [0.57,0.60]  surprisal*†=0.582 [0.57,0.60]  head_disagreement*†=0.573 [0.55,0.59] | 0.50 / 0.5±0.01 |
| Llama-70B | 0.04 | sink_drain*†=0.676 [0.66,0.69]  head_disagreement*†=0.644 [0.62,0.66]  varentropy*†=0.590 [0.58,0.60]  entropy*†=0.588 [0.57,0.60]  surprisal*†=0.587 [0.57,0.60] | 0.50 / 0.5±0.01 |

## Headline: the number

Best **blind** pre-pass signal for predicting where the NLA is on-task, pooled over threats, with its 95% interval:

| model | best blind signal | AUROC [95% CI] |
|---|---|---|
| Gemma-27B | `sink_drain` | 0.632 [0.61, 0.65] |
| Llama-70B | `sink_drain` | 0.676 [0.66, 0.69] |
