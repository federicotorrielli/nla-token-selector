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

AUROC of each signal vs the judge's on-task label, with a case-cluster bootstrap 95% interval. `*` = blind (one pass, no counterfactual). `†` = passes Benjamini-Hochberg FDR at q=0.05 over the exploratory family; the pre-specified primaries (`head_disagreement`, `kl`) are confirmatory and shown without penalty. Base rate = share of on-task tokens. Two controls per cell: a random-normal score and the top-signal AUROC under label permutation both sit at ~0.5, confirming the machinery invents no signal.

### eval-awareness

| model | base rate | top signals: AUROC [95% CI] | controls (rand / perm) |
|---|---|---|---|
| Qwen-7B | 0.09 | kl†=0.751 [0.66,0.85]  head_disagreement*†=0.750 [0.67,0.85]  sink_drain*†=0.707 [0.61,0.81]  lookback_ratio*†=0.640 [0.54,0.73]  varentropy*=0.635 [0.50,0.76] | 0.54 / 0.5±0.05 |
| Gemma-12B | 0.20 | lookback_ratio*†=0.660 [0.57,0.74]  sink_drain*†=0.653 [0.58,0.73]  head_disagreement*†=0.646 [0.57,0.74]  kl†=0.636 [0.55,0.72]  resid_jump*†=0.366 [0.28,0.43] | 0.54 / 0.5±0.04 |
| Gemma-27B | 0.33 | kl†=0.630 [0.56,0.70]  lookback_ratio*=0.628 [0.52,0.73]  sink_drain*=0.602 [0.50,0.70]  head_disagreement*†=0.579 [0.46,0.68]  temporal_kl*=0.578 [0.49,0.67] | 0.53 / 0.5±0.03 |
| Llama-70B | 0.37 | kl†=0.753 [0.69,0.81]  head_disagreement*†=0.656 [0.56,0.74]  sink_drain*†=0.632 [0.55,0.72]  lookback_ratio*=0.574 [0.47,0.67]  resid_jump*†=0.438 [0.39,0.48] | 0.49 / 0.5±0.04 |

### injection

| model | base rate | top signals: AUROC [95% CI] | controls (rand / perm) |
|---|---|---|---|
| Qwen-7B | 0.90 | resid_jump*=0.330 [0.19,0.47]  attn_rollout=0.397 [0.24,0.56]  lookback_ratio*=0.603 [0.47,0.70]  kl†=0.411 [0.27,0.57]  surprisal*=0.580 [0.50,0.67] | 0.37 / 0.5±0.05 |
| Gemma-12B | 0.89 | sink_drain*=0.624 [0.40,0.82]  head_disagreement*†=0.606 [0.32,0.82]  attn_rollout=0.397 [0.20,0.60]  lookback_ratio*=0.601 [0.46,0.76]  kl†=0.412 [0.30,0.50] | 0.48 / 0.5±0.05 |
| Gemma-27B | 0.92 | varentropy*=0.718 [0.57,0.83]  entropy*=0.282 [0.17,0.43]  resid_jump*=0.348 [0.20,0.49]  surprisal*=0.625 [0.44,0.76]  attn_rollout=0.386 [0.20,0.59] | 0.54 / 0.5±0.06 |
| Llama-70B | 0.86 | attn_rollout=0.276 [0.10,0.44]  lookback_ratio*=0.645 [0.50,0.80]  w*=0.378 [0.26,0.52]  sink_drain*=0.391 [0.26,0.51]  kl†=0.401 [0.30,0.52] | 0.60 / 0.5±0.05 |

### pooled

| model | base rate | top signals: AUROC [95% CI] | controls (rand / perm) |
|---|---|---|---|
| Qwen-7B | 0.50 | attn_rollout†=0.185 [0.10,0.30]  head_disagreement*†=0.761 [0.68,0.82]  sink_drain*†=0.672 [0.59,0.74]  surprisal*†=0.354 [0.27,0.43]  entropy*†=0.629 [0.53,0.73] | 0.49 / 0.5±0.02 |
| Gemma-12B | 0.55 | attn_rollout†=0.203 [0.12,0.32]  head_disagreement*†=0.750 [0.67,0.81]  resid_jump*†=0.304 [0.24,0.39]  sink_drain*†=0.684 [0.61,0.75]  surprisal*†=0.349 [0.29,0.42] | 0.50 / 0.5±0.02 |
| Gemma-27B | 0.63 | attn_rollout†=0.258 [0.16,0.38]  head_disagreement*†=0.689 [0.59,0.76]  resid_jump*†=0.335 [0.26,0.43]  sink_drain*†=0.635 [0.55,0.70]  lookback_ratio*=0.600 [0.51,0.69] | 0.54 / 0.5±0.02 |
| Llama-70B | 0.62 | head_disagreement*†=0.652 [0.57,0.72]  attn_rollout=0.383 [0.26,0.50]  entropy*†=0.615 [0.53,0.68]  sink_drain*†=0.604 [0.52,0.68]  varentropy*†=0.406 [0.33,0.49] | 0.54 / 0.5±0.02 |

## Headline: the number

Best **blind** pre-pass signal for predicting where the NLA is on-task, pooled over threats, with its 95% interval:

| model | best blind signal | AUROC [95% CI] |
|---|---|---|
| Qwen-7B | `head_disagreement` | 0.761 [0.68, 0.82] |
| Gemma-12B | `head_disagreement` | 0.750 [0.67, 0.81] |
| Gemma-27B | `head_disagreement` | 0.689 [0.59, 0.76] |
| Llama-70B | `head_disagreement` | 0.652 [0.57, 0.72] |
