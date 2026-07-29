# The bridge on Tensor Trust (prompt hijacking, response side) — ALL tokens (chat template + input + response)

> Bridge results. At each token we run the NLA, ask the judge (nvidia/DeepSeek-V4-Flash-NVFP4) whether its explanation is *on-task* for the threat, and test two things: is the NLA more on-task inside the flagged span (Q1), and does a cheap one-pass signal predict where it is on-task (Q2). AUROC: 0.5 is chance; below 0.5 the signal points the other way. Intervals are case-cluster bootstraps over whole transcripts. Pre-specified primary signal(s): head_disagreement.

## Q1 — is the NLA more on-task inside the flagged span?

On-task rate inside vs outside the planted span, the lift, and its case-cluster bootstrap 95% interval (2000 resamples over whole transcripts). A lift whose interval clears 0 is a real localization of the NLA's on-task-ness to the flagged span.

| model | threat | inside | outside | lift [95% CI] |
|---|---|---|---|---|
| Qwen-7B | injection | 0.81 (n=352740) | 0.89 (n=175615) | **-0.07** [-0.08, -0.06] |
| Gemma-12B | injection | 0.82 (n=365135) | 0.91 (n=178645) | **-0.09** [-0.10, -0.08] |
| Gemma-27B | injection | 0.84 (n=368280) | 0.91 (n=182140) | **-0.07** [-0.08, -0.06] |
| Llama-70B | injection | 0.63 (n=372419) | 0.79 (n=192160) | **-0.16** [-0.18, -0.15] |

## Region composition

Share of all tokens and NLA on-task rate per region (template = chat-template scaffolding outside any message content).

| model | region | share | on-task |
|---|---|---|---|
| Qwen-7B | user | 0.53 | 0.80 |
| Qwen-7B | system | 0.40 | 0.88 |
| Qwen-7B | template | 0.04 | 0.85 |
| Qwen-7B | assistant | 0.03 | 0.88 |
| Gemma-12B | user | 0.53 | 0.79 |
| Gemma-12B | system | 0.40 | 0.92 |
| Gemma-12B | assistant | 0.03 | 0.89 |
| Gemma-12B | template | 0.03 | 0.82 |
| Gemma-27B | user | 0.53 | 0.82 |
| Gemma-27B | system | 0.40 | 0.90 |
| Gemma-27B | assistant | 0.04 | 0.91 |
| Gemma-27B | template | 0.03 | 0.99 |
| Llama-70B | user | 0.49 | 0.56 |
| Llama-70B | system | 0.38 | 0.83 |
| Llama-70B | template | 0.10 | 0.62 |
| Llama-70B | assistant | 0.03 | 0.87 |

## Q2 — which cheap pre-pass signal predicts where the NLA is on-task?

Full table: AUROC of **every** signal vs the judge's on-task label, one row per signal, one column per model, with a case-cluster bootstrap 95% interval. The **best signal for the task is bolded** per model. `*` = blind (one pass, no counterfactual). `†` = passes Benjamini-Hochberg FDR at q=0.05 over the exploratory family; the pre-specified primaries are confirmatory and shown without penalty. The last two rows give the on-task base rate and two controls (a random-normal score, and the top signal's AUROC under label permutation), both of which sit at ~0.5 when the machinery is honest.

### injection

| signal | Qwen-7B | Gemma-12B | Gemma-27B | Llama-70B |
|---|---|---|---|---|
| `surprisal` | 0.522 [0.51,0.53]*† | 0.629 [0.62,0.64]*† | 0.599 [0.59,0.61]*† | 0.612 [0.60,0.62]*† |
| `entropy` | 0.510 [0.50,0.52]* | 0.598 [0.59,0.61]*† | 0.589 [0.58,0.60]*† | 0.611 [0.60,0.62]*† |
| `varentropy` | 0.498 [0.49,0.51]* | 0.583 [0.57,0.59]*† | 0.582 [0.57,0.59]*† | 0.583 [0.57,0.59]*† |
| `temporal_kl` | 0.627 [0.62,0.63]*† | 0.648 [0.63,0.66]*† | 0.601 [0.59,0.61]*† | 0.650 [0.64,0.66]*† |
| `resid_jump` | **0.639 [0.63,0.65]*†** | 0.690 [0.68,0.70]*† | 0.661 [0.65,0.67]*† | 0.679 [0.67,0.69]*† |
| `lookback_ratio` | 0.496 [0.49,0.50]*† | 0.495 [0.49,0.50]*† | 0.491 [0.49,0.49]*† | 0.485 [0.48,0.49]*† |
| `sink_drain` | 0.596 [0.59,0.60]*† | 0.522 [0.51,0.53]*† | 0.553 [0.54,0.56]*† | 0.547 [0.54,0.55]*† |
| `head_disagreement` | 0.529 [0.52,0.54]*† | 0.348 [0.34,0.36]*† | 0.449 [0.44,0.46]*† | 0.426 [0.42,0.43]*† |
| `norm_ratio` | 0.517 [0.51,0.53]*† | **0.719 [0.71,0.73]*†** | 0.671 [0.66,0.68]*† | 0.642 [0.63,0.65]*† |
| `peak_ratio` | 0.443 [0.44,0.45]*† | 0.670 [0.66,0.68]*† | 0.393 [0.38,0.40]*† | 0.507 [0.50,0.51]*† |
| `dominant_mass` | 0.430 [0.42,0.44]*† | 0.671 [0.66,0.68]*† | 0.450 [0.44,0.46]*† | 0.343 [0.33,0.35]*† |
| `resid_jump_nla` | 0.619 [0.61,0.63]*† | 0.707 [0.70,0.72]*† | **0.672 [0.66,0.68]*†** | **0.695 [0.68,0.71]*†** |
| _base rate_ | 0.84 | 0.85 | 0.86 | 0.68 |
| _control rand / perm_ | 0.50 / 0.5±0.00 | 0.50 / 0.5±0.00 | 0.50 / 0.5±0.00 | 0.50 / 0.5±0.00 |

## Headline: the number

Best **blind** pre-pass signal for predicting where the NLA is on-task, pooled over threats, with its 95% interval:

| model | best blind signal | AUROC [95% CI] |
|---|---|---|
| Qwen-7B | `resid_jump` | 0.639 [0.63, 0.65] |
| Gemma-12B | `norm_ratio` | 0.719 [0.71, 0.73] |
| Gemma-27B | `resid_jump_nla` | 0.672 [0.66, 0.68] |
| Llama-70B | `resid_jump_nla` | 0.695 [0.68, 0.71] |
