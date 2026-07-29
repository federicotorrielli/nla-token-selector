# The bridge on real injection (OpenPromptInjection, all 800 cases) — ALL tokens (chat template + input + response)

> Bridge results. At each token we run the NLA, ask the judge (nvidia/DeepSeek-V4-Flash-NVFP4) whether its explanation is *on-task* for the threat, and test two things: is the NLA more on-task inside the flagged span (Q1), and does a cheap one-pass signal predict where it is on-task (Q2). AUROC: 0.5 is chance; below 0.5 the signal points the other way. Intervals are case-cluster bootstraps over whole transcripts. Pre-specified primary signal(s): (none — exploratory).

## Q1 — is the NLA more on-task inside the flagged span?

On-task rate inside vs outside the planted span, the lift, and its case-cluster bootstrap 95% interval (2000 resamples over whole transcripts). A lift whose interval clears 0 is a real localization of the NLA's on-task-ness to the flagged span.

| model | threat | inside | outside | lift [95% CI] |
|---|---|---|---|---|
| Qwen-7B | injection (input span) | 0.34 (n=50520) | 0.15 (n=50450) | **+0.19** [+0.18, +0.20] ✓ |
| Gemma-12B | injection (input span) | 0.22 (n=50160) | 0.11 (n=49590) | **+0.11** [+0.11, +0.12] ✓ |
| Gemma-27B | injection (input span) | 0.24 (n=50160) | 0.10 (n=49590) | **+0.14** [+0.14, +0.15] ✓ |
| Llama-70B | injection (input span) | 0.23 (n=48780) | 0.09 (n=49560) | **+0.14** [+0.14, +0.15] ✓ |

## Region composition

Share of all tokens and NLA on-task rate per region (template = chat-template scaffolding outside any message content).

| model | region | share | on-task |
|---|---|---|---|
| Qwen-7B | user | 0.85 | 0.21 |
| Qwen-7B | system | 0.15 | 0.47 |
| Gemma-12B | user | 0.85 | 0.14 |
| Gemma-12B | system | 0.15 | 0.32 |
| Gemma-27B | user | 0.85 | 0.15 |
| Gemma-27B | system | 0.15 | 0.34 |
| Llama-70B | user | 0.85 | 0.14 |
| Llama-70B | system | 0.15 | 0.28 |

## Q2 — which cheap pre-pass signal predicts where the NLA is on-task?

Full table: AUROC of **every** signal vs the judge's on-task label, one row per signal, one column per model, with a case-cluster bootstrap 95% interval. The **best signal for the task is bolded** per model. `*` = blind (one pass, no counterfactual). `†` = passes Benjamini-Hochberg FDR at q=0.05 over the exploratory family; the pre-specified primaries are confirmatory and shown without penalty. The last two rows give the on-task base rate and two controls (a random-normal score, and the top signal's AUROC under label permutation), both of which sit at ~0.5 when the machinery is honest.

### injection (input span)

| signal | Qwen-7B | Gemma-12B | Gemma-27B | Llama-70B |
|---|---|---|---|---|
| `surprisal` | 0.505 [0.50,0.51]*† | 0.504 [0.50,0.51]* | 0.478 [0.47,0.48]*† | 0.509 [0.50,0.51]*† |
| `entropy` | 0.364 [0.36,0.37]*† | 0.399 [0.39,0.41]*† | 0.453 [0.45,0.46]*† | 0.399 [0.39,0.41]*† |
| `varentropy` | 0.347 [0.34,0.35]*† | 0.400 [0.39,0.41]*† | 0.441 [0.44,0.45]*† | 0.377 [0.37,0.38]*† |
| `temporal_kl` | 0.618 [0.61,0.62]*† | 0.514 [0.51,0.52]*† | 0.532 [0.53,0.54]*† | 0.516 [0.51,0.52]*† |
| `resid_jump` | 0.567 [0.56,0.57]*† | 0.525 [0.52,0.53]*† | 0.484 [0.48,0.49]*† | 0.552 [0.55,0.56]*† |
| `lookback_ratio` | 0.610 [0.60,0.62]*† | 0.611 [0.60,0.62]*† | 0.584 [0.58,0.59]*† | 0.556 [0.55,0.56]*† |
| `sink_drain` | 0.521 [0.51,0.53]*† | 0.576 [0.57,0.58]*† | 0.616 [0.61,0.62]*† | 0.605 [0.60,0.61]*† |
| `head_disagreement` | 0.488 [0.48,0.49]*† | 0.549 [0.54,0.55]*† | 0.587 [0.58,0.59]*† | 0.572 [0.57,0.58]*† |
| `w` | 0.459 [0.45,0.47]*† | 0.481 [0.47,0.49]*† | 0.512 [0.50,0.52]*† | 0.522 [0.52,0.53]*† |
| `norm_ratio` | 0.358 [0.35,0.36]*† | 0.610 [0.61,0.61]*† | 0.607 [0.60,0.61]*† | 0.285 [0.28,0.29]*† |
| `peak_ratio` | **0.221 [0.22,0.22]*†** | 0.426 [0.42,0.43]*† | **0.325 [0.32,0.33]*†** | **0.261 [0.26,0.27]*†** |
| `dominant_mass` | 0.230 [0.23,0.23]*† | 0.426 [0.42,0.43]*† | 0.407 [0.40,0.41]*† | 0.279 [0.28,0.28]*† |
| `resid_jump_nla` | 0.617 [0.61,0.62]*† | **0.693 [0.69,0.70]*†** | 0.626 [0.62,0.63]*† | 0.457 [0.45,0.46]*† |
| `attn_rollout` | 0.629 [0.62,0.63]† | 0.573 [0.57,0.58]† | 0.567 [0.56,0.57]† | 0.597 [0.59,0.60]† |
| _base rate_ | 0.25 | 0.17 | 0.17 | 0.16 |
| _control rand / perm_ | 0.50 / 0.5±0.00 | 0.50 / 0.5±0.00 | 0.50 / 0.5±0.00 | 0.50 / 0.5±0.00 |

## Headline: the number

Best **blind** pre-pass signal for predicting where the NLA is on-task, pooled over threats, with its 95% interval:

| model | best blind signal | AUROC [95% CI] |
|---|---|---|
| Qwen-7B | `peak_ratio` | 0.221 [0.22, 0.22] |
| Gemma-12B | `resid_jump_nla` | 0.693 [0.69, 0.70] |
| Gemma-27B | `peak_ratio` | 0.325 [0.32, 0.33] |
| Llama-70B | `peak_ratio` | 0.261 [0.26, 0.27] |
