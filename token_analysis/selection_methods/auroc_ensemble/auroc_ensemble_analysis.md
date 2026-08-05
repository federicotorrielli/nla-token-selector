# Analysis of the exhaustive AUROC rank ensembles

## Scope and principal conclusion

This document interprets the blind exhaustive AUROC-ensemble results for the
repository's `model_best` and `dataset_shared` regimes. These correspond to the
user-facing ideas of "best model-specific metric" and "best shared metric". The
analysis uses the implementation in [`select_metrics.py`](select_metrics.py),
all Parquet and Markdown artifacts under [`results/`](results/), every
positional and segment-comparison figure under [`plots/`](plots/), the
4,705,657-token canonical table, and the current paper draft
[`NLA_Best_Token.pdf`](../../NLA_Best_Token.pdf).

The goal is descriptive and hypothesis-generating: identify token areas and
segments whose hidden states are most likely to yield relevant Natural Language
Autoencoder (NLA) verbalizations. It is not to claim that one exact token is
universally optimal.

The central result is a qualified transition-region hypothesis:

> The final part of the input and selected positions in the five-token
> generation boundary are repeatedly associated with relevant NLA
> verbalizations. This pattern is strongest and most transferable in Taboo and
> OPI, is largely a high-prevalence structural pattern in Tensor Trust, and is
> weak or model-dependent in Liars.

The practical recommendation is therefore to use the figures as area maps:

1. when labelled calibration cases exist for the same model and task, use the
   `model_best` positional and segment profiles;
2. otherwise use `dataset_shared` only where held-out evidence supports the
   transfer claim, most clearly OPI;
3. retain the final input region, exact boundary ordinal, and response region as
   separate candidate areas rather than averaging them into a generic
   "transition token"; and
4. distinguish generalisation to new transcripts from transfer to a new model.

## What the method estimates

The blind pool contains 13 original metrics: nine forward-pass signals
(`surprisal`, `entropy`, `varentropy`, `temporal_kl`, `resid_jump`,
`lookback_ratio`, `sink_drain`, `head_disagreement`, and `w`) and four
activation-derived signals (`norm_ratio`, `peak_ratio`, `dominant_mass`, and
`resid_jump_nla`). The non-blind OPI `attn_rollout` sensitivity is excluded
because it uses the known injected span.

For each metric and model, finite scores are converted to pooled fractional
midranks,

$$
u_{mt}=\frac{\operatorname{midrank}(x_{mt})-1}{n_m-1},
$$

and oriented so that larger aligned values indicate greater NLA-judge
relevance. The selector evaluates 13 single metrics and

$$
\binom{13}{2}\times 3=234
$$

two-component convex combinations at 25/75, 50/50, and 75/25 weights. Thus 247
candidates compete in each model-dataset cell. A pair is unavailable where
either component is non-finite; values are never imputed.

Rank normalisation is suitable for combining heterogeneous signals because it
is invariant to strictly monotone transformations, preserves component AUROC,
and prevents a large numerical scale from dominating the mixture. The aligned
score is a relative ordering statistic, not a probability that an NLA
verbalization will be relevant.

## How to read every plotted quantity

### Positional figures

Files named
`<dataset>_auroc_ensemble_best_metric_by_position.{png,svg}` plot the frozen
`model_best` winner. Files named
`<dataset>_auroc_ensemble_shared_metric_by_position.{png,svg}` plot the frozen
`dataset_shared` winner.

The vertical axis is the mean direction-aligned **within-case percentile**.
A value of 0.5 is the transcript median rank, not 50% probability. Input and
output are shown in 20 normalised position bins; the five generation-boundary
tokens retain exact ordinals. Cases are averaged equally, the bands are
pointwise whole-case bootstrap intervals, and the coloured strip below each
panel gives source-region composition.

Every model row now prints exactly two overall AUROCs, with the same names in
both settings:

| Plot label | Mathematical estimand | What it supports |
|---|---|---|
| `AUROC` | For the named model--dataset pair, the full-data pooled token AUROC of the displayed selected candidate after final direction alignment, $\max(A,1-A)$ | Descriptive discrimination of the frozen candidate whose positional curve is plotted |
| `held-out AUROC` | For the same model--dataset pair, pooled AUROC of grouped out-of-fold token scores, with empirical CDFs, directions and candidate selection fitted only on each outer-training fold | Generalisation of the complete selection procedure to unseen cases from that model and dataset |

The candidate is selected separately in `model_best`. In `dataset_shared`, one
candidate identity and set of weights is selected jointly across the training
models, but the displayed `AUROC` and `held-out AUROC` are still specific to
the named model--dataset row. The equal-model shared selection score and
case-macro validation estimates remain available in the result artifacts and
held-out-validation section; they are deliberately omitted from the figures to
avoid mixing dataset-level and model-level estimands.

The held-out annotation evaluates fold-wise reselection rather than applying
the single frozen full-data fit to held-out cases. This is the statistically
valid estimate of the selection procedure's case generalisation. The frozen
full-data fit is used only for the descriptive positional curve.

### Segment-comparison figures

Files named `<dataset>_auroc_ensemble_{best,shared}_segment_comparison` contain
three panels for every model:

- **Within-segment case-macro AUROC** applies the frozen all-token winner and
  its frozen mapping to one segment, computes AUROC separately in every case
  containing both judge classes in that segment, and averages cases. Error bars
  are descriptive pointwise whole-case intervals.
- **Mean all-token relevance percentile** asks whether a segment receives high
  ranks relative to the entire transcript. It is a concentration statistic,
  not a classification rate.
- **High-rank enrichment** is an auxiliary spatial-concentration view. The
  area-level conclusions below rely primarily on on-task prevalence, mean
  percentile, and within-segment case-macro AUROC rather than treating this
  panel as an exact-token objective.

Each segment label also prints `on-task`, the fraction of its token
verbalizations judged relevant, and `tmpl`, the fraction of tokens assigned to
the corrected template region. On-task rate is a property of the labelled
segment, not of the selected metric.

A segment can therefore have:

- high on-task rate but weak within-segment AUROC, meaning that the area is
  promising even though the metric does not reliably order positions inside it;
- high mean percentile but low on-task rate, suggesting structural score
  concentration or a confound; or
- strong within-segment AUROC but only moderate mean percentile, meaning that
  the metric discriminates inside the segment without globally prioritising it.

## AUROC values shown in the `model_best` plots

All values are rounded to the same three decimals used in the positional and
segment-figure model labels. These are the only two overall AUROCs annotated
for each model row:

| Dataset | Model | Selected ensemble | AUROC | Held-out AUROC |
|---|---|---|---:|---:|
| OPI | Qwen2.5-7B | `lookback_ratio@0.50+sink_drain@0.50` | 0.786 | 0.786 |
| OPI | Gemma-3-12B | `lookback_ratio@0.50+sink_drain@0.50` | 0.810 | 0.810 |
| OPI | Gemma-3-27B | `lookback_ratio@0.50+sink_drain@0.50` | 0.810 | 0.810 |
| OPI | Llama-3.3-70B | `peak_ratio@0.50+sink_drain@0.50` | 0.729 | 0.729 |
| Tensor Trust | Qwen2.5-7B | `resid_jump@0.50+w@0.50` | 0.675 | 0.675 |
| Tensor Trust | Gemma-3-12B | `norm_ratio@0.75+resid_jump@0.25` | 0.740 | 0.740 |
| Tensor Trust | Gemma-3-27B | `peak_ratio@0.50+resid_jump@0.50` | 0.690 | 0.690 |
| Tensor Trust | Llama-3.3-70B | `head_disagreement@0.50+sink_drain@0.50` | 0.783 | 0.783 |
| Liars | Gemma-3-27B | `head_disagreement@0.50+w@0.50` | 0.814 | 0.814 |
| Liars | Llama-3.3-70B | `head_disagreement@0.50+norm_ratio@0.50` | 0.709 | 0.709 |
| Taboo | Qwen2.5-7B | `dominant_mass@0.75+w@0.25` | 0.835 | 0.831 |
| Taboo | Gemma-3-12B | `dominant_mass@0.75+resid_jump@0.25` | 0.733 | 0.730 |
| Taboo | Gemma-3-27B | `dominant_mass@0.50+norm_ratio@0.50` | 0.771 | 0.772 |
| Taboo | Llama-3.3-70B | `lookback_ratio@0.50+w@0.50` | 0.844 | 0.844 |

The first panel of each `model_best` segment figure uses these frozen
candidates:

| Dataset | Model | Input macro AUROC | Boundary macro AUROC | Output macro AUROC |
|---|---|---:|---:|---:|
| OPI | Qwen2.5-7B | 0.805 | 0.792 | N/A |
| OPI | Gemma-3-12B | 0.826 | 0.524 | N/A |
| OPI | Gemma-3-27B | 0.827 | 0.566 | N/A |
| OPI | Llama-3.3-70B | 0.747 | 0.791 | N/A |
| Tensor Trust | Qwen2.5-7B | 0.644 | 0.622 | 0.563 |
| Tensor Trust | Gemma-3-12B | 0.574 | 0.536 | 0.576 |
| Tensor Trust | Gemma-3-27B | 0.596 | 0.336 | 0.521 |
| Tensor Trust | Llama-3.3-70B | 0.696 | 0.516 | 0.667 |
| Liars | Gemma-3-27B | 0.615 | 0.422 | 0.422 |
| Liars | Llama-3.3-70B | 0.455 | 0.338 | 0.529 |
| Taboo | Qwen2.5-7B | 0.786 | 0.859 | 0.744 |
| Taboo | Gemma-3-12B | 0.628 | 0.931 | 0.681 |
| Taboo | Gemma-3-27B | 0.718 | 0.833 | 0.838 |
| Taboo | Llama-3.3-70B | 0.835 | 0.732 | 0.703 |

## AUROC values shown in the `dataset_shared` plots

`dataset_shared` is also referred to as the shared-best setting. Its labels no
longer show the equal-model selection score or held-out case-macro values. Each
row shows only the displayed shared candidate's full-data model-specific
`AUROC` and that model's grouped `held-out AUROC`:

| Dataset | Model | Shared ensemble | AUROC | Held-out AUROC |
|---|---|---|---:|---:|
| OPI | Qwen2.5-7B | `lookback_ratio@0.50+sink_drain@0.50` | 0.786 | 0.786 |
| OPI | Gemma-3-12B | `lookback_ratio@0.50+sink_drain@0.50` | 0.810 | 0.810 |
| OPI | Gemma-3-27B | `lookback_ratio@0.50+sink_drain@0.50` | 0.810 | 0.810 |
| OPI | Llama-3.3-70B | `lookback_ratio@0.50+sink_drain@0.50` | 0.610 | 0.610 |
| Tensor Trust | Qwen2.5-7B | `resid_jump_nla@0.75+w@0.25` | 0.644 | 0.644 |
| Tensor Trust | Gemma-3-12B | `resid_jump_nla@0.75+w@0.25` | 0.705 | 0.705 |
| Tensor Trust | Gemma-3-27B | `resid_jump_nla@0.75+w@0.25` | 0.679 | 0.679 |
| Tensor Trust | Llama-3.3-70B | `resid_jump_nla@0.75+w@0.25` | 0.701 | 0.701 |
| Liars | Gemma-3-27B | `dominant_mass@0.50+head_disagreement@0.50` | 0.770 | 0.766 |
| Liars | Llama-3.3-70B | `dominant_mass@0.50+head_disagreement@0.50` | 0.673 | 0.675 |
| Taboo | Qwen2.5-7B | `dominant_mass@0.50+norm_ratio@0.50` | 0.757 | 0.757 |
| Taboo | Gemma-3-12B | `dominant_mass@0.50+norm_ratio@0.50` | 0.691 | 0.691 |
| Taboo | Gemma-3-27B | `dominant_mass@0.50+norm_ratio@0.50` | 0.771 | 0.772 |
| Taboo | Llama-3.3-70B | `dominant_mass@0.50+norm_ratio@0.50` | 0.779 | 0.779 |

The first panel of each shared segment figure reports:

| Dataset | Model | Input macro AUROC | Boundary macro AUROC | Output macro AUROC |
|---|---|---:|---:|---:|
| OPI | Qwen2.5-7B | 0.805 | 0.792 | N/A |
| OPI | Gemma-3-12B | 0.826 | 0.524 | N/A |
| OPI | Gemma-3-27B | 0.827 | 0.566 | N/A |
| OPI | Llama-3.3-70B | 0.622 | 0.549 | N/A |
| Tensor Trust | Qwen2.5-7B | 0.541 | 0.575 | 0.481 |
| Tensor Trust | Gemma-3-12B | 0.548 | 0.721 | 0.569 |
| Tensor Trust | Gemma-3-27B | 0.579 | 0.617 | 0.502 |
| Tensor Trust | Llama-3.3-70B | 0.583 | 0.358 | 0.524 |
| Liars | Gemma-3-27B | 0.638 | 0.573 | 0.579 |
| Liars | Llama-3.3-70B | 0.501 | 0.293 | 0.433 |
| Taboo | Qwen2.5-7B | 0.807 | 0.941 | 0.748 |
| Taboo | Gemma-3-12B | 0.733 | 0.906 | 0.836 |
| Taboo | Gemma-3-27B | 0.718 | 0.833 | 0.838 |
| Taboo | Llama-3.3-70B | 0.766 | 0.895 | 0.632 |

## On-task rates printed in the `model_best` segment plots

Values are shown to three decimals here and to two decimals below segment names
in the figures.

| Dataset | Model | Selected ensemble | Input | Boundary | Output |
|---|---|---|---:|---:|---:|
| OPI | Qwen2.5-7B | `lookback_ratio@0.50+sink_drain@0.50` | 0.250 | 0.355 | N/A |
| OPI | Gemma-3-12B | `lookback_ratio@0.50+sink_drain@0.50` | 0.168 | 0.205 | N/A |
| OPI | Gemma-3-27B | `lookback_ratio@0.50+sink_drain@0.50` | 0.180 | 0.299 | N/A |
| OPI | Llama-3.3-70B | `peak_ratio@0.50+sink_drain@0.50` | 0.130 | 0.098 | N/A |
| Tensor Trust | Qwen2.5-7B | `resid_jump@0.50+w@0.50` | 0.833 | 0.991 | 0.878 |
| Tensor Trust | Gemma-3-12B | `norm_ratio@0.75+resid_jump@0.25` | 0.846 | 0.969 | 0.886 |
| Tensor Trust | Gemma-3-27B | `peak_ratio@0.50+resid_jump@0.50` | 0.860 | 0.985 | 0.912 |
| Tensor Trust | Llama-3.3-70B | `head_disagreement@0.50+sink_drain@0.50` | 0.671 | 0.865 | 0.872 |
| Liars | Gemma-3-27B | `head_disagreement@0.50+w@0.50` | 0.009 | 0.036 | 0.017 |
| Liars | Llama-3.3-70B | `head_disagreement@0.50+norm_ratio@0.50` | 0.022 | 0.042 | 0.032 |
| Taboo | Qwen2.5-7B | `dominant_mass@0.75+w@0.25` | 0.106 | 0.431 | 0.308 |
| Taboo | Gemma-3-12B | `dominant_mass@0.75+resid_jump@0.25` | 0.161 | 0.585 | 0.298 |
| Taboo | Gemma-3-27B | `dominant_mass@0.50+norm_ratio@0.50` | 0.256 | 0.550 | 0.288 |
| Taboo | Llama-3.3-70B | `lookback_ratio@0.50+w@0.50` | 0.112 | 0.577 | 0.166 |

## On-task rates printed in the `dataset_shared` segment plots

| Dataset | Model | Shared ensemble | Input | Boundary | Output |
|---|---|---|---:|---:|---:|
| OPI | Qwen2.5-7B | `lookback_ratio@0.50+sink_drain@0.50` | 0.250 | 0.355 | N/A |
| OPI | Gemma-3-12B | `lookback_ratio@0.50+sink_drain@0.50` | 0.168 | 0.205 | N/A |
| OPI | Gemma-3-27B | `lookback_ratio@0.50+sink_drain@0.50` | 0.180 | 0.299 | N/A |
| OPI | Llama-3.3-70B | `lookback_ratio@0.50+sink_drain@0.50` | 0.130 | 0.098 | N/A |
| Tensor Trust | Qwen2.5-7B | `resid_jump_nla@0.75+w@0.25` | 0.833 | 0.991 | 0.878 |
| Tensor Trust | Gemma-3-12B | `resid_jump_nla@0.75+w@0.25` | 0.846 | 0.969 | 0.886 |
| Tensor Trust | Gemma-3-27B | `resid_jump_nla@0.75+w@0.25` | 0.860 | 0.985 | 0.912 |
| Tensor Trust | Llama-3.3-70B | `resid_jump_nla@0.75+w@0.25` | 0.671 | 0.865 | 0.872 |
| Liars | Gemma-3-27B | `dominant_mass@0.50+head_disagreement@0.50` | 0.009 | 0.036 | 0.017 |
| Liars | Llama-3.3-70B | `dominant_mass@0.50+head_disagreement@0.50` | 0.022 | 0.042 | 0.032 |
| Taboo | Qwen2.5-7B | `dominant_mass@0.50+norm_ratio@0.50` | 0.106 | 0.431 | 0.308 |
| Taboo | Gemma-3-12B | `dominant_mass@0.50+norm_ratio@0.50` | 0.161 | 0.585 | 0.298 |
| Taboo | Gemma-3-27B | `dominant_mass@0.50+norm_ratio@0.50` | 0.256 | 0.550 | 0.288 |
| Taboo | Llama-3.3-70B | `dominant_mass@0.50+norm_ratio@0.50` | 0.112 | 0.577 | 0.166 |

The two on-task tables are numerically identical by construction. Changing the
metric changes the score assigned to tokens, but it cannot change their segment
membership or NLA-judge label. Showing both tables makes clear which candidate
is drawn in each setting without attributing the segment base rate to that
candidate.

## Plot-by-plot interpretation: `model_best`

### OPI

The files `opi_auroc_ensemble_best_metric_by_position` and
`opi_auroc_ensemble_best_segment_comparison` show the same
`lookback_ratio@0.50+sink_drain@0.50` winner for Qwen and both Gemmas, and
`peak_ratio@0.50+sink_drain@0.50` for Llama. The displayed `AUROC` / `held-out AUROC` pairs are 0.786/0.786,
0.810/0.810, 0.810/0.810, and 0.729/0.729. Their equality to three decimals
shows that fold-wise selection preserves pooled discrimination on unseen OPI
cases.

The positional plot places the boundary well above the transcript median:
mean boundary percentiles are 0.883, 0.757, 0.712, and 0.894 for Qwen,
Gemma-12B, Gemma-27B, and Llama. The exact final input token also spikes,
reaching 0.617, 0.483, 0.462, and 0.732. Boundary ordinal 3 is a shared trough
for Qwen and both Gemmas, so the boundary should not be treated as five
exchangeable tokens.

The segment plot explains what the concentration means. Boundary on-task rate
exceeds input for Qwen and both Gemmas, 0.355 versus 0.250, 0.205 versus 0.168,
and 0.299 versus 0.180. Llama reverses this prevalence ordering, 0.098 versus
0.130. Yet input case-macro AUROC exceeds boundary for both Gemmas, 0.826
versus 0.524 and 0.827 versus 0.566. Thus the boundary is a promising area for
verbalization, but the Gemma metric is better at discriminating relevant from
irrelevant positions in the input than among the five boundary tokens.

**Conclusion:** for OPI, inspect the final input token and selected boundary
ordinals. This is supported by high positional percentiles, higher boundary
prevalence in three models, and stable held-out AUROC.

### Tensor Trust

The files `tt_auroc_ensemble_best_metric_by_position` and
`tt_auroc_ensemble_best_segment_comparison` contain four different winners:
`resid_jump@0.50+w@0.50`, `norm_ratio@0.75+resid_jump@0.25`,
`peak_ratio@0.50+resid_jump@0.50`, and
`head_disagreement@0.50+sink_drain@0.50`. The displayed `AUROC` / `held-out AUROC` pairs are 0.675/0.675,
0.740/0.740, 0.690/0.690, and 0.783/0.783. Pooled discrimination therefore
persists under grouped case holdout, while the segment panel separately tests
within-case ordering inside each region.

The positional patterns are heterogeneous. Qwen assigns mean percentile 0.939
to output and 0.741 to boundary; Llama assigns 0.793 to boundary; the two
Gemmas show more moderate boundary means of 0.562 and 0.630. Within-segment
AUROC usually favours input: input/boundary values are 0.644/0.622,
0.574/0.536, 0.596/0.336, and 0.696/0.516.

The on-task labels show why Tensor Trust should be interpreted mainly as an
area-prevalence result. Boundary rates are 0.991, 0.969, 0.985, and 0.865, and
input/output rates are also high. Nearly any boundary verbalization is already
likely to be judged on-task, regardless of metric.

**Conclusion:** boundary and response are relevant Tensor Trust areas, but the
dataset supplies limited evidence that fine-grained metric ordering is needed.
Its high boundary prevalence should not be conflated with a universally strong
within-boundary selector.

### Liars

The files `liars_auroc_ensemble_best_metric_by_position` and
`liars_auroc_ensemble_best_segment_comparison` show
`head_disagreement@0.50+w@0.50` for Gemma-27B and
`head_disagreement@0.50+norm_ratio@0.50` for Llama. Their displayed `AUROC` / `held-out AUROC` pairs are 0.814/0.814 and
0.709/0.709. Thus pooled discrimination is stable under case holdout, although
the segment panel shows that this does not imply reliable ordering within every
transcript region.

The positional profiles point in opposite directions. Gemma assigns mean
percentile 0.827 to output and 0.367 to input, whereas Llama assigns 0.312 to
output and 0.574 to input. The two 2,000-case samples are disjoint and have
different Liars subdataset composition, so this is a distributional
comparison, not evidence that the plotting code reversed a direction.

Boundary on-task rates are the largest segment rates in both models, 0.036 for
Gemma and 0.042 for Llama, compared with input rates 0.009 and 0.022. However,
boundary case-macro AUROC is only 0.422 and 0.338. Exact boundary ordinals 4 and
5 have elevated on-task rates in the canonical positional data, but the
selected metrics do not consistently prioritise that same area.

**Conclusion:** Liars supports the final boundary positions as a weak
prevalence-based area prior, not a universal learned profile. Model-specific
input or output patterns should be treated as hypotheses tied to their
particular on-policy sample.

### Taboo

The files `taboo_auroc_ensemble_best_metric_by_position` and
`taboo_auroc_ensemble_best_segment_comparison` show
`dominant_mass@0.75+w@0.25` for Qwen,
`dominant_mass@0.75+resid_jump@0.25` for Gemma-12B,
`dominant_mass@0.50+norm_ratio@0.50` for Gemma-27B, and
`lookback_ratio@0.50+w@0.50` for Llama. The displayed `AUROC` /
`held-out AUROC` pairs are 0.835/0.831, 0.733/0.730, 0.771/0.772, and
0.844/0.844.

Boundary on-task rate is the largest segment rate in every model: 0.431,
0.585, 0.550, and 0.577, compared with input rates 0.106, 0.161, 0.256, and
0.112. Boundary case-macro AUROC is also strong for Qwen and both Gemmas,
0.859, 0.931, and 0.833. Llama is the instructive exception: its boundary mean
percentile is extremely high at 0.930, but input discriminates better than
boundary, 0.835 versus 0.732. Gemma-12B shows the complementary behaviour:
boundary mean percentile is only 0.479, while boundary AUROC is 0.931. These
two cells demonstrate why mean rank, on-task prevalence, and within-segment
AUROC must be read separately.

The positional figures refine the area claim. At least one of the last three
input positions rises strongly in every model. Boundary ordinal 1 has no
on-task labels in the current data, while later boundary ordinals carry most
of the relevant verbalizations.

**Conclusion:** Taboo provides the clearest evidence for the late input and
later boundary positions as NLA-verbalization areas. Exact boundary ordinal
matters; the first boundary token should not be merged with the later slots.

## Plot-by-plot interpretation: `dataset_shared`

### OPI

The files `opi_auroc_ensemble_shared_metric_by_position` and
`opi_auroc_ensemble_shared_segment_comparison` use
`lookback_ratio@0.50+sink_drain@0.50` for all four models. The displayed
`AUROC` / `held-out AUROC` pairs are 0.786/0.786, 0.810/0.810, 0.810/0.810,
and 0.610/0.610.

Qwen and both Gemmas reproduce their `model_best` curves and values. Llama is
the transfer test: its AUROC falls from 0.729 under
`peak_ratio@0.50+sink_drain@0.50` to 0.610 under the shared pair. Nevertheless, its boundary mean percentile remains
0.840. The corresponding input/boundary segment AUROCs are 0.622/0.549. A
visually high boundary can therefore coexist with weak discrimination.

**Conclusion:** the shared OPI identity is stable and held-out performance is
strong for Qwen and both Gemmas. For Llama, use the shared plots to identify a
transition area, but prefer its model-specific pair when ranking within that
area.

### Tensor Trust

The files `tt_auroc_ensemble_shared_metric_by_position` and
`tt_auroc_ensemble_shared_segment_comparison` use
`resid_jump_nla@0.75+w@0.25`. The displayed `AUROC` / `held-out AUROC`
pairs are 0.644/0.644, 0.705/0.705, 0.679/0.679, and 0.701/0.701.

The shared positional profiles are more coherent than the four
`model_best` profiles. Boundary mean percentiles are 0.552, 0.862, 0.810, and
0.591 for Qwen, Gemma-12B, Gemma-27B, and Llama, with a recurring ordinal-3
trough. Within-segment results remain model-dependent: boundary exceeds input
for both Gemmas, 0.721 versus 0.548 and 0.617 versus 0.579, but is much worse
for Llama, 0.358 versus 0.583. Boundary on-task rate remains very high in every
model.

**Conclusion:** the shared metric reveals a common structural transition shape,
especially in the Gemmas, but the mixed within-segment case-macro AUROCs do
not support a universal within-boundary ordering. Boundary remains a sensible area primarily
because of prevalence.

### Liars

The files `liars_auroc_ensemble_shared_metric_by_position` and
`liars_auroc_ensemble_shared_segment_comparison` use
`dominant_mass@0.50+head_disagreement@0.50`. The displayed `AUROC` /
`held-out AUROC` pairs are 0.770/0.766 for Gemma and 0.673/0.675 for Llama.

Unlike their segment base rates, the shared score does not concentrate on the
boundary. Mean input/boundary/output percentiles are 0.547/0.148/0.441 for
Gemma and 0.585/0.263/0.167 for Llama. Within-segment AUROCs also favour input:
0.638/0.573/0.579 and 0.501/0.293/0.433. Yet boundary on-task rates, 0.036 and
0.042, exceed the corresponding input rates, 0.009 and 0.022.

**Conclusion:** the shared Liars metric and the raw location prior disagree.
For suggesting areas to verbalize, retain the final boundary as a weak
prevalence-based hypothesis and treat the input-focused shared score as a
separate model-derived hypothesis. The data do not justify merging them into a
single universal Liars profile.

### Taboo

The files `taboo_auroc_ensemble_shared_metric_by_position` and
`taboo_auroc_ensemble_shared_segment_comparison` use
`dominant_mass@0.50+norm_ratio@0.50`. The displayed `AUROC` /
`held-out AUROC` pairs are 0.757/0.757, 0.691/0.691, 0.771/0.772, and
0.779/0.779.

The segment plot is the most consistent shared result in the repository.
Boundary case-macro AUROC is 0.941, 0.906, 0.833, and 0.895, exceeding input in
all four models. Boundary on-task rates are also consistently larger than input
and output. Mean boundary percentiles are 0.688, 0.682, 0.525, and 0.519.
Gemma-27B instead gives input a slightly higher mean percentile, 0.606, and
Llama gives output the highest mean, 0.599. Thus the boundary is consistently
discriminative even when another segment receives higher global ranks.

The position plots show late-input emphasis and generally rising response
profiles, while exact boundary shapes remain model-family-specific. The shared
claim is therefore about the boundary **area**, not one universal boundary
ordinal.

**Conclusion:** Taboo supplies the strongest case for focusing NLA
verbalizations on late input and boundary areas across models. The high
held-out AUROC stability shows that the pooled signal is not merely a
full-data visual pattern.

## Held-out model transfer

Grouped case validation asks whether selection generalises to new transcripts
for the observed models. Leave-one-model-out (LOMO) asks whether candidate
identity, weights, and directions transfer to a model excluded from selection.

| Dataset | Label-free case-macro AUROC / AP | Target-calibrated case-macro AUROC / AP | Interpretation |
|---|---:|---:|---|
| OPI | 0.802 / 0.436 | 0.761 / 0.406 | Strongest transfer evidence; every holdout selects `lookback_ratio@0.50+sink_drain@0.50` |
| Tensor Trust | 0.599 / 0.891 | 0.587 / 0.882 | Modest AUROC; high AP reflects the dense positive class |
| Liars | 0.470 / 0.064 | 0.479 / 0.051 | Weak evidence because each holdout trains on one other model |
| Taboo | 0.647 / 0.412 | 0.704 / 0.474 | Useful case generalisation but inconsistent model transfer |

OPI is the only clean label-free model-transfer result. Its target-model macro
AUROCs span 0.778--0.820. Label-free Llama reaches 0.778 while
target-calibrated Llama reaches 0.615 because the training-model consensus
orients `lookback_ratio` higher, whereas all five Llama calibration folds
orient its marginal effect lower. Marginal component alignment need not
optimise the joint ensemble; here the common direction acts as regularisation.

Taboo illustrates the difference between case and model generalisation. Its
shared boundary area is strong on the observed models, but the selected pair
and its gain over a single metric do not transfer uniformly when a model is
excluded. Tensor Trust also selects heterogeneous LOMO identities. Liars has
only two models, so each LOMO fold learns from one model and cannot support a
population-level model-transfer claim.

## Cross-dataset conclusions about relevant areas

The plots support four area-level conclusions.

1. **The generation transition is repeatedly informative.** OPI, Tensor Trust,
   and Taboo show elevated boundary prevalence, boundary rank, boundary
   discrimination, or a combination of the three.
2. **The last input region is at least as important as the visible boundary.**
   OPI has a sharp exact final-input spike, and Taboo repeatedly emphasises the
   last few input positions.
3. **Boundary tokens are heterogeneous.** OPI often has an ordinal-3 trough,
   shared Tensor Trust repeats that trough, Taboo favours selected later slots,
   and Liars places more judge-positive explanations near its last boundary
   positions. Exact ordinal must be retained.
4. **Response relevance is task- and model-specific.** It is prominent for
   Tensor Trust/Qwen, Liars/Gemma, and parts of Taboo, but weak for
   Liars/Llama and absent by construction in OPI.

There is little evidence for a universal first-output-token rule. The robust
ordering of areas is: immediately before the transition, selected positions in
the generation boundary, and then task-specific response regions.

## Scientific interpretation of boundary effects

Two non-exclusive explanations remain plausible.

The semantic-aggregation interpretation follows autoregressive causality: a
boundary hidden state is conditioned on the complete preceding transcript and
can therefore integrate an injected instruction, latent secret, or deceptive
setup. The original NLA study often probes an Assistant control token and
reports reward-model features there
([Fraser-Taliente et al., 2026](https://transformer-circuits.pub/2026/nla/)).
It is also directionally consistent with causal-tracing evidence that a
semantic unit's final token can become an information-aggregation site
([Meng et al., 2022](https://proceedings.neurips.cc/paper_files/paper/2022/hash/6f1d43d5a82a37e89b0665b33bf3a182-Abstract-Conference.html)).
Neither result proves that every chat delimiter is a semantic summary token.

The structural-artifact interpretation is equally credible. Massive residual
activations occur at starting and delimiter tokens and can act as
input-agnostic bias terms
([Sun et al., 2024](https://arxiv.org/abs/2402.17762)). Attention sinks can
concentrate on initial tokens even when those tokens are not semantically
important ([Xiao et al., 2023](https://arxiv.org/abs/2309.17453)). This is
especially relevant to `norm_ratio`, `peak_ratio`, `dominant_mass`,
`sink_drain`, and `w`. The paper should not cite Sun et al. as direct evidence
that delimiter states are compressed semantic summaries; their central result
is bias-like massive activation.

The attention prior is also useful but limited. Lookback Lens uses
context-versus-generation attention ratios to detect contextual hallucination
([Chuang et al., 2024](https://aclanthology.org/2024.emnlp-main.84/)). This
repository's aggregate `lookback_ratio` is inspired by that mechanism, but it
is not the trained per-head Lookback Lens classifier.

The present association cannot distinguish semantic integration from a
structural NLA or judge response. A targeted causal follow-up could preserve
the boundary token sequence while swapping or ablating preceding task content,
then test whether the boundary verbalization changes.

## Recommended NLA-verbalization areas

| Dataset | Suggested areas | Evidence and qualification |
|---|---|---|
| OPI | Exact final input token and selected generation-boundary ordinals | Strong `model_best` and held-out evidence; shared transfer is strong for Qwen/Gemmas but weaker for Llama |
| Tensor Trust | Generation boundary and response region | Extremely high on-task prevalence makes these safe areas, but fine metric ordering adds limited evidence |
| Liars | Final boundary positions as a weak positional prior; model-specific input or output regions as separate hypotheses | Pooled-to-macro gaps and disjoint samples prevent a universal profile |
| Taboo | Late input, later boundary positions, and task-dependent late response | Strongest cross-model boundary prevalence and within-boundary discrimination |

These are areas to prioritise for subsequent NLA verbalization and qualitative
inspection. They are not claims that every token in the area is relevant or
that one exact position is optimal.

## Limitations

1. **Validation scope is conditional.** Grouped validation supports new cases
   for the observed models. Its intervals condition on one fold assignment and
   do not rerun selection inside every bootstrap replicate. LOMO has four
   targets, or two for Liars, so no model-population interval is justified.
   There is no held-out-dataset or held-out-task validation.
2. **A direct position/region baseline is still useful.** A cross-fitted model
   using normalised position, exact boundary ordinal, segment, length, and
   model family would determine how much signal the ensemble adds beyond the
   area patterns visible in the figures.
3. **Template and activation confounds remain.** Boundary peaks may reflect
   semantic integration, input-agnostic massive activations, NLA training
   priors, or judge sensitivity. Content-preserving/template-changing and
   template-preserving/content-changing counterfactuals would separate these
   mechanisms.
4. **Judge validity is external to the confidence intervals.** Whole-case
   resampling measures sampling variation, not construct validity. Human or
   second-model annotation and prompt-paraphrase checks remain appropriate.
5. **Dataset provenance limits cross-panel comparison.** Liars model samples
   are disjoint; Tensor Trust is dense; OPI has no output; and part of
   Tensor Trust and Taboo output is a deterministic reconstructed tail. Shared
   case IDs also do not imply shared token positions across tokenizers.

The descriptive use of the same labels to select a metric and draw its
full-data positional profile is intentional for the stated hypothesis-generation
goal. The grouped held-out results are supplied to quantify stability and
generalisation, not to turn the positional figures into a causal test.

## Scope boundary and future work

Measuring direct downstream utility is outside the scope of this stage. The
current contribution is a principled method for measuring association and
suggesting token areas on which future NLA verbalization work can focus.

A subsequent study could test whether verbalizing the proposed areas improves
an auditor's diagnosis or causal understanding relative to random,
position-only, or alternative area policies. Counterfactual content swaps and
boundary-preserving ablations would be especially informative because they
could distinguish semantic aggregation from structural template effects. Such
experiments would extend the present area-level evidence rather than being a
prerequisite for its descriptive conclusions.
