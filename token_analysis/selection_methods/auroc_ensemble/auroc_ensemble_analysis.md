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
three panels for every model. They use the **fixed all-token winner**, not the
independently selected segment winners in `auroc_ensemble_selection.md`. Before
any segment is compared, the candidate identity, component weights,
model-specific pooled rank mappings, component directions, and final candidate
direction are frozen. The score is not re-ranked or re-oriented inside a
segment. This makes the panels comparable across segments and prevents an
optimistically reselected candidate for every region.

For `model_best`, the candidate is selected separately for each model--dataset
pair. For `dataset_shared`, the component names and weights are common to all
models in a dataset, but pooled rank mappings and directions are still fitted
separately by model. The segment calculations are otherwise identical.

#### Analysis population and notation

Let $c$ index cases, $t$ index finite-score tokens, and
$R\in\{\text{input},\text{boundary},\text{output}\}$ denote a segment. OPI has
no output. Liars chat-template trailers are excluded before ranks, budgets,
summaries, and plots are calculated. Input contains everything before the final
generation prompt, including earlier scaffolding and Liars prior-assistant
turns. Boundary is the final contiguous generation-prompt run, normally five
tokens after the documented Tensor Trust corrections. Output is final assistant
content.

Write $s_{ct}$ for the frozen, finally aligned ensemble score and
$y_{ct}\in\{0,1\}$ for the NLA judge label. Here 1 means that the
already-generated NLA verbalisation was judged on-task. Non-finite component
values are not imputed: an ensemble score exists only where both components are
finite. The figure therefore measures retrospective association with the judge
label. It does not observe NLA quality before verbalisation or establish that
selecting a token causes better downstream auditing.

##### Symbol key

| Symbol | Plain-language meaning |
|---|---|
| $c$ | One case: a complete transcript for one dataset example and probed model |
| $t$ | One token position in that transcript |
| $R$ | The segment being examined: input, boundary, or output |
| $T_c$ | All finite-score, non-trailer token positions available in case $c$ |
| $s_{ct}$ | The frozen ensemble score for token $t$ in case $c$; larger means that the selected rule ranks the token as more likely to yield an on-task NLA verbalisation |
| $y_{ct}$ | The observed NLA-judge label: 1 for on-task and 0 for off-task |
| $|S|$ | The number of elements in set $S$; for example, $|T_c|$ is the number of available tokens in a case |
| $\bar x$ | An average of case-level values, so each contributing case receives one vote |

Subscripts identify the unit being described. For example, $A_{cR}$ is the
AUROC for case $c$ within segment $R$, whereas $\bar A_R$ is the average of
those case-level AUROCs for segment $R$.

#### Panel 1: within-segment case-macro AUROC

For each case and segment containing at least one positive and one negative
finite-score token, the code computes the tie-aware AUROC

$$
A_{cR}
=
\Pr(s_{ct^+}>s_{ct^-}\mid t^+,t^-\in R)
+\tfrac12\Pr(s_{ct^+}=s_{ct^-}\mid t^+,t^-\in R).
$$

In this formula, $t^+$ is an on-task token and $t^-$ is an off-task
token from the same case and segment. $\Pr(\cdot)$ means the fraction of all
such positive--negative pairs for which the statement is true. The first term
counts correctly ordered pairs. The second term awards half credit when the two
scores tie. Thus, if $A_{cR}=0.80$, an on-task token outranks an off-task token
in 80% of pairwise comparisons after tie adjustment; it does not mean that 80%
of tokens are on-task.

The plotted dot is the unweighted mean across eligible cases,

$$
\bar A_R=\frac{1}{|\mathcal E_R|}\sum_{c\in\mathcal E_R}A_{cR},
$$

where $\mathcal E_R$ contains only cases with both judge classes in $R$.
`eligible n` is $|\mathcal E_R|$. It can be much smaller than the total case
count when on-task labels are rare, as in Liars, or nearly universal, as at the
Tensor Trust boundary. Every eligible transcript has equal weight regardless
of its length or class counts.

Here $\mathcal E_R$ is the set of eligible cases and
$|\mathcal E_R|$ is the printed `eligible n`. The summation adds their AUROCs
and division by the number of cases gives the macro average.

The vertical scale has the following direct interpretation:

| Panel-1 value | Meaning |
|---:|---|
| 0 | Every comparable positive--negative pair is ordered in the wrong direction |
| below 0.5 | The globally aligned score reverses direction locally in this segment |
| 0.5 | Chance pairwise ordering |
| above 0.5 | On-task tokens tend to outrank off-task tokens within the segment |
| 1 | Every comparable pair is ordered correctly |

The distance from 0.5 describes ranking strength, but there is no universal
threshold for a scientifically or operationally important effect. That depends
on the NLA budget and the cost of false positives. A value below 0.5 is possible
because direction is frozen globally, not re-fit by segment. It should not be
silently replaced by $1-A_{cR}$ in this plot because doing so would change the
fixed selection rule after seeing the segment.

The row label's `AUROC` is different: it pools all finite tokens from all
segments and cases for the full-data winner. `held-out AUROC` is the pooled
out-of-fold result of refitting mappings, directions, and candidate selection on
each training fold and applying them to unseen cases. Neither row-label value is
the dot in panel 1, and the held-out value validates overall case
generalisation rather than the exact full-data segment profile. For these
overall AUROCs, 0.5 is chance and 1 is perfect pooled ordering. A high full-data
value with a similar held-out value suggests that the complete selection
procedure is stable on new cases from the same model and dataset. A material
held-out drop would indicate full-data optimism or case shift. Similar values
do not independently validate the precise peaks and troughs in the descriptive
position or segment figures.

#### Panel 2: mean all-token relevance percentile

The frozen score is converted to a fractional midrank within each complete
case, using all finite input, boundary, and output tokens together:

$$
p_{ct}
=
\frac{\operatorname{midrank}_{u\in T_c}(s_{cu})-1}{|T_c|-1},
$$

where $T_c$ contains the case's finite non-trailer tokens. Ties receive their
average rank; the exceptional one-token case is assigned 0.5. The code first
averages percentiles over a case's tokens in segment $R$, then gives every case
with a finite score in that segment one vote:

$$
M_{cR}=\frac{1}{|T_c\cap R|}\sum_{t\in T_c\cap R}p_{ct},
\qquad
\bar M_R=\frac{1}{|\mathcal C_R|}\sum_{c\in\mathcal C_R}M_{cR}.
$$

Here `midrank` is the token's rank after tied scores receive their average
rank. Subtracting 1 and dividing by $|T_c|-1$ rescales the smallest and largest
possible ranks to 0 and 1. $T_c\cap R$ means the tokens that are both available
in case $c$ and members of segment $R$. $M_{cR}$ averages their percentiles
inside one case; $\mathcal C_R$ is the set of cases contributing to that
segment; and $\bar M_R$ averages the case means.

The plotted dot is $\bar M_R$. A value of 0.75 means that tokens in the segment
have, on average, the 75th-percentile frozen score relative to other candidate
tokens in the same transcript. It does **not** mean that 75% are on-task, that
75% will yield a useful verbalisation, or that the selector is 75% accurate.
The dashed 0.5 line is the centre of the within-case rank scale. Above it means
the score globally prioritises the segment; below it means the score
deprioritises it. After the winner and direction have been fixed, this panel
uses positions and scores, not $y$, so it measures score location rather than
label discrimination.

| Panel-2 value | Meaning |
|---:|---|
| near 0 | The segment is concentrated near the bottom of each transcript's score ranking |
| 0.25 | Its average token lies around the lower quartile |
| 0.5 | Its average token lies around the middle of the within-case score ranking |
| 0.75 | Its average token lies around the upper quartile |
| near 1 | The segment is concentrated near the top of the score ranking |

“High” and “low” are therefore relative to other tokens in the same transcript,
not absolute signal magnitudes. A high value says where the selection rule
spends attention; only panel 1 and the printed `on-task` rate say whether this
location agrees with the judge labels.

#### Panel 3: high-rank enrichment

For budget $q\in\{0.01,0.10\}$, let $K_{cq}$ contain the
$k=\max(1,\lceil q|T_c|\rceil)$ highest-scoring tokens in case $c$. Segment
representation in that top-score budget is divided by the segment's available
finite-token share:

$$
E_{cRq}
=
\frac{|K_{cq}\cap R|/|K_{cq}|}{|T_c\cap R|/|T_c|},
\qquad
\bar E_{Rq}
=
\frac{1}{|\mathcal C_R|}\sum_{c\in\mathcal C_R}E_{cRq}.
$$

Here $q$ is the budget fraction: 0.01 for the top 1% and 0.10 for the
top 10%. The ceiling in $k=\max(1,\lceil q|T_c|\rceil)$ converts that fraction
to a whole number of selected tokens and guarantees at least one. $K_{cq}$ is
that selected set. In $E_{cRq}$, the numerator is the segment's share of the
selected set and the denominator is its share of all available tokens. Their
ratio is the within-case enrichment. $\bar E_{Rq}$ then averages those ratios
over cases.

Circles show $\bar E_{R,0.01}$ and open squares show
$\bar E_{R,0.10}$. The dashed 1 line means representation proportional to
segment length. A value of 2 means that the segment supplies twice the share of
top-ranked tokens expected from its available token share. A value of 0 means
that no token from that segment enters the top set in the contributing cases.
The plotted statistic is a mean of per-case ratios, not a ratio of pooled token
counts. A deterministic stable ordering resolves ties at the selection cutoff.

| Panel-3 value | Meaning |
|---:|---|
| 0 | The segment supplies none of the selected top-score tokens |
| 0.5 | It supplies half as many as expected from its available-token share |
| 1 | It is represented exactly in proportion to its available-token share |
| 2 | It is represented at twice its available-token share |
| above 1 | Overrepresentation among the globally highest-scoring tokens |
| below 1 | Underrepresentation among the globally highest-scoring tokens |

Unlike AUROC and percentile, enrichment has no common upper bound of 1. Within a
case an upper bound is $1/a_{cR}$, where
$a_{cR}=|T_c\cap R|/|T_c|$ is the segment's availability. A segment occupying
only 2% of available tokens could therefore reach enrichment 50 if it supplied
the entire selected set. This denominator effect is why the absolute selected
share and segment length must be checked alongside very high enrichment.

This is **spatial score enrichment**, not on-task-label enrichment. It is
neither precision, recall, nor lift in NLA relevance. Short segments can have
very large enrichment because their availability denominator is small. Putting
one of five boundary tokens in the top 1% of a long transcript can therefore
produce a large value even though only one token was selected. The unplotted
`top_1_share` and `top_10_share` columns report the absolute share of selected
tokens contributed by the segment.

#### Labels beneath each segment

The labels provide denominator and composition context:

- `tok` is the number of all tokens in the segment, including any token whose
  selected ensemble score is non-finite;
- `cases` is the number of cases with at least one finite score in the segment;
- `on-task` is the token-weighted fraction $\sum y_{ct}/n_R$ over all segment
  tokens, independent of the selected score; and
- `tmpl` is the token-weighted fraction assigned to `template` by the corrected
  `analysis_region`, also independent of the score.

Thus `on-task` estimates the unconditional yield of verbalising a segment,
whereas panel 1 asks whether the score can order positives and negatives inside
it. `case_balanced_base_rate` in the summary Parquets instead averages each
case's prevalence equally; it can differ from the printed rate when transcript
lengths vary.

The annotation scales are straightforward but should not be conflated with the
three score panels:

| Annotation value | Meaning |
|---|---|
| `on-task 0` | No token verbalisation in the segment was judged on-task |
| `on-task 0.5` | Half of the segment's tokens were judged on-task |
| `on-task 1` | Every token in the segment was judged on-task |
| `tmpl 0` | No segment token is assigned to the corrected template region |
| `tmpl 1` | Every segment token is assigned to the corrected template region |
| high `tok` or `cases` | More observations contribute, not necessarily a larger or better effect |
| low `eligible n` relative to `cases` | Panel 1 is estimated from a selective subset because many cases lack one judge class |

A high `on-task` value means high unconditional verbalisation yield for that
area, even if panel 1 is near 0.5. A low `on-task` value means most tokens in the
area are off-task, although a strong panel-1 AUROC may still identify the rare
useful ones. A high `tmpl` value flags a structural-template interpretation as a
plausible alternative to semantic aggregation; it is not evidence that the
scores are invalid.

#### Intervals, dependence, and direct contrasts

Every error bar is a percentile interval from 2,000 deterministic whole-case
bootstrap replicates with seed 0. The statistic is computed within each case
first, cases are sampled with replacement, and the replicate mean is recorded.
Liars is resampled within source-subdataset strata; the other datasets have one
stratum. For panel 1, cases without both classes have undefined AUROC and are
removed before resampling. The winner, rank mapping, weights, and directions
remain fixed in every replicate.

These are pointwise descriptive 95% intervals. They account for clustering of
tokens within a transcript, but not winner-selection uncertainty,
fold-assignment uncertainty, judge error, or simultaneous inspection of many
model--segment panels. Overlap or non-overlap of two separate error bars is not
a test of a segment difference. The matched-case contrasts in
`auroc_ensemble_{best,shared}_segment_contrasts.parquet` directly bootstrap
$A_{cR_1}-A_{cR_2}$ among cases eligible in both segments. A contrast interval
containing zero means that these data do not resolve the direction; it does not
demonstrate equivalence.

The plotted quantities map to result columns as follows:

| Figure element | Parquet column | Averaging unit | Reference |
|---|---|---|---|
| panel 1 dot and interval | `case_macro_auroc`, `case_macro_auroc_ci_*` | Eligible cases | 0.5 |
| panel 2 dot and interval | `mean_relevance_percentile`, `mean_relevance_percentile_ci_*` | Cases with a finite segment score | 0.5 |
| panel 3 circle and interval | `top_1_enrichment`, `top_1_enrichment_ci_*` | Cases with a finite segment score | 1.0 |
| panel 3 square and interval | `top_10_enrichment`, `top_10_enrichment_ci_*` | Cases with a finite segment score | 1.0 |
| `on-task` | `token_base_rate` | Tokens | No universal null |
| `tmpl` | `template_fraction` | Tokens | No universal null |
| `eligible n` | `auc_case_count` | Cases | Not an effect size |

#### Visual grammar and reading order

In panels 1 and 2, the dot is the point estimate and the vertical capped line is
its 95% whole-case bootstrap interval. In panel 3, the filled circle is the top
1% budget and the open square is the top 10% budget. Dashed horizontal lines
mark the relevant no-discrimination or no-enrichment reference, not a
significance threshold.

A reliable reading order for one model row is:

1. Read the candidate name, overall `AUROC`, and `held-out AUROC` at left to
   understand the frozen rule and its same-dataset case generalisation.
2. Read `tok`, `cases`, `on-task`, and `tmpl` under each segment. For example,
   `on-task 0.80` means 80% of that segment's token verbalisations were judged
   on-task; `tmpl 1.00` means every token is template-labelled.
3. Use panel 2 to see whether the score ranks the segment high or low relative
   to the whole transcript.
4. Use panel 3 to see whether the segment actually receives more or less of a
   strict top-score budget than its length would predict.
5. Use panel 1 and `eligible n` to judge fine-grained ordering inside the
   segment. Prefer the matched-case contrast Parquet when comparing two segment
   AUROCs directly.
6. Treat the error bars as descriptive sampling uncertainty. Do not read a
   reference-line crossing as a multiplicity-adjusted hypothesis test.

#### Worked reading example

In the `model_best` OPI/Gemma-3-12B row, the boundary contains 4,000 tokens from
800 cases, has on-task rate 0.206, and is entirely template-labelled. Its mean
all-token relevance percentile is 0.757 and top-10% enrichment is 2.277, so the
frozen score strongly concentrates on this five-token region. However, only
583 cases contain both judge classes in the boundary and their case-macro AUROC
is 0.524, close to chance. The valid reading is: "the boundary is a
high-scoring, moderately higher-prevalence area, but this frozen metric barely
distinguishes which boundary tokens are on-task". It would be incorrect to read
0.757 as a relevance probability or 2.277 as a 2.277-fold increase in on-task
precision.

#### Joint interpretation

A segment can therefore have:

- high on-task rate but weak within-segment AUROC, meaning the area is a useful
  prior even though the metric cannot reliably order positions inside it;
- high mean percentile or enrichment but low on-task rate, meaning the score
  concentrates there without matching label prevalence, consistent with a
  structural or template confound;
- strong within-segment AUROC but moderate mean percentile, meaning the score
  discriminates inside the segment without allocating it much of the global
  token budget; or
- high values in all three views, the strongest descriptive evidence that a
  region is prevalent, globally prioritised, and internally rankable, but still
  not evidence of causal auditing utility.

For area selection, first inspect `on-task` for unconditional segment yield,
then panel 2 or 3 for score-budget allocation, and finally panel 1 for evidence
of fine-grained ordering within the area. The conclusions below use this joint
reading rather than treating any one panel as an exact-token objective.

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

Each interpretation below follows the same reading order: unconditional
`on-task` yield, global score concentration from percentile and enrichment,
fine-grained within-segment AUROC with its eligible-case support, and finally
held-out evidence for the overall selection procedure. The bootstrap intervals
remain descriptive and pointwise throughout.

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

The high-rank panel turns these mean locations into a budget statement.
Boundary top-1%/top-10% enrichments are 8.31/5.20 for Qwen, 2.24/2.28 for
Gemma-12B, 2.88/1.88 for Gemma-27B, and 20.98/5.91 for Llama. Input enrichment
is below 1 at both budgets for every model, so the frozen rule allocates a
disproportionate share of its strictest budget to the five-token boundary.
These ratios are spatial, not relevance lift; Llama's 20.98 is possible because
the boundary occupies very little of a transcript.

Panel-1 support also changes the strength of the claim. Boundary AUROC is
defined in 733, 583, 699, and 356 of 800 cases, respectively. The matched
boundary-minus-input AUROC contrast is unresolved for Qwen, -0.010 with interval
[-0.027, 0.006], clearly negative for both Gemmas, -0.300 and -0.263, and
positive for Llama, 0.035 [0.007, 0.063]. Thus all four rules identify the
boundary as a high-budget area, but only Llama provides evidence that it orders
tokens better there than in the input.

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

The budget panel reveals which high-percentile segments actually absorb top
scores. Qwen's output is the clearest example: mean percentile 0.939 and
top-1%/top-10% enrichment 32.33/7.84, despite output case-macro AUROC of only
0.563. Llama instead overallocates to the boundary, 6.05/3.63, while
Gemma-27B moderately overallocates to both boundary, 2.43/1.55, and output,
2.04/1.39. These are allocation patterns, not improvements in the already-high
on-task rate.

Eligibility explains the wide or unstable boundary estimates. Only 67 of
1,552 Qwen cases, 222 of 1,544 Gemma-12B cases, and 113 of 1,548 Gemma-27B
cases have both boundary judge classes; Llama has 787 of 1,550. In matched
cases, boundary-minus-input AUROC is negative for Gemma-12B, Gemma-27B, and
Llama, with intervals excluding zero; Qwen's -0.002 [-0.084, 0.089] is
unresolved. The boundary can therefore be an excellent place to spend NLA
budget because almost every explanation is on-task while offering little
within-boundary sorting headroom.

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

Mean percentile and enrichment expose different failure modes in the two
models. Gemma assigns output mean percentile 0.827 and top-1%/top-10%
enrichment 4.83/3.60, yet its output AUROC is 0.422: the score spends budget on
output while locally ordering its rare positives backwards. Llama gives input
and boundary similar mean percentiles, 0.574 and 0.572, but their AUROCs are
0.455 and 0.338. Its output is mildly discriminative at 0.529 while receiving
mean percentile 0.312 and virtually none of either top budget.

Only 272 Gemma and 390 Llama cases are boundary-AUROC eligible, compared with
2,000 cases contributing a boundary score. Matched boundary-minus-input
contrasts are negative in both models, -0.080 [-0.111, -0.049] and -0.142
[-0.166, -0.118]. Hence the boundary's slightly higher raw prevalence does not
make either model-specific ensemble a good within-boundary selector, and high
global pooled AUROC should not be projected onto every segment.

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

The strict budgets show that the four model-specific rules operationalise
this area claim differently. Qwen overrepresents boundary at the top 10%,
2.67, but not at the top 1%, 0.49; Gemma-12B and Gemma-27B put their top-1%
budget mainly in input, with enrichment 2.91 and 2.72, even though boundary
prevalence and AUROC are stronger. Llama is the opposite extreme: boundary
enrichment is 13.48 at 1% and 7.23 at 10%. A “boundary is useful” conclusion
therefore does not imply that every winner selects it at every budget.

Boundary AUROC is supported by all 96 cases for Qwen, Gemma-12B, and Llama and
94 for Gemma-27B; output AUROC uses only 31--69 cases and is correspondingly
less stable. Matched boundary-minus-input contrasts favour boundary for Qwen
and both Gemmas, with intervals excluding zero, but favour input for Llama,
-0.101 [-0.147, -0.057]. This is direct evidence of model heterogeneity, not
merely overlapping marginal error bars.

**Conclusion:** Taboo provides the clearest evidence for the late input and
later boundary positions as NLA-verbalization areas. Exact boundary ordinal
matters; the first boundary token should not be merged with the later slots.

## Plot-by-plot interpretation: `dataset_shared`

The shared plots use the same reading order, but only candidate identity and
weights are shared. Each model retains its own rank mappings and directions, so
agreement in shape is empirical rather than imposed by a common numerical
scale.

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

The budget view makes the Llama tradeoff especially clear. Its shared
boundary remains strongly overrepresented, with top-1%/top-10% enrichment
6.08/5.17 and mean percentile 0.840, but within-boundary AUROC is only 0.549
[0.526, 0.573] across 356 eligible cases. The matched boundary-minus-input
contrast is -0.082 [-0.107, -0.057]. The shared score therefore finds the same
transition region while losing much of the model-specific rule's ability to
choose among its tokens. For Qwen and the Gemmas, candidate identity is
unchanged, so their segment statistics are exactly the `model_best` values.

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

Score-budget concentration is strongest for the Gemmas: boundary top-1% and
top-10% enrichment is 31.05/7.41 for Gemma-12B and 24.46/5.73 for Gemma-27B.
Qwen and Llama are also above availability at the boundary, 3.55/2.62 and
2.61/1.64. These large ratios say that boundary tokens occupy the score budget;
they do not override the dense prevalence or eligible-case limitations.

Only 67 Qwen, 222 Gemma-12B, and 113 Gemma-27B cases support boundary AUROC,
versus 787 Llama cases. The matched boundary-minus-input contrast is clearly
positive only for Gemma-12B, 0.150 [0.096, 0.206]; it is unresolved for Qwen
and Gemma-27B and clearly negative for Llama, -0.233 [-0.257, -0.209]. The
apparently coherent high-boundary shape therefore coexists with sharply
different local ranking behavior.

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

The top-budget panel reinforces this disagreement. Input is overrepresented
for both models, at 1.53/1.39 for Gemma and 1.32/1.31 for Llama at the 1%/10%
budgets. Boundary contributes no top-1% tokens in the case-macro average and
almost no top-10% tokens, despite having the highest segment on-task rate.
Gemma's boundary AUROC of 0.573 is based on 272 cases; Llama's 0.293 is based
on 390, leaving most of the 2,000 cases ineligible because the segment contains
only one judge class.

Matched boundary-minus-input contrasts are negative for Gemma, -0.075
[-0.105, -0.046], and Llama, -0.206 [-0.233, -0.179]. The shared candidate
thus expresses a consistent input-allocation policy, but it does not turn the
weak boundary prevalence prior into a useful boundary ranking rule.

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

The enrichment panel prevents “consistency” from being overstated. Qwen
strongly allocates to boundary, 12.09/3.52 at the 1%/10% budgets. Gemma-12B
allocates its very strict budget more to input, 2.58 versus boundary 0.98, but
boundary dominates at 10%, 3.79. Gemma-27B assigns zero mean top-1% enrichment
to boundary and 2.72 to input. Llama assigns essentially no top budget to
boundary, 0.00/0.02, despite boundary AUROC 0.895 and on-task rate 0.577; its
global budget instead favours input at 1% and output at 10%.

These are not contradictions. Boundary discrimination asks whether on-task
tokens outrank off-task tokens *within boundary*, while enrichment asks whether
boundary outranks other segments. The matched boundary-minus-input AUROC
contrast is positive for all four models, from 0.114 to 0.173, with every
interval excluding zero. Boundary AUROC also uses 94--96 cases, whereas output
uses only 31--69. This supports a shared within-boundary discrimination claim,
but not a universal boundary-first allocation policy.

**Conclusion:** Taboo supplies the strongest case for focusing NLA
verbalizations on late input and boundary areas across models. Stable held-out
AUROC supports case generalisation of the overall shared selection procedure;
the exact positional peaks and segment allocations remain descriptive
full-data patterns.

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
