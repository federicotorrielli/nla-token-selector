# How to read the token-position figures

This guide explains all eight plotting regimes and the AUROC-ensemble segment-comparison figures. The four per-model-best figures are in `token_analysis/selection_methods/pooled_auroc/plots/model_best/`:

- `opi_selected_metric_by_position`
- `tt_selected_metric_by_position`
- `liars_selected_metric_by_position`
- `taboo_selected_metric_by_position`

The parallel figures in `token_analysis/selection_methods/pooled_auroc/plots/dataset_shared/` use the names `{dataset}_shared_metric_by_position`. They keep one metric fixed across all model rows in each dataset: OPI `peak_ratio`, Tensor Trust `resid_jump_nla`, Liars `head_disagreement`, and Taboo `dominant_mass`. This makes cross-model shapes easier to compare while retaining each model's own raw AUROC and relevant direction.

The figures in `token_analysis/selection_methods/cv_map/plots/model_best/` use `{dataset}_cv_best_metric_by_position`. These use the cross-validated case-balanced MAP winner for each model-dataset pair. Their row labels report CV MAP first and pooled raw AUROC second. `(tied)` means the paired winner-minus-runner bootstrap interval includes zero. `‡` means the selected metric does not beat both cross-fitted position-only and region-prior baselines; the figure is still produced because the experiment must return one deterministic winner.

The figures in `token_analysis/selection_methods/cv_map/plots/dataset_shared/` use `{dataset}_cv_shared_metric_by_position`. Every row in a dataset uses the same cross-validated winner: OPI `peak_ratio`, Tensor Trust `sink_drain`, Liars `temporal_kl`, and Taboo `dominant_mass`. The row may retain a model-specific raw direction because orientation is fitted on training cases. `shared CV MAP` is the equal-model dataset selection score and is therefore repeated across rows; `model CV MAP` reports how that constrained winner performs for the particular model. Pooled raw AUROC is included only as a continuity diagnostic. `(tied)` and `‡` have the same meanings as in CV-best.

The figures in `token_analysis/selection_methods/spearman/plots/model_best/` use `{dataset}_cor_best_metric_by_position`. Metrics maximize the absolute full-data pooled Spearman correlation with binary `on_task`. Row labels report signed model rho, absolute rho, and pooled AUROC. The figures in `token_analysis/selection_methods/spearman/plots/dataset_shared/` use `{dataset}_cor_shared_metric_by_position`; labels report the equal-model mean absolute rho used for shared selection and that row model’s signed rho. In both regimes, `(tied)` means the fixed-rank case-bootstrap winner-minus-runner interval includes zero. The line and y-axis still show within-case relevance percentile, not rho.

The AUROC-ensemble positional figures are in
token_analysis/selection_methods/auroc_ensemble/plots/model_best/ and
plots/dataset_shared/. Their filenames contain auroc_ensemble_best_metric or
auroc_ensemble_shared_metric. Before selection, every source component is
converted to a pooled fractional midrank and oriented toward the on-task class:

$$
u_{mt}
=
\frac{\operatorname{midrank}(x_{mt})-1}{n_m-1},
$$

$$
e_{ij,\alpha,t}
=
\alpha r_{it}+(1-\alpha)r_{jt}.
$$

The selected ensemble value is then ranked again within each case for the
positional y-axis. The first normalization makes unlike component metrics
commensurable for mixing; the second answers where the selected candidate ranks
a token relative to other tokens in the same transcript. Neither value is a
probability.

### Paper exports

The publication-only exports are in
`token_analysis/selection_methods/auroc_ensemble/plots/paper/`. They cover all
four datasets under both `model_best` and `dataset_shared` and use embedded
STIXGeneral typography for a LaTeX-style serif appearance. They are generated directly from
the frozen position- and segment-summary Parquets; rendering does not refit a
selector or recompute a confidence interval. Regenerate all files with:

```bash
python -m token_analysis.render_auroc_ensemble_paper_plots --overwrite
```

There are 32 positional PDFs. Eight preserve the model-row layout, one per
dataset and regime; they retain segment facets, confidence ribbons, and
composition strips, but omit score metadata, the global title, explanatory top
and bottom prose, and every per-bin `n` annotation. Eight additional files use
the suffix `_metric_by_position_aggregated.pdf`. Each has one facet per segment
and overlays all model curves and their model-specific confidence ribbons using
stable colors. The aggregate view omits composition strips because token
provenance is model/tokenizer-specific and has no defined cross-model aggregate.

Eight positional files use the suffix `_metric_by_position_heatmap.pdf`. They
mirror the original model-row layout: every model and segment has its own
one-row heatmap panel. A further eight files use the suffix
`_metric_by_position_heatmap_aggregated.pdf`; these combine all models into one
model-by-position matrix per segment. Both layouts show the same case-balanced
mean percentiles on a common 0–1 `RdBu_r` color scale whose midpoint is 0.5.
Gray denotes an unavailable cell. Heatmaps display point estimates only; use
the line versions for pointwise whole-case bootstrap intervals. Each of these
16 heatmaps also has a same-named 600-dpi PNG version. The raster versions avoid
PDF-renderer seams or interpolation artifacts when figures are embedded in
LaTeX/Overleaf; the numerical values, bin boundaries, typography, dimensions,
and color normalization are identical to the corresponding PDFs.

Within each of the four positional layout families (model-row line,
model-overlay line, model-row heatmap, and model-matrix heatmap), every dataset
and both selection regimes use exactly the same physical canvas. The respective
sizes are 10.65 by 12.0, 9.6 by 3.35, 9.6 by 4.87, and 9.6 by 4.17 inches.
Position exports bypass content-dependent tight cropping, so OPI's two-segment
layout has the same outer dimensions as the three-segment datasets. The four
families retain different heights because they contain different numbers of
axes and forcing one common height would add substantial whitespace or reduce
readability.

There are 24 segment PDFs: AUROC, mean relevance percentile, and high-rank
enrichment are separate files for every dataset and regime. Each file overlays
all available models using the same stable model colors as the standard plots.
Small horizontal offsets prevent coincident estimates and confidence intervals
from hiding one another. The model legend retains the unconditional on-task
rates in input/boundary/output order (`I/B/O`; OPI uses `I/B`). Token counts,
case counts, eligible-case counts, template fractions, global titles, and
explanatory top/bottom prose are omitted. In enrichment figures, filled circles
show the top-1% budget and open squares show the top-10% budget.

### AUROC-ensemble segment-comparison figures

The AUROC-ensemble plot folders also contain files named
`{dataset}_auroc_ensemble_{best,shared}_segment_comparison`. Each model row
compares input, boundary, and, where present, output. OPI has no output; Liars
trailers are rejected from all segment ranks, budgets, summaries, and figures.

Every panel uses the **frozen all-token winner**. Candidate identity, weights,
model-specific pooled rank mappings, component directions, and final direction
are fixed before segment evaluation. They are not re-fit inside a segment.
This matters because it makes segment values comparable and allows a local
AUROC below 0.5 to expose a genuine direction reversal. Segment-specific winner
tables answer a different question: they re-run normalization and selection
inside each segment and must not be used to compare mean score heights across
segments.

The candidate printed at left is model-specific in `model_best`. In
`dataset_shared`, component names and weights are common across models, but
rank mappings and directions remain model-specific. Each row then prints:

- `AUROC`: the displayed frozen candidate's direction-aligned, full-data pooled
  token AUROC over the whole model--dataset table;
- `held-out AUROC`: pooled out-of-fold AUROC after mappings, directions, and
  candidate selection are fitted only on each training fold.

For both, 0.5 is chance and 1 is perfect ordering. Similar full-data and
held-out values support same-model, same-dataset case generalisation of the
selection procedure; they do not independently validate the exact descriptive
segment shape. These row-label values are not the dots in the first panel.

The three diagnostic columns answer different questions:

| Panel | Plotted value | Low, reference, and high values |
|---|---|---|
| Within-segment case-macro AUROC | Compute tie-aware AUROC separately in every case with both judge classes in the segment, then average eligible cases equally | 0 is perfectly reversed, below 0.5 is locally reversed, 0.5 is chance, above 0.5 means on-task tokens tend to outrank off-task tokens, and 1 is perfect ordering |
| Mean all-token relevance percentile | Rank the frozen score across all finite non-trailer tokens within each case, average percentiles within the segment, then average cases equally | near 0 means bottom-ranked, 0.5 means around the transcript's middle rank, and near 1 means top-ranked; this is score location, not relevance probability |
| High-rank enrichment | Divide a segment's share of the case's top-score set by its share of all available finite tokens, then average case ratios | 0 means absent from the top set, below 1 means underrepresented, 1 means proportional representation, and above 1 means overrepresented |

In panel 3, filled circles use a top-1% per-case budget and open squares use a
top-10% budget. Enrichment is not bounded by 1: a value of 2 means twice the
representation expected from segment length. It is **spatial score
enrichment**, not twofold NLA relevance, precision, or label lift. Short
boundaries can produce large enrichment because their availability denominator
is small.

The labels beneath a segment use different denominators from the dots:

- `tok` is its total token count, including non-finite selected scores;
- `cases` counts cases with at least one finite selected score there;
- `on-task` is the token-weighted fraction of that segment's NLA
  verbalisations judged on-task;
- `tmpl` is the token-weighted corrected-template fraction; and
- `eligible n` in panel 1 counts only cases containing both judge classes in
  that segment.

Thus high `on-task` means high unconditional yield even if AUROC is near 0.5;
high AUROC with low `on-task` means the rule may rank rare positives; and high
`tmpl` makes a structural-template explanation an important alternative. The
`on-task` and `tmpl` annotations do not change between `model_best` and
`dataset_shared` because changing the score does not change tokens, segments,
or judge labels.

Dots are full-data point estimates and capped vertical lines are descriptive
pointwise 95% intervals from 2,000 whole-case bootstrap replicates with seed 0.
Liars is resampled within source-subdataset strata. The winner and score mapping
remain fixed in every replicate, so the intervals do not include winner
selection, fold assignment, judge error, or multiple-panel uncertainty.
Separate interval overlap is not a segment-difference test; use the matched-case
`*_segment_contrasts.parquet` artifacts for that comparison.

For the full notation, equations, component-by-component explanations, value
tables, and worked OPI/Gemma example, see
`selection_methods/auroc_ensemble/auroc_ensemble_analysis.md`.

### Positional figure layout

Each row is one model–dataset combination and reports that combination's
selected all-token metric. Columns split the transcript into analysis segments.
The line is the case-balanced mean relevance rank, the ribbon is a pointwise
95% whole-case bootstrap interval, and the narrow strip below each panel shows
which kinds of tokens contribute at that position.

## What a case is

A **case** is one dataset example as evaluated for one particular probed model.
In the positional table it is identified by the combination of `dataset`,
`model`, and `case_id`. It contains that example's complete analyzed token
sequence: the prompt or prior conversation, the final generation boundary, and,
when available, the final response and any trailer tokens. Each token in the
case has its own metric value and `on_task` judgment, so one case can contain a
mixture of on-task and off-task tokens.

A case is therefore a model-specific transcript, not an individual token, a
position bin, or a relevance class. Repeated tokens and positions from the same
case are statistically dependent because they come from the same sequence.
This is why the profile first reduces tokens to one value per case/bin, gives
each contributing case equal weight, and resamples whole cases for its
uncertainty ribbon.

The same `case_id` across models means that the models were evaluated on the
same underlying dataset example; it does **not** imply identical token strings,
token counts, boundaries, responses, or aligned token indices. Tokenizers,
chat templates, and generated responses remain model-specific. Tensor Trust,
OPI, and Taboo have substantial shared case coverage across models, whereas the
Liars/Gemma and Liars/Llama cases are disjoint on-policy transcripts. Thus a
Liars model row describes its own sampled case distribution and is not a paired
case-by-case comparison with the other model.

## Sequence segments

### Input

`input` contains everything through the final non-template token before the
model's final assistant turn. It includes the system prompt, user messages,
their surrounding chat scaffolding, and—where present—earlier assistant turns.
In Liars' Bench those earlier assistant turns are marked `prior assistant` in
the composition strip.

### Boundary

`boundary` is the final contiguous run of chat-template tokens between the
input and the final assistant response. For OPI, which has no stored response,
it is the trailing generation prompt that asks the model to begin an assistant
turn.

These are structural tokens rather than ordinary prompt or response content.
They typically close the preceding turn and open the assistant turn. Examples
from the plotted data are:

| Model family | Typical boundary |
|---|---|
| Qwen | `<|im_end|>`, newline, `<|im_start|>`, `assistant`, newline |
| Gemma | `<end_of_turn>`, newline, `<start_of_turn>`, `model`, newline |
| Llama | `<|eot_id|>`, `<|start_header_id|>`, `assistant`, `<|end_header_id|>`, double newline |

The boundary x-axis uses one-indexed **token ordinal**, not percentage. A
boundary is normally only five discrete, semantically distinct template
tokens. Stretching those tokens to 0–100% would imply a continuous position
scale and make, for example, "open assistant header" look directly comparable
to an arbitrary 75% point in a long response. Ordinals preserve the actual
template-token sequence.

Boundary tokens differ across models because each model family has its own
chat template, special-token vocabulary, and tokenizer. Thus the same logical
transition into an assistant turn is encoded using different token strings and
can occasionally have a different token count. In these data, almost all
boundaries have five tokens. Liars/Gemma-27B has 856 four-token and 1,144
five-token boundaries because a leading response newline sometimes belongs to
the output. The TT source labels contained one apparent 11-token boundary for
Qwen and Llama on case `170140834572836_access_code`: the one-character user
message `a` had remained labeled `template`, merging the final user turn with
the assistant boundary. The analysis table preserves that raw provenance but
corrects the segmentation to a five-token boundary and labels `a` as user
content. Shared case IDs still do not imply shared boundary token indices.

### Output

`output` is the contiguous content of the final assistant turn. OPI has no
output column because its stored examples stop at the assistant-generation
boundary.

### Post-response terminators in Liars' Bench

Liars transcripts contain model-specific chat-template terminators after the
final assistant content, such as Gemma's `<end_of_turn>` plus newline and
Llama's `<|eot_id|>`. These raw rows remain in the canonical all-token parquet
for source fidelity and to avoid changing any all-token selection experiment.
They are excluded from positional aggregation, summary parquets, and figures:
there is no trailer panel, position axis, composition strip, or coverage count.
The plotted Liars sequence therefore ends with the final assistant response.

## What the x-axis means

For `input` and `output`, the x-axis is relative position within that segment:

$$
x_{cts} =
\frac{p_{cts}}{L_{cs}-1},
$$

where `p` is token `t`'s zero-based position in segment `s` of case `c`, and
`L` is that segment's token length. A one-token segment is assigned position
zero by the preprocessing code.

With the default number of bins set to `B = 20`, the bin index is

$$
b_{cts}
=
\min\left(B-1,\left\lfloor Bx_{cts}\right\rfloor\right).
$$

Thus every bin covers a 5% interval and a token at normalized position 1.0 is
explicitly clipped into the final bin. The plotted x-coordinate is the bin
midpoint:

$$
x_b^{\mathrm{plot}}=\frac{b+0.5}{B}.
$$

Tokens are not interpolated, and cases without a token in a bin are missing
from that bin.

For `boundary`, the x-axis is the exact one-indexed token ordinal:
1 is the first structural token, 2 is the second, and so on. This is why those
panels show integers rather than percentages. Internally, their bin is the
exact zero-based segment position:

$$
b_{cts}=p_{cts},
\qquad
x_b^{\mathrm{plot}}=b+1.
$$

This x-position normalization is performed separately within each segment.
Therefore 50% of the input and 50% of the output refer to the midpoints of
different segments, not to the same location in the full transcript. It is
also separate from the score normalization described next: score ranks use
the whole transcript, not one segment at a time.

## Score normalization: what the y-axis means

The y-axis is the direction-aligned **within-case relevance percentile** of the
selected metric. Let `r` be the selected metric's raw value for token `t` in
case `c`. First, the score is aligned so that larger always means more relevant:

$$
a_{ct}
=
\begin{cases}
r_{ct}, & \text{if higher raw values are relevant},\\
-r_{ct}, & \text{if lower raw values are relevant}.
\end{cases}
$$

This sign reversal is essential for every row whose selected direction is
`lower`. The direction printed in the row label comes from the applicable
selection regime.

Define the set of finite aligned scores in the complete transcript as

$$
F_c=\left\{t:a_{ct}\text{ is finite}\right\},
\qquad
n_c=\left|F_c\right|.
$$

The plotted token-level score is the fractional midrank

$$
u_{ct}
=
\frac{
\operatorname{midrank}_{t'\in F_c}\!\left(a_{ct}\right)-1
}{
n_c-1
}.
$$

- `1` means the token is among the highest-ranked positions in that case.
- `0.5` is the case median rank, shown by the dashed horizontal line.
- `0` means the token is among the lowest-ranked positions.

Tied raw values receive their shared midrank. Non-finite raw scores stay
missing: they are not ranked or imputed. The plotting code verifies that every
included case has at least two finite scores, so the denominator is positive.

This score is a **relative rank, not a probability**. A value of 0.8 does not
mean an 80% probability that the token is relevant. It means that the token's
metric score ranks above roughly 80% of the finite token scores in the same
case. Because every case contributes an approximately uniform set of ranks,
the figure should be read spatially: a peak means high within-case ranks are
concentrated in that sequence area.

This normalization is scientifically useful for comparing heterogeneous
metrics because it:

- maps all metrics to the common interval from zero to one;
- gives the same interpretation to higher-is-relevant and lower-is-relevant
  metrics;
- is invariant to monotone transformations of a metric;
- is robust to the extreme raw magnitudes documented by the audit;
- removes nuisance differences in scale and offset between transcripts; and
- matches the operational task of ranking token positions within a transcript.

The tradeoff is that it deliberately discards raw score spacing. It supports
comparison of **where** a selector concentrates high ranks, but not comparison
of raw metric magnitudes or calibrated relevance probabilities. Naive min–max
normalization would preserve spacing but would be unstable under the observed
activation spikes. Pooled normalization would also allow long transcripts and
between-case offsets to influence the scale. The selected within-case midrank
therefore remains the primary paper normalization.

The metric-selection statistic is not the y-axis. For example, Spearman
correlation, AUROC, or cross-validated MAP determines which metric supplies
the row, while the row itself always plots `u`. Consult those selection
statistics—not the profile's overall height—to compare selector quality.

## Case-balanced aggregation

The plot reduces millions of token rows to one profile using a two-stage
estimator. This makes the case, rather than the token, the independent unit of
aggregation.

For segment `s`, position bin `b`, and case `c`, define the finite-score token
set

$$
T_{cbs}
=
\left\{
t:
t\text{ belongs to case }c,\text{ segment }s,\text{ and bin }b,
\text{ and }u_{ct}\text{ is finite}
\right\}.
$$

The first stage averages all finite token percentiles in that case/bin:

$$
\bar{u}_{cbs}
=
\frac{1}{\left|T_{cbs}\right|}
\sum_{t\in T_{cbs}}u_{ct}.
$$

For exact boundary ordinals there is normally one token in this set. Input and output bins can contain multiple tokens, particularly in long
segments.

Let the contributing-case set be

$$
C_{bs}
=
\left\{
c:\left|T_{cbs}\right|>0
\right\}.
$$

The plotted solid line is

$$
\widehat{\mu}_{bs}
=
\frac{1}{\left|C_{bs}\right|}
\sum_{c\in C_{bs}}\bar{u}_{cbs}.
$$

This estimates the expected within-case relevance percentile for a randomly
selected contributing case at that segment position. Every contributing case
has equal weight after its within-bin tokens are averaged. In particular, a
long transcript cannot dominate merely because it places more tokens in a
bin. The plot does **not** use the token-weighted alternative

$$
\widetilde{\mu}_{bs}
=
\frac{
\sum_{c\in C_{bs}}\sum_{t\in T_{cbs}}u_{ct}
}{
\sum_{c\in C_{bs}}\left|T_{cbs}\right|
},
$$

which would weight a case in proportion to its number of tokens in that bin.
Cases without a finite score in a bin remain missing for that point; they are
not assigned zero and no interpolation is performed.

### Composition-strip aggregation

The context-composition strip uses the same two-stage, equal-case principle,
but its denominator includes every case with tokens in the bin, even if its
selected metric is missing there. For token category `k`, define

$$
q_{cbsk}
=
\frac{1}{\left|A_{cbs}\right|}
\sum_{t\in A_{cbs}}
\mathbb{1}\!\left[\operatorname{category}(t)=k\right],
$$

where `A` contains all tokens in the case/bin. The displayed fraction is

$$
\widehat{q}_{bsk}
=
\frac{1}{\left|D_{bs}\right|}
\sum_{c\in D_{bs}}q_{cbsk},
\qquad
D_{bs}
=
\left\{
c:\left|A_{cbs}\right|>0
\right\}.
$$

Thus the strip reports the expected within-bin composition of a randomly
selected bin-bearing case. It is not token-pooled, and it is not a relevance
score.

### Whole-case bootstrap intervals

The solid line is always computed from the full original dataset. The
bootstrap is used only to quantify sampling uncertainty around that full-data
estimate; it does not replace the data used for the line.

For each dataset–model pair, a replicate samples the original case IDs with
replacement, using the same sampled case multiplicities across every segment
and position bin. If `w` is case `c`'s multiplicity in replicate `r`, the
replicate estimate for a bin is

$$
\widehat{\mu}_{bs}^{*(r)}
=
\frac{
\sum_{c\in C_{bs}}w_c^{(r)}\bar{u}_{cbs}
}{
\sum_{c\in C_{bs}}w_c^{(r)}
}.
$$

The pointwise 95% ribbon is the percentile interval over the default 2,000
deterministic replicates with seed zero:

$$
\operatorname{CI}_{bs}^{95\%}
=
\left[
\operatorname{quantile}_{0.025}
\left(\widehat{\mu}_{bs}^{*}\right),
\operatorname{quantile}_{0.975}
\left(\widehat{\mu}_{bs}^{*}\right)
\right].
$$

Resampling complete cases preserves dependence among tokens and among bins
within one transcript. A token-level bootstrap would incorrectly treat
correlated positions as independent and would again favor long cases. The
ribbons are descriptive, pointwise intervals: they are neither simultaneous
confidence bands nor formal significance tests. No ribbon is drawn where
fewer than two cases contribute a finite score.

## What `n` means in positional figures

This section concerns the `n` annotation on positional line plots. It is not
the segment-comparison figure's `eligible n`, which counts only cases with both
judge classes for a within-segment AUROC.

The annotation in the upper-right of a panel is the number of distinct cases
contributing a finite selected-metric score to its plotted point:

$$
n_{bs}=\left|C_{bs}\right|.
$$

- `n=800` means every point in that panel uses 800 cases.
- `n=214–1,552/bin` means different bins use between 214 and 1,552 cases.

It is not a token count. The `/bin` suffix means the displayed range is the
minimum and maximum `n` across positions in that panel. The corresponding
case-coverage value retained in the summary parquet is

$$
\operatorname{coverage}_{bs}
=
\frac{n_{bs}}{N_s},
$$

where `N` is the number of cases containing segment `s`, whether or not the
selected score is finite in the particular bin.

`n` varies because cases have different segment lengths and no interpolation
or score imputation is performed. A short segment may have no token in some of
the 20 normalized bins; an exact boundary ordinal exists only in cases
whose segment is at least that long; and some source metrics are unavailable
at particular tokens, especially the first token.

The corrected TT boundary panels now contain exactly five ordinals with all
cases contributing: 1,552 Qwen, 1,544 Gemma-12B, 1,548 Gemma-27B, and 1,550
Llama cases. Liars/Gemma-27B still shows 1,144–2,000 cases across its boundary
because only 1,144 of the 2,000 transcripts have a fifth boundary token. A
confidence ribbon is deliberately omitted wherever fewer than two cases
contribute.

## Composition-strip labels

The narrow stacked strip below a profile describes token provenance and is
separate from the relevance y-axis. It uses `analysis_region`, so the corrected
TT token `a` appears as user content. Raw `source_region` fractions remain in
each regime's position-summary Parquet. For the current ensemble plots these are
`auroc_ensemble_best_metric_position_summary.parquet` and
`auroc_ensemble_shared_metric_position_summary.parquet` under the corresponding
results directories.

### `response`

Every final-output token is shown with the single visual label `response`. This
combines TT/Taboo `stored_prefix` and `reconstructed_tail` tokens with Liars
`dataset_original` tokens, giving the output strip the same meaning across all
datasets. The detailed provenance fractions remain separate in every regime's
position-summary Parquet, including both AUROC-ensemble position summaries, for
sensitivity analysis.

### Retained `stored_prefix` provenance

`stored_prefix` is used only for TT and Taboo output tokens and is no longer a
separate visual category. These are assistant tokens that were present in the historical response prefix retained
by the original experiment (`probe_tok_idx >= 0`). The original runs retained
at most 30 response tokens, so “stored response” does not necessarily mean the
complete original response.

### Retained `reconstructed_tail` provenance

A `reconstructed tail` is an assistant token not present in that retained
historical prefix (`probe_tok_idx == -1`). For capped TT and Taboo cases, the
repository greedily continued the stored prefix to reconstruct the remaining
analysis sequence—up to 64 response tokens for TT or 48 for Taboo.

This continuation is deterministic given the stored prefix, but it is not
always the historical suffix. In particular, Taboo originally used
temperature sampling, so its greedy reconstructed tail should not be treated
as the unavailable sampled tail that the model produced in the original run.

### `prior assistant`

`prior assistant` corresponds to source region `assistant_prior`: content from
an assistant turn that occurred **before the final assistant turn** in a
multi-turn transcript. It is assigned to the analysis `input` segment because
it is context available when the final response is produced. It is not part of
the plotted final `output`, and it is unrelated to reconstructed tails.

For completeness, Liars output tokens retain source provenance
`dataset_original`: the final assistant response stored directly in the
benchmark transcript. They are displayed as `response`, are not
stored-prefix/reconstructed-tail mixtures, and are not paired across the two
models.

## Recommended interpretation

Use the figures to ask where each selected metric concentrates its highest
within-transcript ranks:

- Is relevance elevated near the end of the input?
- Does it spike on the assistant-generation boundary?
- Is it concentrated early or late in the response?
- Does a peak coincide with template tokens or reconstructed output?

Compare shapes across model rows, but do not interpret their absolute mean
heights as model quality or calibrated relevance probabilities. In the per-model-best figures, models may use different selected metrics; the dataset-shared figures remove that particular source of variation. The CV-best figures align selection with within-transcript retrieval and show uncertainty/baseline warnings, but they may again use different metrics across rows. Tokenizers, templates, model-specific score distributions, and—particularly for Liars—different unpaired samples still differ in every regime.

## CV-shared figure interpretation

CV-shared changes only metric selection, not positional normalization or aggregation. The y-axis remains a within-case fractional rank and the x-axes retain the same normalized input/output bins and exact boundary ordinals. Therefore its spatial profiles can be compared directly with the other regimes. The shared constraint improves metric consistency across model rows, but it does not make raw metric magnitudes comparable and it can sacrifice model-specific performance; consult the model CV MAP printed in each row and the baseline marker before interpreting a shared profile as useful.

## Spearman-regime interpretation

The Spearman selector changes which metric and direction supply the ranks, but does not change positional normalization or case-balanced plotting. Selection rho is pooled across finite tokens; the plotted profile then converts the selected metric to direction-aligned ranks within each transcript. Consequently, a row label such as `rho=-0.393` describes global label association, while a y-value such as `0.8` means a token position ranks above roughly 80% of finite positions in its own case. Neither is a relevance probability. All original data determine the winner and line; bootstrap resampling supplies uncertainty only.
