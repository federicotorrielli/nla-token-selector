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

Each ensemble method folder also contains segment-comparison figures. Their
three diagnostic columns show the frozen all-token winner’s within-segment
case-macro AUROC, case-balanced mean all-token relevance percentile, and top-1%
and top-10% enrichment. Segment-specific winner tables are separate: they
recompute ranks and selection inside input, boundary, or output and must not be
used to compare mean rank heights across independently normalized segments.
Liars trailers are rejected from these summaries and figures.

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

## What `n` means

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
TT token `a` appears as user content; raw `source_region` fractions remain in
`selection_methods/pooled_auroc/results/model_best/position_plot_summary.parquet` for provenance checks.

### `response`

Every final-output token is shown with the single visual label `response`. This
combines TT/Taboo `stored_prefix` and `reconstructed_tail` tokens with Liars
`dataset_original` tokens, giving the output strip the same meaning across all
datasets. The detailed provenance fractions remain separate in
`selection_methods/pooled_auroc/results/model_best/position_plot_summary.parquet` for sensitivity analysis.

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
