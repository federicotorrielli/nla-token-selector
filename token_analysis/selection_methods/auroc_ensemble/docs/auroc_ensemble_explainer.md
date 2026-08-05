# AUROC rank-ensemble selector: statistical explainer

## Question and estimand

For each model–dataset pair, the method asks which already-computed metric—or
which simple convex combination of two metrics—best ranks judged on-task tokens
above judged off-task tokens across the complete stored transcript. Every token
is included in the primary analysis. The dataset-shared variant asks the same
question while constraining all models in a dataset to use the same component
names and weights.

A case is one dataset example evaluated for one probed model. It is a complete
model-specific transcript, not one token. Tokens within a case are dependent,
which motivates grouped folds and whole-case uncertainty calculations.

## Rank normalization

For source metric m, finite values are replaced by pooled fractional midranks:

$$
u_{mt}
=
\frac{\operatorname{midrank}(x_{mt})-1}{n_m-1}.
$$

Ties receive their average rank. Non-finite values remain missing. When only
one finite value exists it is assigned 0.5, although such a constant scoring
pool is ineligible for selection.

This maps every metric to [0,1], is invariant to strictly monotone
transformations, prevents extreme activation magnitudes from dominating a
mixture, and preserves the source metric’s pooled AUROC. It is fitted
separately for every model–dataset combination, so the scale is a relative
token rank within that population, not a calibrated probability and not an
absolute activation magnitude.

Each component is oriented using its own pooled AUROC:

$$
r_{mt}
=
\begin{cases}
u_{mt}, & \operatorname{AUROC}(u_m,y)\geq 0.5,\\
1-u_{mt}, & \operatorname{AUROC}(u_m,y)<0.5.
\end{cases}
$$

Thus larger r always means stronger association with the on-task class. This
orientation is learned independently per model, including under the
dataset-shared constraint.

## Exhaustive candidate family

For every unordered pair i,j and weight

$$
\alpha\in\{0.25,0.50,0.75\},
$$

the ensemble score is

$$
e_{ij,\alpha,t}
=
\alpha r_{it}+(1-\alpha)r_{jt}.
$$

The primary blind family contains 13 originals, 78 unordered pairs, 234
weighted ensembles, and 247 total candidates. Originals remain eligible in the
same final pool, so an ensemble must outperform the best original to win.

The adjusted selection score is

$$
S(c)
=
\max\left(
\operatorname{AUROC}(c,y),
1-\operatorname{AUROC}(c,y)
\right).
$$

The complete candidate is reversed downstream if its unadjusted AUROC is below
0.5. Component directions and the final candidate direction are stored
separately. Exact ties within tolerance 1e-12 prefer lower missingness, then an
original, then lower summed computation-cost order, then the canonical
candidate ID.

For a dataset with M models, shared selection uses

$$
S_{\mathrm{shared}}(c)
=
\frac{1}{M}\sum_{j=1}^{M}S_j(c).
$$

Models receive equal weight. Tokens remain pooled within a model. This avoids
letting a model with longer transcripts determine the shared winner.

## Missingness and sensitivity analyses

An ensemble is available only where both components are finite. No value is
imputed, and no underlying token row is deleted. A candidate is ineligible
only when it has fewer than two finite observations, lacks a judgment class,
or has a constant score. Shared selection additionally requires eligibility in
every model.

The primary pairwise-complete analysis lets each candidate use all support
available to it. The common-finite-support sensitivity instead restricts every
primary component to the same tokens, then recomputes midranks, component
directions, ensembles, and winners. Agreement indicates that differing
first-token or other missingness is not driving selection.

OPI attn_rollout uses the known injected span. It and its 39 associated
ensembles are therefore excluded from the blind primary analysis and evaluated
only in the clearly labeled non-blind OPI sensitivity.

## Full-data selection and grouped held-out validation

The complete-data winner is retained for continuity with the historical
pooled-AUROC analysis and for the positional figures. It is exploratory because
the same judgments choose and score the winner.

For `model_best`, five-fold grouped validation keeps every case entirely
within one fold. Training cases determine empirical-CDF mappings, component
directions, and winners separately for each model. Those mappings are applied
to untouched held-out tokens.

The synchronized `dataset_shared` validation applies the same procedure with
two necessary changes. Folds are built from the union of dataset case IDs,
stratified by case label and Liars source subdataset, so a case cannot enter
training through one model while being tested through another. Inside each
training fold, all 247 candidates are evaluated separately per model and then
averaged with one vote per model. One shared candidate and one independently
selected shared original-only comparator are applied to every model's untouched
test cases using that model's training-fitted empirical CDF and directions.

The bridge `label` is already case-constant for Liars, Tensor Trust, and Taboo.
OPI instead stores token-level injected-span membership and has no separate
case label. The implementation therefore defines the fold-stratification label
as $\max_t y^{\mathrm{source}}_{ct}$, which is identical to the stored case
label in the first three datasets. It equals one for every current OPI case, so
OPI folds are deterministically balanced by case ID rather than by an
unavailable binary case outcome. This reduction uses source labels, not NLA
judge outcomes.

Both regimes report held-out pooled AUROC, case-macro AUROC, case-macro AP,
fixed-budget retrieval diagnostics, selection frequencies, and paired
whole-case intervals for ensemble-minus-original differences. The shared
interval resamples union case-ID clusters before averaging cases within models
and models within datasets, preserving dependence for cases present in several
model files. These values validate the complete selection procedure; they are
not used to redraw the full-data winner.

## Leave-one-model-out transfer

Shared case validation assumes that every deployed model has labelled
calibration cases. LOMO instead selects candidate identity and weights using
all other models in the dataset and evaluates the omitted model.

The `label_free` protocol learns one component orientation and one final
candidate orientation from the equal-model signed AUROCs of the training
models. Target-model empirical CDFs are fitted on target-training folds without
using their judgment labels, and the frozen directions are applied to
untouched target cases. This tests label-free transfer after unsupervised scale
normalization. Target cross-fitting folds are balanced by source subdataset and
stable case-ID hash only, so target labels are not consulted even during fold
construction.

The `target_calibrated` protocol also fixes candidate identity and weights
using only the other models, but relearns component and final directions from
target-training cases. It tests transfer of the ensemble specification when a
small labelled target calibration set is available.

LOMO reports each held-out model separately and gives only a descriptive
equal-model mean. Four models, or two for Liars, are insufficient for a
population-level model-transfer interval. In particular, each Liars holdout
learns from only one other model and is explicitly marked weak evidence.

## Segment importance

The only analyzed segments are input, boundary, and output. OPI has no output.
Liars trailer rows are excluded from all segment tables and figures.

Input includes everything before the final generation boundary, including chat
scaffolding and Liars prior-assistant turns. Boundary is the final contiguous
generation-prompt run. Output is final assistant content. Composition uses the
corrected analysis_region field, including the Tensor Trust one-character user
turn correction.

Boundary strings are mostly template tokens, but the hidden state at a boundary
position is conditioned on the preceding transcript. High boundary relevance
can therefore indicate information accumulated at the generation transition;
it is not automatically a template artifact.

Two distinct analyses must not be conflated.

### Segment-specific re-selection

The full candidate procedure is repeated after restricting to each segment:
midranks, directions, ensembles, and winners are all recomputed. This estimates
which candidate would rank tokens best if selection were confined to that
segment. Because each segment’s rank distribution is re-normalized, mean scores
from these separate fits are not comparable across segments.

### Fixed all-token winner by segment

The all-token winner, rank mapping, component directions, weights, and final
direction are frozen. Its AUROC is then evaluated inside each segment:

$$
A_R
=
\operatorname{AUROC}(s_t,y_t\mid t\in R).
$$

Pairwise contrasts are

$$
\Delta_{R_1-R_2}
=
A_{R_1}-A_{R_2}.
$$

Case-macro contrasts use only cases with both judgment classes in both
segments. A positive contrast favors the first segment; a pointwise interval
containing zero means the data do not resolve a difference. It is not evidence
of formal equivalence.

## High-rank concentration

The frozen all-token winner also defines each case’s top 1% and top 10%
positions. Segment enrichment is

$$
E_{R,k}
=
\frac{
|\operatorname{Top}_k\cap R|/|\operatorname{Top}_k|
}{
|R|/|T|
}.
$$

Values above one mean that a segment is overrepresented among globally
high-ranked positions relative to its token availability. Values below one
mean underrepresentation. Shares and enrichment are computed within each case
before cases are averaged so long transcripts cannot dominate.

Judged on-task prevalence, within-segment discrimination, and high-rank
concentration answer different questions. None proves downstream causal
usefulness. Establishing whether verbalizing a token or segment improves NLA
output requires a new intervention.
