# Analysis of pooled-AUROC `model_best` token selection

## Scope and bottom line

This note interprets the per-dataset, per-model pooled-AUROC winners in the
[`model_best` audit](results/model_best/bridge_input_audit.md), the
[`position_plot_summary.parquet`](results/model_best/position_plot_summary.parquet), and the four
model-best figures ([OPI](plots/model_best/opi_selected_metric_by_position.svg),
[Tensor Trust](plots/model_best/tt_selected_metric_by_position.svg),
[Liars' Bench](plots/model_best/liars_selected_metric_by_position.svg), and
[Taboo](plots/model_best/taboo_selected_metric_by_position.svg)). It also checks the plotted
summaries against the canonical 4,705,657-token table and the source result parquets. The intended
use is to decide where to spend a limited NLA-verbalization budget, not to claim that a high-scoring
token is causally important to model behavior.

The main conclusion is conditional:

- **OPI has useful token-level selection signal**, especially Qwen's direction-reversed
  `peak_ratio`. Gemma and Llama require explicit removal of chat-template tokens; for Gemma-27B
  and Llama-70B the content-only winner is not the all-token winner.
- **Tensor Trust is too dense to be a compelling selection benchmark.** Between 68.1% and 86.4%
  of all tokens are already judged on-task. Some metrics discriminate pooled tokens, but the gain
  from selecting a small subset is inconsistent and much of the top tail is template structure.
- **The pooled Liars selector should not be used to choose positions within a transcript.** Its
  apparently good direction-adjusted pooled AUROC is a between-case/length/composition effect: the
  case-macro AUROC is at or below chance and top-1% precision is exactly zero for both models.
- **Taboo has genuine within-case ranking signal at moderate budgets**, but its global top 1% is a
  deterministic special token in every case. The useful locations are the final assistant boundary
  and response content after template filtering. Late-response conclusions are provisional because
  many late tokens are deterministic reconstructions rather than the historically sampled tail.

Thus the evidence supports **dataset- and model-conditional selectors with structural-token
controls**, not a universal “best token.” The safest paper-level claim is that pooled AUROC reveals
retrospective association, while deployable token selection additionally requires held-out,
case-balanced, fixed-budget evaluation.

## What the figures estimate

For candidate metric `m`, the published selection is

$$
m^* = \arg\max_m \left|\widehat{\operatorname{AUROC}}_m-\tfrac12\right|.
$$

If the raw AUROC is below 0.5, the score is sign-reversed. This is necessary: for OPI/Qwen and both
Liars models, *lower* raw scores are associated with on-task verbalizations. The direction-adjusted
AUROC is therefore $\max(A,1-A)$, e.g. OPI/Qwen is 0.761 rather than 0.239.

The positional plot then replaces raw scale with a direction-aligned within-case fractional rank,

$$
u_{ct}=\frac{r_{ct}-1}{n_c-1}\in[0,1],
$$

where ties receive midranks and non-finite values are omitted. Tokens are averaged within each
case/bin before cases are averaged. Consequently:

1. `u = 1` means “highest selected-metric rank in this transcript,” not a relevance probability.
2. The mean rank across all finite tokens in a case is approximately 0.5 by construction. A peak in
   one segment necessarily coexists with a trough elsewhere.
3. Pooled AUROC can be high because scores separate *different transcripts* even when they do not
   order useful tokens correctly *within a transcript*.
4. The bootstrap bands quantify case-sampling uncertainty for each bin separately. They are
   descriptive, pointwise intervals, not simultaneous tests over 20 bins, five boundary ordinals,
   multiple segments, and 14 model-dataset pairs.

These distinctions are essential for turning the figures into an NLA allocation policy.

## Overall quantitative assessment

The following table combines the published winner, direction-adjusted pooled AUROC, base rate, and
the audit's per-case top-1% result. “Template” is the fraction of selected tokens that are chat
scaffolding. The last column is the action justified by the evidence.

| Dataset | Model | all-token winner | adjusted pooled AUROC | base rate | top-1% precision | top-1% template | assessment |
|---|---|---:|---:|---:|---:|---:|---|
| OPI | Qwen-7B | lower `peak_ratio` | 0.761 | 0.254 | 0.534 | 0.000 | usable on user content |
| OPI | Gemma-12B | higher `resid_jump_nla` | 0.685 | 0.170 | 0.006 | 1.000 | filter templates before ranking |
| OPI | Gemma-27B | higher `resid_jump_nla` | 0.635 | 0.184 | 0.258 | 0.995 | use content-only re-selection |
| OPI | Llama-70B | higher `sink_drain` | 0.673 | 0.129 | 0.249 | 0.189 | position-confounded; use content-only re-selection |
| Tensor Trust | Qwen-7B | higher `resid_jump` | 0.639 | 0.837 | 0.921 | 0.006 | limited gain; output onset is useful |
| Tensor Trust | Gemma-12B | higher `norm_ratio` | 0.719 | 0.849 | 0.653 | 0.426 | top tail is worse than no selection |
| Tensor Trust | Gemma-27B | higher `resid_jump_nla` | 0.672 | 0.864 | 0.963 | 0.572 | high precision, but dense and template-heavy |
| Tensor Trust | Llama-70B | higher `resid_jump_nla` | 0.695 | 0.681 | 0.486 | 0.692 | top tail is worse than no selection |
| Liars | Gemma-27B | lower `head_disagreement` | 0.720 | 0.011 | 0.000 | 0.321 | reject for within-case selection |
| Liars | Llama-70B | lower `head_disagreement` | 0.682 | 0.023 | 0.000 | 0.876 | reject for within-case selection |
| Taboo | Qwen-7B | higher `dominant_mass` | 0.796 | 0.212 | 0.000 | 1.000 | filter templates; use a wider budget |
| Taboo | Gemma-12B | higher `dominant_mass` | 0.681 | 0.276 | 0.000 | 1.000 | filter templates; moderate evidence |
| Taboo | Gemma-27B | higher `dominant_mass` | 0.651 | 0.300 | 0.000 | 1.000 | filter templates; weak top-budget gain |
| Taboo | Llama-70B | higher `resid_jump_nla` | 0.723 | 0.162 | 0.000 | 1.000 | use content-only `norm_ratio` |

AUROC is an average over all positive-negative pairs and does not optimize the extreme upper tail.
The OPI/Gemma-12B and all Taboo rows demonstrate this sharply: respectable AUROC coexists with a
useless top 1%. A metric can rank most pairs correctly while assigning one repeated structural token
the maximum score in every transcript.

### Pooled versus within-case discrimination

As a post-hoc diagnostic, I recomputed AUROC separately in every transcript containing at least one
positive and one negative judge label, then averaged cases equally:

$$
\operatorname{AUC}_{\mathrm{case}}=
\frac{1}{|C'|}\sum_{c\in C'}\operatorname{AUC}_c.
$$

Liars trailer tokens were excluded to match the candidate positions in the figures. This statistic is
descriptive—it uses the same data that selected the winner—but it directly tests whether the pooled
winner can rank tokens within the unit on which the NLA budget is spent.

| Dataset | Model | adjusted pooled AUROC | case-macro AUROC | valid cases |
|---|---|---:|---:|---:|
| OPI | Qwen / Gemma-12 / Gemma-27 / Llama | 0.761 / 0.685 / 0.635 / 0.673 | 0.765 / 0.683 / 0.625 / 0.675 | 800 each |
| Tensor Trust | Qwen / Gemma-12 / Gemma-27 / Llama | 0.639 / 0.719 / 0.672 / 0.695 | 0.539 / 0.523 / 0.563 / 0.551 | 1552 / 1544 / 1528 / 1550 |
| Liars | Gemma-27 / Llama | 0.720 / 0.682 | **0.490 / 0.439** | 1533 / 1183 |
| Taboo | Qwen / Gemma-12 / Gemma-27 / Llama | 0.796 / 0.681 / 0.651 / 0.723 | 0.820 / 0.698 / 0.617 / 0.686 | 96 each |

OPI and Taboo largely preserve the pooled ordering within cases. Tensor Trust falls close to chance
once between-case differences are removed. Liars reverses the intended result. This explains why
the Liars positional curve and top-budget precision conflict with its headline pooled AUROC.

## Dataset-level interpretation and recommended tokens

### OpenPromptInjection

**What is consistent.** The planted span occupies the later part of the user message. Across models,
the judge-label lift inside versus outside the planted span is +0.10 to +0.17, so the NLA does
localize the injection above its background rate. The useful content prior is approximately the
middle-to-late user region: Qwen's selected rank peaks at 60–65% of the input (0.674), while Llama
peaks at 70–75% (0.763). The injected-span fraction rises strongly through these bins. Concrete
Qwen top-tail tokens include `Ġif`, `Ġwith`, `ĠWrite`, `ĠAnswer`, and `Ignore`; 81.5% of its selected
top-1% user tokens fall inside the planted span.

**Qwen-7B is the cleanest result.** Lower `peak_ratio` gives adjusted pooled AUROC 0.761,
case-macro AUROC 0.765, and top-1% precision 0.534 versus a 0.254 base rate, with no templates in
the top 1%. For this model, ranking user-content tokens by *ascending* `peak_ratio` is justified.
Position should be a weak prior, not a replacement for the metric; the strongest region is roughly
60–75% of the serialized input.

**Gemma-12B and Gemma-27B expose a turn-transition artifact.** `resid_jump_nla` is the residual
distance between adjacent NLA-layer states after removing spike channels. Large changes are expected
at role and turn transitions. For Gemma-12B, the maximum-ranked top-1% token is the template token
`user` in all 800 cases, followed by `<start_of_turn>` in 627 cases. Precision is only 0.006 and the
selection is 100% template. Gemma-27B behaves similarly: 99.5% of its top 1% is template. The
boundary sawtooth—very high ranks at ordinals 1, 2, and 4 but a low rank at ordinal 3—is therefore
primarily a property of the fixed chat transition, not evidence that those particular vocabulary
items are universally semantically important.

The audit's content-only provenance changes Gemma-27B from `resid_jump_nla` to `peak_ratio`.
Recomputing this candidate on non-template source rows gives raw AUROC 0.325, i.e. adjusted AUROC
0.675 with **lower** values relevant. Gemma-12B retains `resid_jump_nla` after content filtering.
The practical policies are therefore:

- Gemma-12B: exclude templates, then rank user tokens by higher `resid_jump_nla`.
- Gemma-27B: exclude templates and reselect by lower `peak_ratio`; do not deploy the all-token
  `resid_jump_nla` top tail.

**Llama-70B mainly learns position.** Its `sink_drain` rank rises from 0.023 in the first input bin
to 0.763 at 70–75%; raw score correlates 0.689 with normalized position. `sink_drain` is negative
attention mass on the early sink, so this trend is mathematically expected as later queries move
attention away from the initial parking positions. The late placement of the injection makes this
useful, but not threat-specific. The all-token and content-only winners differ; non-template
`peak_ratio` has raw AUROC 0.261 (adjusted 0.739, lower is relevant). Prefer that content-only score
to a positional `sink_drain` policy.

**Unexpected early positives.** For Qwen and both Gemma models, the largest input *judge-label*
rate occurs around 10–15%, where the injected-span label is zero. Because a causal activation cannot
encode a later injected span, this cannot be evidence of early causal processing of that span. The
more plausible explanation is construct contamination: the system/original-task text itself contains
instruction language, and a judge that sees only the NLA explanation cannot always distinguish a
legitimate instruction from an instruction embedded in user data. The paper should report this as a
false-positive mode, not interpret it as long-range anticipation.

### Tensor Trust

Tensor Trust is a dense “following input steering” task: pooled on-task rates are 0.837, 0.849,
0.864, and 0.681. On-task rates are actually 0.07–0.16 lower inside the nominal attack/access-code
span than outside it. This is consistent with the judge question covering broad instruction-following
state rather than sharply localizing the human-labelled attack. In this regime, a selector has little
room to save NLA calls because almost any token is judged relevant.

The strongest actionable exception is **Qwen's output onset**. Its first 5% response bin has mean
rank 0.842, judge on-task rate 0.986, and full case coverage; higher `resid_jump` also gives top-1%
precision 0.921 with only 0.6% templates. If one must verbalize a small fixed set, the first response
token(s) are defensible for Qwen.

The other models do not show a stable response rule. Gemma-27B assigns every output bin a rank below
0.5 even though output on-task rates are about 0.96. Llama's output curve remains near 0.5, and
Gemma-12B is only modestly above it in some middle bins. Boundary on-task rates are 0.89–1.00, but
metric ranks form tokenizer-specific sawtooth patterns and sometimes rank an almost-always-on-task
boundary token very low. For example, Qwen boundary ordinal 3 has rank 0.251 but on-task rate 0.982;
Llama ordinal 5 has rank 0.296 but on-task rate 0.975.

The case-macro AUROCs of 0.523–0.563 and template-heavy top tails for Gemma/Llama show that much of
the pooled signal is between transcripts. The scientifically honest recommendation is therefore to
treat Tensor Trust as evidence about **scope**—selection is unnecessary in dense tasks—rather than
as evidence for a universal token policy. If a budget is nevertheless imposed, remove templates and
sample response content, using Qwen's output onset as the only strong position-specific exception.

### Liars' Bench

The apparent cross-model agreement on lower `head_disagreement` is misleading. This metric is the
sum across layers of

$$
H\!\left(\frac1H\sum_h a_h\right)-\frac1H\sum_h H(a_h),
$$

a Jensen-Shannon-type measure that is small when attention heads look alike. It has adjusted pooled
AUROC 0.720/0.682, but the within-case results are 0.490/0.439. Case-mean aligned score also correlates
-0.980/-0.959 with transcript length. The two models use disjoint on-policy transcript sets and
different subdataset mixtures, so their similar pooled winner is not paired replication.

The failure is visible in the actual selected tokens:

- Gemma-27B ranks the first 5% of input at 0.980 and selects `<bos>` in all 2,000 cases; its top-1%
  tokens are template/system tokens and precision is zero.
- Llama-70B ranks the first 5% at 0.973 and selects `<|begin_of_text|>` in all 2,000 cases; top-1%
  precision is again zero.
- Every output bin lies below 0.5 for both models. Yet the final 5% of the output has the highest
  response-local on-task rate (0.135 for Gemma-27B and 0.081 for Llama), exactly where the metric
  rank is low (0.254 and 0.110).
- Useful boundary ordinals are inverted too: Gemma boundary ordinal 4 is on-task in 11.3% of cases
  but has rank 0.383; Llama ordinal 5 is on-task in 14.5% but has rank 0.169.

This is a form of aggregation failure: pooled token AUROC rewards score differences caused by case
length, scenario, chat format, and subdataset composition, while the downstream decision is a rank
within one transcript. The correct conclusion is **no supported Liars token selector under this
regime**. If an NLA budget must be allocated before a better selector is validated, use a transparent
position baseline such as the last few response tokens; do not use lower `head_disagreement`.

### Taboo organisms

Taboo provides the most coherent cross-model *content* pattern. For Qwen and both Gemmas,
`dominant_mass`—the fraction of squared activation norm in dataset/model-specific massive-activation
channels—is the winner. Llama instead selects `resid_jump_nla`. Case-macro AUROC remains 0.617–0.820,
so unlike Liars the pooled association mostly survives within transcripts.

However, the global maximum is structurally wrong in every case:

- Qwen selects a template newline `Ċ` in all 96 cases.
- Gemma-12B and Gemma-27B select `<bos>` in all 96 cases.
- Llama selects `<|start_header_id|>` in all 96 cases.

All have zero top-1% precision. This is predicted by the definition of `dominant_mass` and by prior
work showing massive activations at start and delimiter tokens. It is also why a good average AUROC
does not imply a good one-token policy. At 10%, Qwen improves to precision 0.513 versus a 0.212 base
rate; the Gemma-12B, Gemma-27B, and Llama gains are small or negative before template filtering.

After structural-token removal, the best candidate locations are:

- **Final assistant-generation boundary.** Across tokenizers, the start-of-assistant transition often
  yields high judge relevance because its residual state has integrated the secret-bearing prompt.
  The exact useful ordinal is tokenizer-specific: Qwen ordinal 3, Gemma-12B ordinals 2/5, and Llama
  ordinals 2/5 have both high rank and high on-task rate. The shared semantic event is more credible
  than a shared integer ordinal.
- **Middle-to-late response content.** The metric-rank peaks are 60–65% (Qwen), 65–70%
  (Gemma-12B), 90–95% (Gemma-27B), and 95–100% (Llama). The exact maximum is not stable enough to
  justify one universal fraction, but the latter half of the response is a defensible prior.
- **Llama content should use `norm_ratio`.** The all-token winner changes under content-only
  re-selection; non-template `norm_ratio` has AUROC 0.735 with higher values relevant, slightly
  stronger than the selected `resid_jump_nla` restricted to content (0.711).

The late-response result needs an explicit provenance caveat. Historically stored TT/Taboo outputs
were capped at 30 tokens; missing suffixes were deterministically continued. In the selected peak
bin, reconstructed-token fractions are 0.15 (Qwen), 0.29 (Gemma-12B), 0.60 (Gemma-27B), and 0.76
(Llama). For sampled Taboo responses these are reproducible counterfactual continuations, not the
unobserved sampled history. The Gemma-27B and Llama late peaks should therefore be replicated on
fully stored responses before becoming a headline result.

## Mechanistic interpretation across models

Three mechanisms explain most visible structures without requiring a stronger causal claim.

1. **Delimiter/state-transition effects.** `resid_jump` and `resid_jump_nla` are distances between
   adjacent hidden states. Role changes and special tokens deliberately change serialization state,
   so large boundary values are expected. Delimiters can also carry a summary of prior content, as
   suggested by work on massive activations and semantic-unit endpoints, but a high residual jump
   alone cannot distinguish semantic compression from chat-format discontinuity.
2. **Massive-activation effects.** `dominant_mass`, `peak_ratio`, and `norm_ratio` are directly
   sensitive to activation geometry. Repeated BOS/role/delimiter maxima agree with
   [Sun et al. (2024)](https://openreview.net/forum?id=F7aAhfitX6), who identify large,
   input-agnostic activations at start and delimiter tokens. These tokens may be computationally
   special without producing a task-relevant NLA verbalization.
3. **Attention/position effects.** `sink_drain` measures negative attention mass on early sink tokens.
   Attention-sink geometry ([Xiao et al., 2023](https://arxiv.org/abs/2309.17453)) and attention
   grounding analyses such as [Lookback Lens](https://aclanthology.org/2024.emnlp-main.84/) predict
   systematic position and length dependence. Llama/OPI's near-monotone curve is therefore more
   parsimoniously explained as a positional proxy that happens to align with the appended injection.

Model size does not yield a monotonic pattern. The two Gemmas often resemble each other at fixed
turn boundaries because they share tokenizer/chat conventions, while Llama and Qwen behave
differently despite both using byte-level BPE. Tokenizer, attention architecture, NLA layer, and
selected metric family are all confounded with model identity. The current four-model comparison
cannot attribute a curve to parameter count.

## Validity issues that should appear in the paper

1. **Selection and evaluation use the same pooled data.** The case-cluster intervals account for
   token dependence within transcripts, but they do not remove winner's-curse optimism from choosing
   the largest absolute AUROC and reporting it on the same cases. BH-FDR controls exploratory tests;
   it does not evaluate the full metric-selection procedure on unseen transcripts.
2. **The objective is misaligned with deployment.** Pooled AUROC weights token pairs, whereas the
   deployment action is top-k retrieval within each case. Liars proves that the two objectives can
   disagree qualitatively. Held-out case-macro Average Precision, precision/recall at fixed NLA
   budgets, and a nested metric-selection estimate should be primary for a conference claim.
3. **Template filtering must be pre-specified.** Removing templates after seeing that they fail is a
   post-hoc intervention. A paper-ready experiment should define the candidate pool in advance and
   report both all-token and content-only policies. Boundary tokens can be evaluated as their own
   pre-specified stratum rather than mixed with content.
4. **The audit's content-winner table is easy to misread.** The displayed `content winner` is a
   provenance lookup, but `content AUROC` is computed for the *all-token selected metric after
   filtering*, not for the adjacent content-winner name. For changed winners, the actual
   content-winner raw AUROCs are: OPI/Gemma-27 `peak_ratio` 0.325 (lower), OPI/Llama `peak_ratio`
   0.261 (lower), TT/Gemma-27 `norm_ratio` 0.670 (higher), and Taboo/Llama `norm_ratio` 0.735
   (higher). The report should relabel these columns or report both metrics explicitly.
5. **The metric count in the draft is stale.** [`NLA_Best_Token.pdf`](../../NLA_Best_Token.pdf)
   states 11 candidates. The published common tables contain 13 blind candidates; OPI additionally
   contains `attn_rollout`, which is not blind because it uses the known cause span. `act_norm` is
   stored but is not in `CANDIDATE_METRICS`. The methods and multiplicity family must state these
   counts exactly.
6. **The judge target is not ground truth.** The judge sees only the verbalization, not the transcript
   or token. OPI's causally impossible early “injection” positives show that legitimate instruction
   language can satisfy the question. A selected metric predicts what this AV-plus-judge pipeline
   calls on-task, not intrinsic information content or behavioral causality. This is especially
   important because the original NLA work reports thematically plausible but false claims in
   verbalizations ([Fraser-Taliente et al., 2026](https://transformer-circuits.pub/2026/nla/)).
7. **Missing and reconstructed values are not random.** Difference metrics are undefined at the
   first token of each case. The plots correctly leave these ranks null, but early-bin means then use
   a slightly different token set. Late TT/Taboo results partly use reconstructed suffixes, with the
   strongest Taboo late peaks also having the largest reconstruction fraction.
8. **Cross-model comparisons are not uniformly paired.** OPI and Taboo share cases across models;
   Tensor Trust differs by up to nine cases; Liars has zero shared case IDs and materially different
   subdataset mixtures. Liars model differences are distributional, not paired model effects.

## Recommended NLA allocation policy from these results

For a prospective experiment, the following policy is the strongest one supportable by the current
evidence:

1. Exclude BOS and ordinary chat-template tokens from the general candidate pool. Keep the final
   assistant-generation boundary as a separate, small, explicitly budgeted stratum.
2. Rank only content tokens within each transcript; never compare raw scores across models or cases.
3. Use dataset/model-specific metrics:
   - OPI: lower `peak_ratio` for Qwen, Gemma-27, and Llama; higher `resid_jump_nla` for Gemma-12.
   - Tensor Trust: no selector is necessary in the dense regime; if forced, prioritize Qwen response
     onset and otherwise sample content.
   - Liars: do not use the pooled winner; use a declared last-response-token baseline until a
     held-out within-case selector beats it.
   - Taboo: higher `dominant_mass` for Qwen and both Gemmas after template removal; higher
     `norm_ratio` for Llama. Allocate one slot to the final generation boundary and the rest across
     response content, without treating a precise late percentile as established.
4. Evaluate the complete policy on held-out cases with case-macro AP and fixed budgets (e.g. 1, 2,
   4, 8, and 1%/10% tokens), alongside random, normalized-position, region, and “final token”
   baselines. Report bootstrap intervals for paired per-case differences.
5. Replicate Taboo on fully preserved outputs and validate the judge on stratified samples containing
   templates, legitimate instructions, injected instructions, and response content.

This policy is consistent with the draft's framing that an NLA verbalization is expensive and must be
aimed selectively, while respecting the central limitation: the bridge measures where an NLA
*verbalizes judged task content*, not which activation is objectively or causally most important.

## Relevant prior work cited in the draft

- [Natural Language Autoencoders](https://transformer-circuits.pub/2026/nla/) define the AV/AR
  framework and document both the richness and fallibility of activation verbalizations.
- [Massive Activations in Large Language Models](https://openreview.net/forum?id=F7aAhfitX6)
  supports the structural-token interpretation of BOS/delimiter activation peaks.
- [Locating and Editing Factual Associations in GPT](https://proceedings.neurips.cc/paper_files/paper/2022/hash/6f1d43d5a82a37e89b0665b33bf3a182-Abstract-Conference.html)
  motivates—but does not prove here—the possibility that semantic-unit endpoints accumulate a
  summary in the residual stream.
- [Efficient Streaming Language Models with Attention Sinks](https://arxiv.org/abs/2309.17453) and
  [Lookback Lens](https://aclanthology.org/2024.emnlp-main.84/) motivate the attention-sink and
  context-grounding interpretations, while also warning that such signals have intrinsic positional
  geometry.
- [Lost in the Middle](https://aclanthology.org/2024.tacl-1.9/) is relevant to long-context position
  effects, but most OPI and Taboo sequences here are short; it should not be used as a blanket
  explanation for every U-shaped curve.

