# Bridge experiment: design and validity

> Methods and validity document for the token-selector experiment that opens the
> NLA-calibration paper. It states the question, the pre-specified hypotheses, the
> operationalization, the datasets, the statistics, the controls, and the threats
> to validity with their mitigations. Written so the experiment can be run,
> checked, and reproduced from this file alone. Plain language throughout; every
> technical term is defined once.

## 1. Question and claim

An NLA (Natural Language Autoencoder) reads a language model's internal state at
one token and writes a paragraph explaining it. It is expensive: about 500
generated tokens per explanation. A long transcript has hundreds of tokens, and
most are uninteresting, so we cannot explain them all. The calibration paper
assumes you have already chosen *where* to point the NLA. This experiment earns
that assumption.

**Claim.** A cheap number, computed in one ordinary forward pass and before any
NLA call, identifies the tokens where the NLA will produce a *task-relevant*
explanation, well enough to spend the explanation budget by it.

"Task-relevant" is threat-specific: for a prompt injection, an explanation that
says the model is following an instruction embedded in its input; for
eval-awareness, that the model knows it is being tested; for deception, that the
model is lying or concealing.

## 2. Hypotheses (pre-specified)

The signals were fixed *before* the benchmark runs, from the earlier 27-case and
four-model results, so the benchmark tests are **confirmatory**, not a search.

- **H1 (localization).** The NLA is on-task at a higher rate inside the labelled
  threat span than outside it. Directional, one-sided.
- **H2 (prediction), primary signals.** A blind one-pass signal predicts where
  the NLA is on-task above chance. The pre-specified primary blind signal is
  `head_disagreement`; the pre-specified referenced (two-pass) primary is `kl`.
  Both were the winners on the four-model continuation test, so testing them on
  the external benchmarks is confirmatory.
- **H3 (scope).** Selection only helps when the on-task tokens are *rare*. When a
  whole response executes the threat (a fully compromised injection reply), the
  NLA is on-task nearly everywhere and there is nothing to select. This is a
  stated boundary condition, tested by contrasting dense and sparse regimes.

Every other signal is **exploratory** and reported as such, under multiple-
comparison control (§5.3). For the injection *input*-token setting, which is new,
Q2 is exploratory for all signals (no prior).

## 3. Operationalization

- **The NLA.** The AV checkpoint per model in `configs/base.yaml` (q7/g12/g27/l70),
  reading the raw residual stream at the registered layer. Decoding is greedy
  (temperature 0) so an explanation is a deterministic function of the activation.
- **The activation.** Captured by a forward hook on `model.model.layers[L]`,
  output[0], bf16 forward, f32 storage — identical to `build_corpus.py`, the
  convention the AV was trained under.
- **On-task label.** A judge reads each explanation and answers Yes/No to a fixed,
  threat-specific question (verbatim prompts in `bridge_judge_ontask.py`). Judge:
  `nvidia/DeepSeek-V4-Flash-NVFP4`, temperature 0. One binary label per token.
- **Cheap signals.** Computed on a single teacher-forced pass in
  `signals.py` (continuation) / `signals_injection.py`
  (input). Blind signals need one pass and no counterfactual; referenced signals
  need a second pass over a counterfactual.
- **Threat span.** The ground-truth region: the injected span (OPI, from the
  attacker's own `get_injected_prompt`), the decision span (hand eval cases), or
  the whole last assistant message (Liars', a response-level label).

## 4. Conditions

| condition | data | unit the NLA explains | span label | positives |
|---|---|---|---|---|
| Hand injection/eval | 27 authored cases | response tokens | marked span | dense (~half) |
| **OPI injection** | OpenPromptInjection, all 800 cases | input tokens | injected span | ~59% |
| **Liars' deception** | Liars' Bench, on-policy | response tokens | whole response (lie vs honest) | by design |

The hand cases are the pilot (already run: `bridge-hand.md`). The two
benchmarks are the confirmatory tests. OPI is run at full scale (all 800 cases,
every input token). Liars' Bench is run at the sample size fixed in §5.2.

Models: all four park families for OPI and the hand cases; for Liars' only the two
that *generated* the transcripts (Llama-3.3-70B, Gemma-3-27B), so the base model
we probe is the one that produced the text — an **on-policy** constraint that
avoids the off-policy confound the Liars' Bench authors warn about.

## 5. Statistical analysis

### 5.1 Effect sizes and their intervals

Two effect sizes, both reported with a 95% interval, never a bare point:

- **Q1, on-task lift** = (on-task rate inside the span) − (outside). A difference
  of proportions.
- **Q2, AUROC** of a signal against the on-task label. AUROC is itself an effect
  size (probability a random on-task token outscores a random off-task one);
  0.5 is chance.

**Clustered bootstrap.** Tokens within one transcript are correlated, so a naive
token-level interval is too narrow. All intervals resample **whole transcripts**
with replacement (case-cluster bootstrap, 2000 iterations, fixed seed), the same
scheme as the selector metrics. The effective sample size is the transcript
count, not the token count.

### 5.2 Sample size (Liars' Bench)

The unit is the transcript. For a proportion the 95% half-width is
`1.96·√(p(1−p)/n)`, worst case p=0.5: n=600 → ±0.040, **n=1000 → ±0.031**,
n=2400 → ±0.020. For AUROC≈0.7 (Hanley–McNeil, equal groups) n=500/group →
±0.03, n=1000/group → ±0.022. **Target: 1000 lying + 1000 honest transcripts per
on-policy model**, giving a ±0.03 interval on both effect sizes. Response tokens
are capped at ~30/transcript (enough to estimate a response's on-task fraction)
to bound NLA cost. OPI needs no such calculation — it is run in full.

### 5.3 Multiple comparisons

Q2 scans ~11 signals per mode per model. The primary signals (`head_disagreement`,
`kl`; §2) are reported without penalty as confirmatory. All exploratory signals
are reported with Benjamini–Hochberg false-discovery-rate control at q=0.05
across the exploratory family, and flagged as exploratory. We report the full
signal table, not only the winner, so the reader sees the selection.

### 5.4 Deterministic pipeline

Greedy NLA decode, greedy judge, fixed sampling seed for case selection and the
bootstrap, pinned checkpoints and `code_version`. Re-running reproduces the
numbers exactly.

## 6. Controls and robustness

- **Permutation control.** Shuffle the on-task labels within a mode and recompute
  each signal's AUROC; it must collapse to 0.5 (95% CI covering 0.5). Confirms the
  AUROC machinery is not manufacturing signal.
- **Random-signal baseline.** A standard-normal random score must score AUROC
  0.5. The reported signals must clear it.
- **Positive control / ceiling.** The threat-span label itself predicts on-task;
  its lift is the achievable ceiling a cheap signal is measured against.
- **Position confound.** Rolled-out attention concentrates on early sink tokens,
  so a signal can track *token position* rather than the threat (we already saw
  `in_attention` sit far below chance for this reason). Mitigation: report each
  signal's AUROC **within position bins** and as a partial AUROC controlling for
  normalized position; a signal that survives position control is real.
- **Length confound.** `head_disagreement` and raw attention entropy grow with
  sequence length. Mitigation: report both raw and per-transcript z-scored
  variants; the z-scored one removes any transcript-level scale effect.
- **Response-structure confound (Liars').** A high score can reflect the *shape*
  of the reply (a short denial versus a real answer) rather than the lie — the
  reason a near-perfect AUROC on the "impossible" HP-KR set is not trusted.
  Mitigation: report the balanced accuracy at an Alpaca-calibrated threshold
  alongside AUROC (the gap flags shape-reading), and restrict the headline to
  datasets where lie and honest responses are structurally matched.
- **Judge reliability.** The on-task labels are judge-produced, so we quantify the
  judge:
  - *Discrimination check.* Report the judge's base rate per mode; a judge that
    says Yes to everything is caught by a near-1.0 base rate (the hand-case
    injection base rate 0.90 is reported openly for this reason).
  - *Second-rater agreement.* On a random 200-explanation sample, a second judge
    (Llama-3.3-70B) answers the same questions; report Cohen's κ. κ≥0.6 supports
    the labels; lower means the on-task construct is noisy and the AUROC is a
    floor.
  - *Judge-free corroboration.* On the same sample, score on-task by embedding
    similarity between the explanation and the threat description (EmbeddingGemma,
    already in the project); the embedding score must correlate with the judge's
    Yes/No. This shows the result is not an artifact of one judge model.
  - *Prompt robustness.* The exact prompts are fixed and reported; a paraphrase of
    each question is run on the sample to confirm the label is not prompt-brittle.

## 7. Reproducibility

Everything is scripted and pinned:

- `bridge_extract_activations.py` (hand cases) / `bridge_extract_opi.py` (OPI) /
  the Liars' extractor — capture the NLA-layer activation, the span label, and the
  cheap signals in one pass.
- `bridge_run_nla.py` — greedy NLA decode per token via the AV server.
- `bridge_judge_ontask.py` — the judge (fixed model, fixed prompts) → on-task
  labels, and the `--analyze` step.
- `bridge_report.py` — the effect-size tables with clustered-bootstrap intervals,
  the controls, and the FDR-corrected exploratory columns.
- `run_bridge_all.sh` / `run_bridge_opi.sh` — the GPU-phased orchestration
  (base → AV server → judge, one heavy model at a time), resumable per phase.

Pinned: the four AV/AR checkpoints and layers in `configs/base.yaml`; the judge
`nvidia/DeepSeek-V4-Flash-NVFP4`; the OPI build (`build_injection_cases.py`, seed
0); the Liars' Bench revision; `code_version`; bootstrap seed 0.

## 8. Threats to validity

| threat | direction | mitigation |
|---|---|---|
| Judge is lenient / says Yes to all | inflates on-task rate, muddies Q1 | report base rate; second-rater κ; embedding corroboration |
| Judge is one model | result is a judge artifact | second judge + judge-free embedding score on a sample |
| Tokens within a response correlated | intervals too narrow | clustered bootstrap over transcripts |
| Fishing over 11 signals | false winner | pre-specified primaries; BH-FDR on the rest; full table shown |
| Position/sink confound | signal tracks position, not threat | within-position AUROC + position-controlled partial AUROC |
| Length confound | signal tracks transcript length | raw and per-transcript z-scored both reported |
| Reply-shape confound (Liars') | AUROC reads denial-vs-answer, not the lie | balanced accuracy vs AUROC gap; structurally matched datasets |
| Off-policy activations (Liars') | probing a model that did not produce the text | on-policy only (Llama-70B, Gemma-27B) |
| Dense positives (hand injection) | Q1 lift undetectable | run OPI (sparser) and state H3 as a scope condition |
| Non-determinism | numbers not reproducible | greedy decode + judge, fixed seeds, pinned checkpoints |

## 9. Scope and limitations

- The claim is about *where to spend NLA calls*, not about the NLA's accuracy —
  the calibration paper covers whether a given explanation's claims are supported;
  this experiment leaves that axis out deliberately.
- On-task-ness is a judged construct, not ground truth; §6 bounds how much to
  trust it, and the headline rests on effect sizes that survive the controls.
- Selection demonstrably helps only in the sparse-positive regime (H3). The two
  benchmarks probe both regimes, and the paper states the boundary rather than
  hiding it.
- Three threats (injection, eval-awareness, deception) and four model families
  are covered, but all are open-weight instruction models in the 7B–70B range;
  we do not claim beyond that class.
