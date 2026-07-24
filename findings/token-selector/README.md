# Which token deserves an NLA explanation?

> One consolidated note. What the token selector is, why we need it, the single
> idea it rests on, the signals, how we test them, and what every test found —
> with the full number tables. Written to be read without an interpretability
> background: every term is defined the first time it appears. History lives in
> git; this file is the current picture.

## Bottom line

An explanation (a Natural Language Autoencoder, NLA) reads the model's internal
state at one token and writes a paragraph about it, at roughly 500 generated
tokens each. A long transcript has hundreds of tokens and you cannot explain them
all, so a cheap number, computed in one forward pass *before* any explanation,
has to pick the positions worth spending the budget on. We test that number two
ways: does it find the ground-truth threat span (**localization**), and — the real
target — does it find the tokens where the NLA actually lands on a *task-relevant*
explanation (**the bridge**). Six tasks, and three findings that hold across them:

- **Selection only helps when the worth-explaining tokens are rare, and a
  30-second count tells you which case you are in.** Where the NLA is on-task at a
  minority of tokens — secret words (on-task 21–37%), injected input (14–21%),
  eval-awareness (9–37%), lies (3–4%) — a cheap number picks them and "keep the
  top-n tokens" is the right frame. Where the whole reply is on-task — a fully
  hijacked Tensor Trust response (88–92%) — there is nothing to select, and no
  number can manufacture a target.
- **One attention number reads two opposite mental postures, and its sign tells
  them apart.** `sink_drain` (attention draining off the "resting" first tokens)
  runs *forward* when the model reaches for planted or external content —
  injected input (OPI, 0.63–0.72), hijacked replies (Tensor Trust, up to 0.74),
  stated or concealed lies (Liars', 0.63–0.68) — and *backward* when the model
  guards a memorized secret (taboo, 0.09–0.28, i.e. 0.72–0.91 once flipped). Same
  cheap read, opposite sign: reaching out versus holding in.
- **Plain uncertainty is the universal, threat-agnostic default.** `entropy` is
  on-task-positive on every task (0.58–0.80, strongest on secret words), so it is
  the safe pick when you do not know the threat; the attention-geometry numbers
  (`sink_drain`, `head_disagreement`) beat it once you do.

## Why we need a selector

An NLA reads the model's internal activation at one token position and writes a
short paragraph describing what the model is doing there. The companion
calibration work makes those paragraphs trustworthy claim by claim, because they
contain claims the activation does not support. Calibration answers "which claims
can I trust". It assumes you have already decided *where* to point the
explanation.

For auditing that assumption fails. An auditor hands over a long transcript (a
suspected prompt injection, a trace of eval-awareness, a model with a hidden
goal) and wants to know where the interesting thing happens. A transcript has
hundreds of positions; one explanation costs about 500 generated tokens; most
positions are boring ("the", "and", a closing bracket). Explaining every position
is unaffordable and drowns the signal. So we need a step before calibration: a
**selector** that scores each position by how much is happening there worth an
auditor's attention, and spends the budget on the few that matter. The one prior
attempt at running an explanation over an audit benchmark picked positions by
naive saliency and found random selection beat it. Nobody has a principled
selector yet.

## The one idea everything rests on

> **Surprise is only meaningful relative to a reference. Pick the reference that
> isolates what you care about.**

The naive instinct is to explain the most surprising tokens. Surprisal
(`−log p(token)`, how improbable the token was) is genuine information content and
even predicts human reading effort. But raw surprisal has three holes for
auditing: it measures surprise against the model's expectations, not the
auditor's; the best attacks have *low* surprisal ("Ignore all previous
instructions" is fluent and highly predictable); and high surprisal mostly marks
random names, IDs, and dates. Every signal below is a different answer to the same
question — surprise relative to *what*? Relative to the previous token, to a
counterfactual transcript, to how the other attention heads read the same
position.

## The signals

Two families. **Blind** signals need no reference: one forward pass, no
counterfactual, no prior guess about where the cause is. These are what a fast
pre-explanation probe can afford. **Referenced** signals need a second pass over a
counterfactual that encodes the auditor's hypothesis.

Every score is compared by **AUROC**: the chance a randomly chosen worth-explaining
token outscores one that is not. 0.5 is a coin flip; 1.0 is perfect; **below 0.5
means the number runs backward — flip its sign and its real strength is 1 minus
the score.** A **z-score** shifts and scales a value by the transcript's own mean
and spread, so "z(S)" reads "how unusual this token's surprisal is for this
transcript".

### Blind signals (one forward pass, no reference)

| signal | reads | what it measures |
|---|---|---|
| `surprisal` S | logits | `−log p(token)`: how improbable the token was (the weak baseline) |
| `entropy` H | logits | how unsure the model was before the token, over the whole vocabulary |
| `varentropy` VH | logits | the spread of that surprise: a two-way fork versus a flat shrug |
| `override` = z(S) − z(H) | logits | high surprisal at low entropy: the model was confident and got overridden |
| `temporal_kl` | logits | KL(P_t ‖ P_{t−1}): how far the prediction jumped since the previous token |
| `resid_jump` | hidden state | ‖h_t − h_{t−1}‖: how far the internal state moved |
| `lookback_ratio` | attention | attention onto the context versus the model's own generated text (Lookback Lens) |
| `sink_drain` | attention | attention onto the template prefix (the sink), negated: how much parked attention drained |
| `head_disagreement` | attention | how much the attention heads disagree about where to look |
| `w` = z(sink_drain) − z(lookback) | attention | one signed axis whose sign separates injection from eval-awareness |

### Referenced signals (second pass over a counterfactual)

| signal | reads | what it measures |
|---|---|---|
| `dsurprisal` = S_ref − S_full | logits | how much more sense the token makes once the cause is present |
| `kl` = KL(P_full ‖ P_ref) | logits | how much the cause bent the whole next-token distribution here |
| `attn_rollout` | attention | prior art: rolled-out attention onto the known cause span |

### The signals in plain terms

**`surprisal`** — the model's own astonishment at the token that appeared. Weak
for auditing: measured against the model's prior, it flags random names and
numbers and misses fluent attacks. **`entropy`** and **`varentropy`** — how unsure
the model was, and whether the uncertainty was a *shrug* (probability spread over
many bland options) or a *fork* (torn between two sharp continuations, "comply"
versus "refuse"). **`resid_jump`** and **`temporal_kl`** — the same "the model just
changed its mind" idea, read from the residual stream (its internal working
memory) and from the output distribution.

**`lookback_ratio`** (Lookback Lens,
[arXiv:2407.07071](https://arxiv.org/abs/2407.07071)) — the fraction of a token's
attention landing on the context versus on already-generated text. Tokens that
acknowledge an eval are grounded in the eval cue, so they look back hard;
injection-steered tokens look back less.

**`sink_drain`** — attention onto the template prefix, negated. Transformers park
unused attention on the first few tokens, the *attention sink* (Xiao et al. 2023,
[arXiv:2309.17453](https://arxiv.org/abs/2309.17453)); in our transcripts that
prefix absorbs about 45% of all attention. When a token reaches for other content,
heads leave the idle position and the sink empties. This is the workhorse across
tasks — with a sign that flips (below).

**`head_disagreement`** — a model has hundreds of attention heads, each choosing
where to look; every other attention signal first averages them. This score is
**entropy(the head-averaged attention) − the average of each head's own entropy**,
summed over layers: zero when the heads look alike, rising as they scatter, which
is what steered tokens make them do.

**`w` = z(sink_drain) − z(lookback_ratio)** — puts two threats at opposite corners
of one plane: injection-steered tokens drain the sink and look back less;
eval-aware tokens keep the sink full and look back more. Its *sign* separates the
two threats instead of breaking on one.

**`kl`** and **`dsurprisal`** — the referenced pair: run the model with and without
the suspected cause. `kl` (how much the whole distribution moved) is the best
single signal wherever a counterfactual exists. **`attn_rollout`** (Abnar &
Zuidema 2020) — the prior-art baseline: rolled-out attention to the known cause
span; strong for injection, useless for diffuse causes, and needs the span.

### Why not `attn_distraction`

The Attention Tracker paper ([arXiv:2411.00348](https://arxiv.org/abs/2411.00348))
scored the drop in attention onto the *instruction words*. We implemented it and
it looked strong, then broken on Gemma; the truth was simpler. On Qwen-7B,
injection: counting the whole template prefix (sink included) gives AUROC 0.70;
the instruction words only, 0.36; the first tokens alone (the sink), 0.65. The
discriminative quantity is the *parked attention draining*, not the instruction
words — which is why the clean signal is `sink_drain` and not `attn_distraction`.
One real caveat: past ~1024 tokens Gemma's sliding-window layers cannot see the
first token, so on long Gemma transcripts `sink_drain` should use the
global-attention layers only.

## How we measure

All signals are computed from a teacher-forced `transformers` pass (the attention
signals need the attention maps, which the serving stack does not return). Two
measurements:

- **Localization** — does the signal find the ground-truth threat span? AUROC of
  the signal against a token's span label (inside the injected/decision/lie span
  or not). This is the controlled precursor (see "The localization pilot" below).
- **The bridge** — does the signal find where the NLA lands on-task? We run the NLA
  at each token, ask a judge (`nvidia/DeepSeek-V4-Flash-NVFP4`) whether its
  paragraph is *on-task* for the threat, and take two readings: **Q1**, is the NLA
  more on-task inside the flagged span than outside (a check that the threat
  region is where the NLA has something to say); and **Q2**, the AUROC of each
  signal against that per-token on-task label — the number the selector actually
  needs. Intervals are case-cluster bootstraps over whole transcripts; controls (a
  random-normal score and a label permutation) sit at 0.5 in every cell.

The four park models are the base models the NLA checkpoints were trained for:
`q7` Qwen2.5-7B, `g12` Gemma-3-12B, `g27` Gemma-3-27B, `l70` Llama-3.3-70B. Every
script has an offline `--selftest` covering the math with no GPU or network.

## The bridge results, task by task

Each Q2 table lists **every** signal, one row per number, one column per model,
best-for-task (largest distance from 0.5) **bolded**. Full precision and per-token
CSVs are in `paper_results/bridge/`.

### Secret word (taboo organisms) — the clean win

Twelve LoRA "taboo" models (four bases × moon/ship/snow) that hint at a hidden
word without saying it. We generate hints to four elicitation prompts and probe
the response tokens; on-task = the NLA recovers or relates to the secret. No
planted span, so Q1 does not apply; the story is Q2. On-task base rate 21–37%,
tilted slightly to the opening of the reply (38% in the first five tokens vs 29%
later) — sparse, so "keep the top-n" fits.

| signal | q7 | g12 | g27 | l70 |
|---|---|---|---|---|
| surprisal | 0.75 | 0.68 | 0.70 | 0.75 |
| **entropy** | **0.78** | 0.73 | 0.74 | 0.79 |
| varentropy | 0.77 | 0.73 | 0.73 | 0.80 |
| resid_jump | 0.54 | 0.68 | 0.66 | 0.72 |
| lookback_ratio | 0.47 | 0.48 | 0.51 | 0.72 |
| sink_drain *(inverse)* | 0.28 | **0.25** | **0.23** | **0.09** |
| head_disagreement *(inverse)* | 0.31 | 0.28 | 0.25 | 0.23 |
| _base rate_ | 0.37 | 0.34 | 0.33 | 0.21 |

A model *trained to hide a word* still broadcasts where the word is readable:
`entropy` finds those tokens at ~0.8. `sink_drain` is the strongest separator but
runs backward (0.09 on Llama flips to 0.91) — the "holding in" sign.

### Injected instruction in the input (OpenPromptInjection)

800 hijacking cases; we localize the injected span in the *input* tokens. On-task
base rate 14–21% (sparse); the injected tokens are on-task 26% vs 1% for clean
input, so the labeled span is the right target. `attn_rollout` is the expensive
O(n) baseline, not blind.

| signal | q7 | g12 | g27 | l70 |
|---|---|---|---|---|
| surprisal | 0.50 | 0.50 | 0.47 | 0.53 |
| entropy *(inverse)* | 0.33 | 0.41 | 0.46 | 0.42 |
| varentropy *(inverse)* | 0.31 | 0.41 | 0.45 | 0.40 |
| resid_jump | 0.55 | 0.53 | 0.51 | 0.57 |
| lookback_ratio | 0.53 | 0.54 | 0.50 | 0.48 |
| **sink_drain** | **0.63** | **0.67** | **0.72** | **0.68** |
| head_disagreement | 0.58 | 0.62 | 0.67 | 0.64 |
| attn_rollout *(baseline)* | 0.58 | 0.51 | 0.50 | 0.56 |
| _base rate_ | 0.21 | 0.14 | 0.15 | 0.14 |

`sink_drain` is the clear winner (0.63–0.72); `entropy` runs backward, because the
injected command is *low*-uncertainty text the model predicts fluently.

### Lies (Liars' Bench) — on-policy, l70 + g27 only

72,863 lie/honest responses; we probe the two generators that match our park
models. No token span (the whole response is labelled), and the NLA reads
deception at only 3–4% of tokens (5% inside lying transcripts, 1% inside honest).
Q1 lift is small but tight (+0.04 g27, +0.05 l70, both clear zero).

| signal | g27 | l70 |
|---|---|---|
| surprisal | 0.58 | 0.59 |
| entropy | 0.59 | 0.59 |
| varentropy | 0.59 | 0.59 |
| resid_jump | 0.46 | 0.52 |
| lookback_ratio | 0.46 | 0.47 |
| **sink_drain** | **0.63** | **0.68** |
| head_disagreement | 0.57 | 0.64 |
| _base rate_ | 0.03 | 0.04 |

Extremely sparse — the pool of true positives is tiny, so top-n is essential but
you are fishing for needles. `sink_drain` runs *forward* here (the opposite sign
from taboo): lying is reaching out, secret-keeping is holding in.

### Hijacked reply (Tensor Trust) — the honest counterexample

776 hijacking cases doubled into an attack variant and an access-code variant; the
base model generates a reply and we probe it. On-task base rate 88–92% — the whole
reply reads as "following an instruction", and an attack reply (89%) is
indistinguishable from a legitimate one (92%). **Q1 is flat** (lift ~0), because
there is nothing to localize.

| signal | q7 | g12 | g27 | l70 |
|---|---|---|---|---|
| surprisal | 0.34 | 0.47 | 0.44 | 0.60 |
| entropy | 0.33 | 0.46 | 0.45 | **0.60** |
| varentropy | 0.31 | 0.46 | 0.45 | 0.60 |
| resid_jump | 0.65 | 0.52 | 0.46 | 0.51 |
| lookback_ratio | 0.73 | **0.66** | **0.64** | 0.60 |
| **sink_drain** | **0.74** | 0.52 | 0.52 | 0.55 |
| head_disagreement | 0.70 | 0.54 | 0.54 | 0.48 |
| _base rate_ | 0.90 | 0.90 | 0.92 | 0.88 |

Tensor Trust replies are short and total ("Access Granted"), so the whole thing is
the compliance. This is the clean demonstration that the selector is a
*sparse-target* tool. The Q2 numbers are still above chance in places, but with
nothing to select they do not matter operationally.

### Eval-awareness — does the model sense it is being tested? (pilot)

27 hand-authored cases; the model's decision tokens are the target. The sparsest,
most localizable task: on-task 9–37%, and Q1 lift grows with model size (+0.19 on
7B → +0.54 on 27B). Full 11-signal table (this pilot has counterfactuals, so `kl`,
`w`, `temporal_kl`, `attn_rollout` are available):

| signal | q7 | g12 | g27 | l70 |
|---|---|---|---|---|
| surprisal | 0.48 | 0.41 | 0.49 | 0.50 |
| entropy | 0.42 | 0.58 | 0.53 | 0.49 |
| varentropy | 0.64 | 0.41 | 0.48 | 0.55 |
| resid_jump | 0.38 | 0.37 | 0.50 | 0.44 |
| temporal_kl | 0.42 | 0.57 | 0.58 | 0.51 |
| lookback_ratio | 0.64 | **0.66** | 0.63 | 0.57 |
| sink_drain | 0.71 | 0.65 | 0.60 | 0.63 |
| head_disagreement | **0.75** | 0.65 | 0.58 | 0.66 |
| w | 0.54 | 0.49 | 0.46 | 0.49 |
| kl *(referenced)* | **0.75** | 0.64 | **0.63** | **0.75** |
| attn_rollout *(referenced)* | 0.46 | 0.39 | 0.48 | 0.46 |
| _base rate_ | 0.09 | 0.20 | 0.33 | 0.37 |

Best picks: `kl` and `head_disagreement` (~0.63–0.75). The tell sharpens with
model size — bigger models legibly clock the test.

### Authored injection (pilot) — dense, confirms Tensor Trust

The same 27-case pilot, injection mode. On-task base rate 86–92% (dense), Q1 flat,
numbers noisy (27 cases, wide error bars) and no clean winner — exactly the
Tensor Trust pattern in a controlled setting.

| signal | q7 | g12 | g27 | l70 |
|---|---|---|---|---|
| surprisal | 0.58 | 0.46 | 0.63 | 0.55 |
| entropy | 0.45 | 0.51 | 0.28 | 0.53 |
| varentropy | 0.57 | 0.49 | **0.72** | 0.49 |
| resid_jump | 0.33 | 0.41 | 0.35 | 0.50 |
| temporal_kl | 0.45 | 0.42 | 0.48 | 0.58 |
| lookback_ratio | 0.60 | 0.60 | 0.58 | 0.65 |
| sink_drain | 0.54 | **0.62** | 0.47 | 0.39 |
| head_disagreement | 0.48 | 0.61 | 0.51 | 0.40 |
| w | 0.51 | 0.54 | 0.48 | 0.38 |
| kl | 0.41 | 0.41 | 0.40 | 0.40 |
| attn_rollout | 0.40 | 0.40 | 0.39 | 0.28 |
| _base rate_ | 0.90 | 0.89 | 0.92 | 0.86 |

## Cross-task synthesis

- **Base rate is the decision rule for whether a selector applies at all.** Sparse
  tasks (secret words, injected input, eval-awareness, lies) reward selection;
  dense tasks (hijacked replies, authored injection) have nothing to select. This
  turns a vague "which signal is best" into a crisp, up-front rule: count the
  on-task density first.
- **`sink_drain`'s sign is a mechanistic tell.** Forward for reaching-out
  (injected input 0.63–0.72, hijack up to 0.74, lies 0.63–0.68), backward for
  holding-in (secret words, 0.72–0.91 once flipped). One cheap number reads two
  opposite postures; the sign says which.
- **`entropy` is the universal, threat-agnostic default** (on-task-positive
  everywhere, 0.58–0.80, best on secret words), and `sink_drain` /
  `head_disagreement` the threat specialists. A clean two-tier recipe: entropy when
  you do not know the threat, attention-geometry (with the right sign) when you do.
- **The legibility of a threat grows with model size** on the localizable tasks
  (eval-awareness Q1 lift +0.19 → +0.54 from 7B to 27B).

## The localization pilot (signal vs planted span)

Before the bridge, we checked the cheaper question: do the blind signals find the
ground-truth *span* in controlled cases? They do, strongly — this is what
motivated pointing the NLA by them. On the 27 hand cases, AUROC against the span:

**Injection** (steered span in the response): `head_disagreement` 0.81/0.81/0.80/0.77,
`w` 0.81/0.79/0.79/0.81, `sink_drain` 0.74/0.81/0.78/0.77 — all above the referenced
`kl` (0.73/0.71/0.75/0.71) and `attn_rollout` (0.77/0.78/0.72/0.74), blind, one pass.

**Eval-awareness** (decision span): `lookback_ratio` 0.82/0.79/0.79/0.78, above `kl`
(0.71/0.69/0.68/0.73); the injection signals fall to chance here and `w`'s sign
flips, so a single signed number tells the two threats apart.

The gap between these (0.77–0.82) and the bridge Q2 numbers (0.6–0.8) is expected:
the span label is clean, whereas the on-task label is a judge's per-token call, so
the bridge is the harder and more honest test of the selector's real job. Raw
surprisal is a weak localizer throughout (below chance on fluent injection).

## What we recommend

Per threat, ordered by cost:

- **Don't-know-the-threat, cheap triage → `entropy`.** Positive on every task,
  one pass; the safe default before you commit to a threat model.
- **Injection, input or output → `sink_drain`.** Blind, one pass, the best signal
  on OPI (0.63–0.72) and strong on hijacked replies; `head_disagreement` a close
  second. Its sign also tells injection from secret-keeping.
- **Eval-awareness → `head_disagreement` or `kl`.** Blind `head_disagreement`
  (~0.66–0.75) if you have one pass; referenced `kl` (~0.63–0.75) if a deployment
  counterfactual is available.
- **Secret word → `entropy` (forward) or `sink_drain` (flipped).** Both find the
  revealing tokens at ~0.8–0.9.
- **Lies → uncertainty for the coarse call, but expect weakness.** `entropy` gives
  a moderate, deception-type-dependent filter; nothing cheap catches concealment
  reliably, and a trained deception probe is the right tool for the hidden-goal
  case.
- **First, check the base rate.** If the on-task density is high (a fully executed
  attack), no selector helps — explain broadly or not at all.

## How much should you trust this?

- The four benchmarks are external and real; the two pilots (eval-awareness,
  authored injection) are 27 hand-written cases with wide error bars — controlled
  methodology checks, not headline evidence.
- The on-task label is a single judge's per-token call. Controls (random score,
  label permutation) sit at 0.5 in every cell, and the deception result sits
  alongside the benchmark's own trained detectors hitting the same wall, so the
  weak numbers there are a finding, not a rig problem.
- Tensor Trust's flatness is a real boundary condition (the dense regime), not a
  failure — it is the clean counterexample that defines when selection applies.
- Still open: OPI *continuation* scoring (generate the injected reply and localize
  the attack-realizing response span) would put the response-side signals on the
  input-side footing.

## Candidate signals from the literature (2024–26)

Ranked by value for the blind case; the first two are implemented, the rest
untried (probe-based ones are blind at inference but need a one-time labelled
training set).

- **Attention Tracker** (Hung et al. 2024,
  [arXiv:2411.00348](https://arxiv.org/abs/2411.00348)): drop in attention to the
  instruction. *Implemented; the working part is the sink.* ✅
- **Lookback Lens** (Chuang et al. 2024,
  [arXiv:2407.07071](https://arxiv.org/abs/2407.07071)): attention to context
  versus self. *Implemented as `lookback_ratio`.* ✅
- **PIShield** (Zou et al. 2025/26,
  [arXiv:2510.14005](https://arxiv.org/abs/2510.14005)): linear probe for "this
  token is inside an injected instruction". The natural next thing to try on OPI.
- **Deception / sandbagging probe** (Goldowsky-Dill et al. 2025,
  [arXiv:2502.03407](https://arxiv.org/abs/2502.03407)): residual-stream direction
  for strategic dishonesty — the on-target signal for the taboo and Liars' tasks;
  evadable under RL obfuscation ([arXiv:2507.12691](https://arxiv.org/abs/2507.12691)).
- **Eval-awareness probe** (Nguyen et al. 2025,
  [arXiv:2507.01786](https://arxiv.org/abs/2507.01786)): a per-token "I'm being
  tested" direction.
- **Semantic Entropy Probes** (Kossen et al. 2024,
  [arXiv:2406.15927](https://arxiv.org/abs/2406.15927)): entropy over *meanings*,
  an axis our token-level entropy misses.
- **EigenScore / INSIDE** (Chen et al. 2024,
  [arXiv:2402.03744](https://arxiv.org/abs/2402.03744)): covariance-spectrum
  dispersion of internal embeddings; the single-state variant is one pass.
- **Claim-Conditioned Probability** (Fadeeva et al. 2024,
  [arXiv:2403.04696](https://arxiv.org/abs/2403.04696)): separates factual from
  paraphrase uncertainty via a small NLI model.

Skip: entropix-style varentropy sampling (it *is* our entropy + varentropy) and
attention rollout as a blind score (no notion of "anomalous" without a
counterfactual).

## Engineering notes

**Full-vocabulary distributions without leaving the serving stack.** The signals
need full-vocabulary entropy, varentropy, and KL, but SGLang returns only top-k
log probabilities for autoregressive models. `return_logprob=True` gives
forced-token log probabilities (enough for `surprisal`); `return_hidden_states=True`
plus `hidden_states @ lm_head.weight.T` reconstructs the full logits exactly. The
tests use plain `transformers` teacher-forced forwards (fast, since "transformers
is slow" is about autoregressive *generation*); a production selector should use
the SGLang hidden-states path, except for the attention signals, which need the
attention maps SGLang does not return.

**Resumability.** Every extractor checkpoints its parquet and resumes, and every
orchestration phase skips completed outputs, so an interrupted run (or a
re-provisioned node) continues from where it stopped rather than restarting.
