# Which token deserves an explanation?

> Everything the token selector study covers: what the problem is, which cheap
> numbers we tried, how they were tested, and what the tests found, with the
> full number tables. Written to be read without a background in
> interpretability. Every term is explained the first time it appears.

## Bottom line

Three results hold across all five benchmarks.

**Whether choosing helps at all depends on how common the target is.** Where the
explanation is on-task at a minority of positions, a cheap number finds them.
Where nearly every position is already on-task, no number can help, and counting
that share is a quick first check that tells an auditor which case they are in.

**No single number works for every threat, and several point the wrong way on
some of them.** Uncertainty finds a concealed word and points away from a planted
instruction. Attention head agreement finds lies and disagreement finds injected
input. The direction has to be fixed per threat before use.

**The plainest measurements of the internal state carry more than expected.** How
large the state is, how much one channel dominates it, and how far it moved since
the previous token are the strongest signals on several benchmarks. They cost
nothing extra, because that state is already computed in order to write the
explanation.

## The problem in one paragraph

A **natural language autoencoder** reads a language model's internal state at a
single token position and writes a short paragraph, in ordinary English, about
what the model appears to be doing at that point. We shorten the name to NLA.
Each paragraph costs roughly 500 generated tokens, and a transcript worth
auditing has hundreds of positions, so explaining every position is far too
expensive. Someone auditing a transcript therefore needs a way to choose. The
question this study asks is whether a **cheap number**, computed in one ordinary
forward pass before any paragraph is written, can rank the positions so that the
few explanations you can afford land where something worth reading is happening.

We call such a number a **signal**, and the procedure that ranks positions by a
signal a **selector**.

## Why the choice matters

An auditor is handed a long transcript and a suspicion: perhaps the model was
steered by text hidden in its input, perhaps it is concealing something. The
auditor wants to know where in the transcript the interesting thing happens.
Most positions are ordinary words that carry nothing of interest. Spending the
whole budget on those wastes it, and the interesting positions stay unexamined.

The natural instinct is to explain the positions where the model was most
surprised. That instinct is weak here, for three reasons. Surprise is measured
against the model's own expectations rather than the auditor's. The most
effective attacks read fluently, so the model finds them easy to predict. And
high surprise mostly marks names, numbers and dates, which are rarely what an
auditor is looking for.

Every signal below is a different answer to one question: **surprise compared to
what?** Compared to the previous token, to how the attention heads read the same
position, or to the size of the internal state itself.

## The signals

Each signal is one number per token position. All of them come from a single
forward pass of the model over the transcript, so computing all of them together
costs about the same as computing one. They fall into three families, grouped by
which part of the model they read.

A note on two words used throughout. The **residual stream** is the running
internal state that a transformer carries from one layer to the next, a long
list of numbers per token. **Attention** is the mechanism by which a token
gathers information from earlier tokens, and a model has hundreds of separate
attention **heads**, each choosing where to look.

### Read from the predicted next word

| signal | what it measures |
|---|---|
| `surprisal` | how improbable the token that appeared was, given everything before it |
| `entropy` | how unsure the model was about what would come next, across the whole vocabulary |
| `varentropy` | whether that uncertainty was a shrug, with probability spread thinly over many bland options, or a fork, with the model torn between two sharp alternatives |
| `temporal_kl` | how far the prediction moved between the previous token and this one |

### Read from the attention pattern

| signal | what it measures |
|---|---|
| `lookback_ratio` | the share of a token's attention landing on the context rather than on text the model has written itself |
| `sink_drain` | how much attention has left the first few tokens of the transcript, where a transformer parks attention it is not using |
| `head_disagreement` | how much the attention heads disagree about where to look |
| `w` | the difference between `sink_drain` and `lookback_ratio`, each first put on a common scale |
| `attn_rollout` | attention traced back through the layers onto a span already known to be suspicious |

### Read from the internal state

| signal | what it measures |
|---|---|
| `resid_jump` | how far the internal state moved between the previous token and this one, at the model's last layer |
| `resid_jump_nla` | the same distance measured at the layer the NLA reads, with the few dominant channels removed |
| `act_norm` | the overall size of the internal state at the layer the NLA reads |
| `norm_ratio` | that size compared with the typical size in the same transcript |
| `peak_ratio` | how much the single largest channel stands out from the rest |
| `dominant_mass` | the share of the state's size held by the handful of channels that dominate it |

### Two of these need something extra

Most signals need nothing but the transcript itself. Two do not, and the
difference matters when judging how practical they are.

`attn_rollout` needs someone to have already marked the suspicious span, so it
can only be used once you know where to look. `kl`, which appears in the pilot
results below, measures how far the model's prediction moves when the suspected
cause is deleted from the transcript, so it needs a second forward pass over an
edited copy. Writing that edited copy is possible when the cause is a piece of
text someone inserted. It is not possible when the thing being audited is a
property of the model, such as a word it has been trained to conceal.

### Putting numbers on a common scale

Several signals are compared by their **z-score**, which says how far a value
sits from the average of its own transcript, counted in standard deviations. A
z-score of 2 means the value is unusually high for that transcript. Comparing
within a transcript matters because transcripts differ in length and subject.

## Two things a transformer does that the signals exploit

Both come from work by Sun, Canziani, LeCun and Zhu, who took them apart in
detail ([arXiv:2603.05498](https://arxiv.org/abs/2603.05498)).

An **attention sink** is the habit transformers have of parking attention they
are not using on the first few tokens of a sequence. In our transcripts those
opening tokens absorb roughly 45% of all attention. The parking is per head, and
it works as a switch: a head that has parked its attention is effectively
switched off, and heads switched off in this way tend to look only at nearby
words. When a token reaches for content further away, heads leave the parking
spot and the sink empties. `sink_drain` measures that emptying, and
`head_disagreement` measures the same event from the other side, because heads
that have all parked in one place agree with each other by construction.

A **massive activation** is a small number of channels in the internal state
holding values orders of magnitude larger than the rest. The channels involved
are fixed for a given model and barely depend on the input. A token whose state
carries one is called a **spike token** in that work, and over 98% of the
vocabulary becomes a spike token when placed first in a sequence. This is
positional rather than meaningful, and it matters here because the NLA reads
exactly the layer where these channels dominate. `norm_ratio`, `peak_ratio` and
`dominant_mass` are three ways of asking how spiky a token's state is.

## How the test works

**The pool.** Every token of the fully rendered transcript is scored, including
the chat template scaffolding that wraps the messages. This is the pool a
deployed selector would face. Each token is tagged with its **region**: template
scaffolding, the system message, the user message, or the assistant reply.

**The label.** At every position we run the NLA, then ask a judge model
(`nvidia/DeepSeek-V4-Flash-NVFP4`) a single question: does this paragraph speak
to the threat being audited? For the hidden word benchmark the question is
whether the paragraph reveals or points at the word. A position where the answer
is yes is called **on-task**. This label is what a selector is trying to find.

**The score.** Signals are compared by **AUROC**, which is the chance that a
randomly chosen on-task position scores higher than a randomly chosen position
that is not on-task. A value of 0.5 is a coin flip. A value **below 0.5 means
the signal runs backwards**, and its real strength is one minus the value, so
0.28 is as informative as 0.72 once the sign is flipped.

**The floor.** The **base rate** is the share of all positions that are on-task.
It is what an auditor gets by choosing positions at random, and it says whether
choosing is worth anything at all. Where nearly every position is on-task there
is nothing to select.

**The intervals.** Every AUROC comes with a 95% interval from a **bootstrap**:
the measurement is repeated on 2000 resampled versions of the data, and the
interval covers the middle 95% of the answers. The resampling draws whole
transcripts rather than individual tokens, because tokens inside one transcript
are related to each other and treating them as independent would make the
intervals too narrow.

**The controls.** Two numbers are computed alongside every table: a score drawn
at random, and the AUROC of the best signal after the labels have been shuffled.
Both sit at 0.5 when the machinery is working. They appear in the last row of
every table below, and a run where they drift away from 0.5 is broken.

**Multiple comparisons.** Two signals were chosen before the benchmarks were
run and are reported without a penalty: `head_disagreement` where only one
forward pass is allowed, and `kl` where an edited transcript is available. Every
other signal is exploratory, and is marked with a dagger when it survives
Benjamini-Hochberg control of the false discovery rate at 0.05, a standard way of
allowing for the fact that testing many signals at once will throw up some
apparent winners by chance.

**Reading the tables.** In each table there is one row per signal and one column
per model. The entry furthest from 0.5, in either direction, is shown in bold.
An asterisk marks a signal needing only one forward pass. A dagger marks a
signal surviving the false discovery rate control described above.

## The models and the benchmarks

Four open models are audited, each the base model its NLA was trained for:
Qwen2.5-7B, Gemma-3-12B, Gemma-3-27B and Llama-3.3-70B. Five benchmarks:

| benchmark | what is being audited | positions |
|---|---|---|
| OpenPromptInjection | an instruction hidden in the input | 111,370 to 126,340 |
| taboo organisms | a word the model was trained to hide | 5,260 to 8,583 |
| Tensor Trust | a reply produced by a successful hijack | 528,355 to 564,579 |
| Liars' Bench | a stated or concealed lie | 593,896 and 1,444,733 |
| authored cases | a controlled pilot, 27 hand written transcripts | 1,574 to 2,232 |

Liars' Bench covers Gemma-3-27B and Llama-3.3-70B only, because the benchmark
text has to have been written by the model being audited. Together the five
benchmarks cover 4,712,681 positions, every one of which has an explanation and
a judge's answer.

## Results, benchmark by benchmark

Each benchmark gets three tables: how often the explanation is on-task inside
the region under suspicion compared with outside it, how the positions are
distributed across regions, and the AUROC of every signal for every model.

### An instruction hidden in the input (OpenPromptInjection)

An attacker hides an instruction inside text the model is asked to process, in
the hope the model follows it. There are 800 such cases, and the hidden span is
known, so we can check whether explanations land on it.

The explanations do land on it. Inside the hidden span they are on-task roughly
twice as often as outside, and every interval clears zero comfortably:

| model | threat | inside | outside | lift [95% CI] |
|---|---|---|---|---|
| Qwen-7B | injection (input span) | 0.34 (n=50520) | 0.18 (n=60850) | **+0.16** [+0.16, +0.17] ✓ |
| Gemma-12B | injection (input span) | 0.22 (n=50160) | 0.12 (n=57590) | **+0.10** [+0.09, +0.10] ✓ |
| Gemma-27B | injection (input span) | 0.24 (n=50160) | 0.13 (n=57590) | **+0.11** [+0.11, +0.12] ✓ |
| Llama-70B | injection (input span) | 0.23 (n=48780) | 0.06 (n=77560) | **+0.17** [+0.16, +0.17] ✓ |

Positions are dominated by the user message, which is where the attack sits. The
system message is short and has a higher on-task rate, which is worth keeping in
mind when reading a number computed over the whole pool:

| model | region | share | on-task |
|---|---|---|---|
| Qwen-7B | user | 0.77 | 0.21 |
| Qwen-7B | system | 0.14 | 0.47 |
| Qwen-7B | template | 0.09 | 0.31 |
| Gemma-12B | user | 0.79 | 0.14 |
| Gemma-12B | system | 0.14 | 0.32 |
| Gemma-12B | template | 0.07 | 0.22 |
| Gemma-27B | user | 0.79 | 0.15 |
| Gemma-27B | system | 0.14 | 0.34 |
| Gemma-27B | template | 0.07 | 0.30 |
| Llama-70B | user | 0.66 | 0.14 |
| Llama-70B | template | 0.22 | 0.02 |
| Llama-70B | system | 0.12 | 0.28 |

The full table of signals:

| signal | Qwen-7B | Gemma-12B | Gemma-27B | Llama-70B |
|---|---|---|---|---|
| `surprisal` | 0.511 [0.51,0.51]*† | 0.506 [0.50,0.51]*† | 0.495 [0.49,0.50]*† | 0.513 [0.51,0.52]*† |
| `entropy` | 0.372 [0.37,0.38]*† | 0.397 [0.39,0.40]*† | 0.452 [0.45,0.46]*† | 0.451 [0.45,0.46]*† |
| `varentropy` | 0.366 [0.36,0.37]*† | 0.391 [0.39,0.40]*† | 0.442 [0.44,0.45]*† | 0.424 [0.42,0.43]*† |
| `temporal_kl` | 0.606 [0.60,0.61]*† | 0.508 [0.50,0.51]*† | 0.509 [0.51,0.51]*† | 0.482 [0.48,0.49]*† |
| `resid_jump` | 0.554 [0.55,0.56]*† | 0.541 [0.54,0.55]*† | 0.515 [0.51,0.52]*† | 0.501 [0.50,0.51]* |
| `lookback_ratio` | 0.602 [0.59,0.61]*† | 0.603 [0.60,0.61]*† | 0.582 [0.58,0.59]*† | 0.469 [0.46,0.47]*† |
| `sink_drain` | 0.531 [0.53,0.54]*† | 0.568 [0.56,0.57]*† | 0.596 [0.59,0.60]*† | **0.673 [0.67,0.68]*†** |
| `head_disagreement` | 0.497 [0.49,0.50]* | 0.545 [0.54,0.55]*† | 0.574 [0.57,0.58]*† | 0.641 [0.64,0.65]*† |
| `w` | 0.468 [0.46,0.47]*† | 0.484 [0.48,0.49]*† | 0.505 [0.50,0.51]* | 0.608 [0.60,0.61]*† |
| `norm_ratio` | 0.359 [0.35,0.36]*† | 0.603 [0.60,0.61]*† | 0.618 [0.61,0.62]*† | 0.345 [0.34,0.35]*† |
| `peak_ratio` | **0.239 [0.24,0.24]*†** | 0.414 [0.41,0.42]*† | 0.384 [0.38,0.39]*† | 0.361 [0.36,0.36]*† |
| `dominant_mass` | 0.261 [0.26,0.27]*† | 0.412 [0.41,0.42]*† | 0.427 [0.42,0.43]*† | 0.380 [0.38,0.38]*† |
| `resid_jump_nla` | 0.596 [0.59,0.60]*† | **0.685 [0.68,0.69]*†** | **0.635 [0.63,0.64]*†** | 0.463 [0.46,0.47]*† |
| `attn_rollout` | 0.603 [0.60,0.61]† | 0.565 [0.56,0.57]† | 0.559 [0.55,0.56]† | 0.502 [0.50,0.51] |
| _base rate_ | 0.25 | 0.17 | 0.18 | 0.13 |
| _control rand / perm_ | 0.50 / 0.5±0.00 | 0.50 / 0.5±0.00 | 0.50 / 0.5±0.00 | 0.50 / 0.5±0.00 |

Three things stand out. `resid_jump_nla` is the strongest signal on the two
Gemma models, and `sink_drain` on Llama-70B, so movement of the internal state
and emptying of the attention parking spot are both picking up the attack.
`entropy` and `varentropy` run backwards on all four models, sitting between
0.366 and 0.452, which fits the nature of the attack: a hidden instruction is
fluent, ordinary text that the model finds easy to predict, so the positions
where it is most uncertain are the wrong ones to explain. `peak_ratio` on
Qwen-7B is the furthest of any entry from a coin flip at 0.239, which is 0.761
once the sign is flipped, and it also runs backwards.

### A word the model was trained to hide (taboo organisms)

Twelve adapted models, four base models each taught to conceal one of the words
moon, ship and snow. Each gives hints about its word without saying it, and an
explanation counts as on-task when it recovers the word or points at it. No span
is marked in advance, so there is no localization table here.

| model | region | share | on-task |
|---|---|---|---|
| Qwen-7B | assistant | 0.42 | 0.31 |
| Qwen-7B | template | 0.38 | 0.12 |
| Qwen-7B | user | 0.20 | 0.18 |
| Gemma-12B | assistant | 0.56 | 0.30 |
| Gemma-12B | user | 0.28 | 0.20 |
| Gemma-12B | template | 0.16 | 0.33 |
| Gemma-27B | assistant | 0.58 | 0.29 |
| Gemma-27B | user | 0.26 | 0.32 |
| Gemma-27B | template | 0.16 | 0.31 |
| Llama-70B | assistant | 0.44 | 0.17 |
| Llama-70B | template | 0.39 | 0.11 |
| Llama-70B | user | 0.17 | 0.26 |

Signals read from the internal state take the top place on three of the four
models:

| signal | Qwen-7B | Gemma-12B | Gemma-27B | Llama-70B |
|---|---|---|---|---|
| `surprisal` | 0.522 [0.50,0.54]*† | 0.504 [0.48,0.53]* | 0.571 [0.55,0.59]*† | 0.491 [0.47,0.51]* |
| `entropy` | 0.566 [0.55,0.58]*† | 0.557 [0.53,0.58]*† | 0.604 [0.58,0.62]*† | 0.522 [0.51,0.53]*† |
| `varentropy` | 0.551 [0.53,0.57]*† | 0.601 [0.57,0.63]*† | 0.622 [0.60,0.65]*† | 0.535 [0.52,0.55]*† |
| `temporal_kl` | 0.499 [0.48,0.52]* | 0.500 [0.48,0.52]* | 0.402 [0.38,0.42]*† | 0.515 [0.50,0.53]* |
| `resid_jump` | 0.527 [0.50,0.55]*† | 0.579 [0.56,0.60]*† | 0.519 [0.50,0.54]* | 0.549 [0.53,0.57]*† |
| `lookback_ratio` | 0.397 [0.35,0.45]*† | 0.485 [0.44,0.54]* | 0.545 [0.49,0.60]* | 0.545 [0.50,0.59]* |
| `sink_drain` | 0.598 [0.57,0.63]*† | 0.498 [0.46,0.54]* | 0.428 [0.40,0.46]*† | 0.552 [0.53,0.57]*† |
| `head_disagreement` | 0.557 [0.53,0.59]*† | 0.514 [0.48,0.55]* | 0.440 [0.41,0.47]*† | 0.568 [0.54,0.60]*† |
| `w` | 0.699 [0.66,0.74]*† | 0.596 [0.55,0.64]*† | 0.530 [0.48,0.58]* | 0.622 [0.58,0.67]*† |
| `norm_ratio` | 0.381 [0.35,0.41]*† | 0.483 [0.44,0.52]* | 0.416 [0.38,0.45]*† | 0.690 [0.64,0.73]*† |
| `peak_ratio` | 0.767 [0.76,0.78]*† | 0.674 [0.66,0.69]*† | 0.572 [0.56,0.59]*† | 0.681 [0.65,0.72]*† |
| `dominant_mass` | **0.796 [0.78,0.81]*†** | **0.681 [0.67,0.69]*†** | **0.651 [0.62,0.68]*†** | 0.703 [0.67,0.74]*† |
| `resid_jump_nla` | 0.326 [0.31,0.34]*† | 0.374 [0.36,0.39]*† | 0.354 [0.34,0.37]*† | **0.723 [0.70,0.74]*†** |
| _base rate_ | 0.21 | 0.28 | 0.30 | 0.16 |
| _control rand / perm_ | 0.51 / 0.5±0.01 | 0.49 / 0.5±0.01 | 0.51 / 0.5±0.01 | 0.50 / 0.5±0.01 |

`dominant_mass` is the strongest signal on Qwen-7B, Gemma-12B and Gemma-27B,
reaching 0.796 on the first. `resid_jump_nla` leads on Llama-70B at 0.723. `w`
reaches 0.699 on Qwen-7B, its best showing on any benchmark. A model trained to
withhold a word still marks the positions where the word is readable, and the
size and shape of its internal state is what marks them.

### A reply produced by a successful hijack (Tensor Trust)

Players of an online game write attacks that try to make a model reveal an
access code, and defences that try to stop them. We take 776 cases, let the base
model produce its reply, and probe the reply. The important number here is the
base rate, which runs from 0.68 to 0.86: most positions already carry an on-task
explanation, because the whole reply is the model following an instruction.

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
| `w` | 0.608 [0.60,0.61]*† | 0.555 [0.55,0.56]*† | 0.568 [0.56,0.58]*† | 0.586 [0.58,0.59]*† |
| `norm_ratio` | 0.517 [0.51,0.53]*† | **0.719 [0.71,0.73]*†** | 0.671 [0.66,0.68]*† | 0.642 [0.63,0.65]*† |
| `peak_ratio` | 0.443 [0.44,0.45]*† | 0.670 [0.66,0.68]*† | 0.393 [0.38,0.40]*† | 0.507 [0.50,0.51]*† |
| `dominant_mass` | 0.430 [0.42,0.44]*† | 0.671 [0.66,0.68]*† | 0.450 [0.44,0.46]*† | 0.343 [0.33,0.35]*† |
| `resid_jump_nla` | 0.619 [0.61,0.63]*† | 0.707 [0.70,0.72]*† | **0.672 [0.66,0.68]*†** | **0.695 [0.68,0.71]*†** |
| _base rate_ | 0.84 | 0.85 | 0.86 | 0.68 |
| _control rand / perm_ | 0.50 / 0.5±0.00 | 0.50 / 0.5±0.00 | 0.50 / 0.5±0.00 | 0.50 / 0.5±0.00 |

The numbers rise above a coin flip in many cells, and `resid_jump_nla` reaches
0.695 on Llama-70B, but the practical value is small. With 86% of positions
already on-task there is little for a selector to find, and an auditor could pick
positions at random and do nearly as well. This benchmark is best read as the
boundary of where selection helps at all.

### Stated and concealed lies (Liars' Bench)

Responses labelled as honest or dishonest, from the two models that wrote them.
The target is very sparse: only 1% to 2% of positions carry an on-task
explanation, which makes this the hardest of the five.

The explanations do concentrate inside lying transcripts, and the intervals are
tight, though the size of the effect is small:

| model | threat | inside | outside | lift [95% CI] |
|---|---|---|---|---|
| Gemma-27B | deception | 0.02 (n=709480) | 0.00 (n=735253) | **+0.01** [+0.01, +0.01] ✓ |
| Llama-70B | deception | 0.04 (n=303649) | 0.00 (n=290247) | **+0.04** [+0.03, +0.04] ✓ |

| model | region | share | on-task |
|---|---|---|---|
| Gemma-27B | assistant_prior | 0.33 | 0.01 |
| Gemma-27B | user | 0.33 | 0.00 |
| Gemma-27B | assistant | 0.14 | 0.02 |
| Gemma-27B | system | 0.14 | 0.03 |
| Gemma-27B | template | 0.06 | 0.01 |
| Llama-70B | user | 0.44 | 0.00 |
| Llama-70B | assistant_prior | 0.15 | 0.00 |
| Llama-70B | template | 0.15 | 0.01 |
| Llama-70B | system | 0.14 | 0.11 |
| Llama-70B | assistant | 0.12 | 0.03 |

| signal | Gemma-27B | Llama-70B |
|---|---|---|
| `surprisal` | 0.610 [0.61,0.61]*† | 0.548 [0.54,0.55]*† |
| `entropy` | 0.683 [0.68,0.69]*† | 0.609 [0.60,0.62]*† |
| `varentropy` | 0.693 [0.69,0.70]*† | 0.596 [0.59,0.60]*† |
| `temporal_kl` | 0.385 [0.38,0.39]*† | 0.375 [0.37,0.38]*† |
| `resid_jump` | 0.342 [0.33,0.35]*† | 0.501 [0.50,0.51]* |
| `lookback_ratio` | 0.441 [0.43,0.45]*† | 0.478 [0.47,0.49]*† |
| `sink_drain` | 0.354 [0.33,0.37]*† | 0.337 [0.33,0.35]*† |
| `head_disagreement` | **0.280 [0.27,0.29]*†** | **0.318 [0.31,0.33]*†** |
| `w` | 0.590 [0.58,0.60]*† | 0.492 [0.48,0.50]* |
| `norm_ratio` | 0.535 [0.52,0.55]*† | 0.382 [0.37,0.39]*† |
| `peak_ratio` | 0.537 [0.53,0.55]*† | 0.583 [0.57,0.59]*† |
| `dominant_mass` | 0.660 [0.65,0.67]*† | 0.580 [0.57,0.59]*† |
| `resid_jump_nla` | 0.522 [0.52,0.53]*† | 0.436 [0.43,0.45]*† |
| _base rate_ | 0.01 | 0.02 |
| _control rand / perm_ | 0.50 / 0.5±0.00 | 0.50 / 0.5±0.00 |

`head_disagreement` is the strongest signal on both models, at 0.280 and 0.318,
which means it runs backwards: attention heads that **agree** with each other
mark the positions worth explaining here. Uncertainty also helps, with `entropy`
at 0.683 and `varentropy` at 0.693 on Gemma-27B. Even so, a base rate of 0.01
means an auditor is searching for a handful of positions among hundreds of
thousands, and no cheap number rescues that.

### Authored cases, a controlled check

Twenty seven transcripts written by hand, each in two versions, one containing
the thing being audited and one with it removed. The second version is what makes
`kl` possible here, since `kl` measures how far the model's prediction moves when
the suspected cause is deleted. Two situations are covered: an instruction
injected into the input, and a model noticing it is being tested.

These are 27 transcripts, so the intervals are wide and the results should be
read as a check on the method rather than as evidence in their own right.

| model | threat | inside | outside | lift [95% CI] |
|---|---|---|---|---|
| Qwen-7B | eval-awareness | 0.19 (n=98) | 0.07 (n=614) | **+0.12** [+0.05, +0.22] ✓ |
| Qwen-7B | injection | 0.88 (n=129) | 0.91 (n=803) | **-0.02** [-0.11, +0.06] |
| Gemma-12B | eval-awareness | 0.42 (n=99) | 0.11 (n=579) | **+0.32** [+0.21, +0.44] ✓ |
| Gemma-12B | injection | 0.92 (n=130) | 0.93 (n=766) | **-0.01** [-0.09, +0.05] |
| Gemma-27B | eval-awareness | 0.62 (n=99) | 0.15 (n=579) | **+0.47** [+0.36, +0.59] ✓ |
| Gemma-27B | injection | 0.91 (n=130) | 0.95 (n=766) | **-0.04** [-0.11, +0.01] |
| Llama-70B | eval-awareness | 0.61 (n=98) | 0.10 (n=895) | **+0.51** [+0.35, +0.66] ✓ |
| Llama-70B | injection | 0.80 (n=128) | 0.76 (n=1111) | **+0.04** [-0.06, +0.14] |

Explanations concentrate strongly in the marked span when the model is noticing a
test, and the effect grows with model size, from +0.12 on Qwen-7B to +0.51 on
Llama-70B. For injection the lift is flat, matching Tensor Trust: when the whole
reply carries out the instruction, there is nothing to localize.

| signal | Qwen-7B | Gemma-12B | Gemma-27B | Llama-70B |
|---|---|---|---|---|
| `surprisal` | 0.523 [0.46,0.58]* | 0.434 [0.37,0.49]*† | 0.503 [0.43,0.57]* | 0.528 [0.50,0.56]* |
| `entropy` | 0.487 [0.41,0.54]* | 0.440 [0.38,0.49]*† | 0.406 [0.36,0.45]*† | 0.532 [0.47,0.59]* |
| `varentropy` | 0.494 [0.44,0.54]* | 0.426 [0.36,0.47]*† | 0.407 [0.36,0.45]*† | 0.499 [0.44,0.56]* |
| `temporal_kl` | 0.484 [0.45,0.52]* | 0.559 [0.52,0.61]*† | 0.645 [0.60,0.70]*† | 0.488 [0.41,0.57]* |
| `resid_jump` | 0.486 [0.41,0.54]* | 0.461 [0.41,0.53]* | 0.627 [0.58,0.67]*† | 0.368 [0.32,0.42]*† |
| `lookback_ratio` | 0.510 [0.45,0.56]* | 0.457 [0.40,0.50]* | 0.415 [0.37,0.45]*† | 0.326 [0.26,0.38]*† |
| `sink_drain` | 0.615 [0.56,0.69]*† | **0.695 [0.65,0.75]*†** | **0.712 [0.68,0.75]*†** | **0.827 [0.78,0.87]*†** |
| `head_disagreement` | 0.614 [0.56,0.68]*† | 0.672 [0.62,0.74]*† | 0.674 [0.63,0.72]*† | 0.795 [0.75,0.84]*† |
| `w` | 0.570 [0.52,0.63]*† | 0.634 [0.60,0.68]*† | 0.659 [0.62,0.70]*† | 0.787 [0.75,0.83]*† |
| `norm_ratio` | **0.301 [0.22,0.38]*†** | 0.586 [0.55,0.62]*† | 0.593 [0.55,0.65]*† | 0.679 [0.65,0.72]*† |
| `peak_ratio` | 0.373 [0.31,0.45]*† | 0.608 [0.54,0.66]*† | 0.300 [0.27,0.33]*† | 0.513 [0.45,0.58]* |
| `dominant_mass` | 0.401 [0.33,0.49]*† | 0.610 [0.54,0.67]*† | 0.555 [0.50,0.61]* | 0.441 [0.36,0.52]* |
| `resid_jump_nla` | 0.515 [0.45,0.57]* | 0.425 [0.36,0.50]* | 0.463 [0.43,0.50]* | 0.578 [0.53,0.62]*† |
| _base rate_ | 0.09 | 0.15 | 0.22 | 0.15 |
| _control rand / perm_ | 0.46 / 0.5±0.03 | 0.51 / 0.5±0.02 | 0.50 / 0.5±0.02 | 0.51 / 0.5±0.02 |
| signal | Qwen-7B | Gemma-12B | Gemma-27B | Llama-70B |
|---|---|---|---|---|
| `surprisal` | 0.475 [0.41,0.54]* | 0.491 [0.39,0.60]* | 0.557 [0.48,0.69]* | 0.411 [0.38,0.44]*† |
| `entropy` | 0.431 [0.34,0.49]* | 0.498 [0.36,0.60]* | 0.533 [0.47,0.63]* | 0.361 [0.32,0.39]*† |
| `varentropy` | 0.430 [0.34,0.50]* | 0.506 [0.38,0.60]* | 0.527 [0.46,0.62]* | 0.362 [0.33,0.39]*† |
| `temporal_kl` | 0.517 [0.44,0.56]* | 0.473 [0.40,0.58]* | 0.477 [0.38,0.57]* | 0.612 [0.58,0.65]*† |
| `resid_jump` | **0.402 [0.33,0.48]*** | 0.396 [0.30,0.48]* | 0.411 [0.35,0.48]* | 0.405 [0.37,0.44]*† |
| `lookback_ratio` | 0.513 [0.47,0.56]* | 0.569 [0.50,0.63]* | 0.551 [0.47,0.64]* | 0.455 [0.43,0.48]*† |
| `sink_drain` | 0.533 [0.47,0.64]* | 0.579 [0.51,0.67]* | 0.442 [0.36,0.51]* | **0.730 [0.66,0.80]*†** |
| `head_disagreement` | 0.503 [0.44,0.61]*† | 0.593 [0.51,0.69]*† | 0.461 [0.38,0.53]*† | 0.710 [0.65,0.78]*† |
| `w` | 0.541 [0.48,0.63]* | 0.548 [0.46,0.66]* | 0.442 [0.35,0.53]* | 0.724 [0.67,0.79]*† |
| `norm_ratio` | 0.468 [0.35,0.57]* | 0.432 [0.35,0.51]* | **0.606 [0.53,0.71]*†** | 0.395 [0.34,0.44]*† |
| `peak_ratio` | 0.554 [0.48,0.62]* | **0.388 [0.31,0.45]*†** | 0.457 [0.32,0.53]* | 0.562 [0.52,0.60]*† |
| `dominant_mass` | 0.537 [0.46,0.60]* | 0.392 [0.32,0.45]*† | 0.477 [0.40,0.57]* | 0.525 [0.49,0.56]* |
| `resid_jump_nla` | 0.446 [0.37,0.49]*† | 0.437 [0.37,0.50]* | 0.537 [0.45,0.62]* | 0.400 [0.37,0.43]*† |
| _base rate_ | 0.90 | 0.93 | 0.94 | 0.76 |
| _control rand / perm_ | 0.47 / 0.5±0.02 | 0.49 / 0.5±0.03 | 0.53 / 0.5±0.03 | 0.47 / 0.5±0.01 |
| signal | Qwen-7B | Gemma-12B | Gemma-27B | Llama-70B |
|---|---|---|---|---|
| `surprisal` | 0.478 [0.45,0.51]* | 0.435 [0.41,0.46]*† | 0.457 [0.42,0.49]*† | 0.457 [0.43,0.48]*† |
| `entropy` | 0.415 [0.39,0.44]*† | 0.475 [0.44,0.51]* | 0.472 [0.43,0.51]* | 0.430 [0.40,0.46]*† |
| `varentropy` | 0.422 [0.39,0.45]*† | 0.483 [0.45,0.52]* | 0.473 [0.44,0.51]* | 0.424 [0.39,0.45]*† |
| `temporal_kl` | 0.551 [0.52,0.59]*† | 0.463 [0.43,0.50]*† | 0.518 [0.48,0.55]* | 0.543 [0.51,0.58]*† |
| `resid_jump` | 0.481 [0.45,0.51]* | **0.369 [0.34,0.40]*†** | 0.417 [0.38,0.47]*† | 0.419 [0.40,0.44]*† |
| `lookback_ratio` | 0.531 [0.50,0.56]*† | 0.531 [0.51,0.56]*† | 0.509 [0.47,0.54]* | 0.453 [0.42,0.48]*† |
| `sink_drain` | 0.590 [0.56,0.62]*† | 0.599 [0.57,0.63]*† | **0.598 [0.56,0.63]*†** | 0.703 [0.67,0.73]*† |
| `head_disagreement` | 0.593 [0.56,0.62]*† | 0.608 [0.57,0.64]*† | 0.598 [0.56,0.63]*† | **0.704 [0.68,0.73]*†** |
| `w` | 0.527 [0.51,0.54]*† | 0.550 [0.53,0.58]*† | 0.553 [0.52,0.58]*† | 0.661 [0.63,0.70]*† |
| `norm_ratio` | **0.348 [0.31,0.39]*†** | 0.427 [0.38,0.48]*† | 0.486 [0.45,0.53]* | 0.478 [0.43,0.52]* |
| `peak_ratio` | 0.406 [0.36,0.46]*† | 0.513 [0.47,0.55]* | 0.434 [0.40,0.47]*† | 0.542 [0.51,0.57]*† |
| `dominant_mass` | 0.409 [0.37,0.46]*† | 0.518 [0.48,0.56]* | 0.545 [0.51,0.59]*† | 0.508 [0.48,0.54]* |
| `resid_jump_nla` | 0.484 [0.46,0.51]* | 0.442 [0.41,0.47]*† | 0.471 [0.44,0.51]* | 0.463 [0.43,0.49]*† |
| _base rate_ | 0.55 | 0.59 | 0.63 | 0.49 |
| _control rand / perm_ | 0.48 / 0.5±0.01 | 0.50 / 0.5±0.01 | 0.49 / 0.5±0.01 | 0.52 / 0.5±0.01 |

`sink_drain` leads on three of the four models here, reaching 0.827 on
Llama-70B, with `head_disagreement` and `w` close behind. The attention signals
do best on the pilot, and the internal state signals do best on the real
benchmarks, which is worth keeping in mind before drawing conclusions from
either set alone.

## What holds across the benchmarks

**How common the target is decides whether choosing helps at all.** Where
explanations are on-task at a minority of positions, a signal can find them and
keeping the top few is the right approach. The base rates are 0.13 to 0.25 for
injected input, 0.16 to 0.30 for a hidden word, 0.01 to 0.02 for lies, and 0.09
to 0.22 for the authored cases. Where nearly every position is on-task, as in
Tensor Trust at 0.68 to 0.86, there is nothing to choose between. Counting the
base rate takes a few minutes and tells an auditor which situation they are in,
so it is the first thing to measure.

**No signal wins everywhere.** The best signal is `resid_jump_nla` or
`sink_drain` for injected input, `dominant_mass` for a hidden word,
`resid_jump_nla` or `resid_jump` for a hijacked reply, `head_disagreement`
reversed for lies, and `sink_drain` for the authored cases. Any recommendation
has to name the threat.

**Several signals run backwards, and which ones depends on the threat.**
`entropy` sits between 0.366 and 0.452 on injected input, so uncertainty points
away from the attack, which makes sense because a planted instruction is fluent
text. On a hidden word the same signal sits between 0.522 and 0.604, pointing the
right way. `head_disagreement` runs forwards on injected input and backwards on
lies. A signal's direction is a property of the threat rather than of the signal,
so it has to be fixed per threat before use.

**The size and shape of the internal state carries more than expected.** Four
signals here read nothing but the vector the explanation itself is built from:
`act_norm`, `norm_ratio`, `peak_ratio` and `dominant_mass`. They are the best
available signal on the hidden word benchmark for three models out of four, and
competitive elsewhere. They also cost nothing extra, since that vector is already
computed in order to write the explanation.

## Positions whose internal state is extreme

A position whose internal state is far larger than its neighbours is a spike
token, in the sense defined above. They are rare, between 0.3% and 1.8% of all
positions, and they are mostly the first token of a transcript. What happens
there depends on how common the target is:

| benchmark | model | spike positions | share | on-task there | on-task elsewhere |
|---|---|---|---|---|---|
| authored | Gemma-12B | 27 | 1.7% | 0.148 | 0.601 |
| authored | Gemma-27B | 27 | 1.7% | 0.519 | 0.632 |
| authored | Llama-70B | 27 | 1.2% | 0.481 | 0.490 |
| authored | Qwen-7B | 27 | 1.6% | 0.444 | 0.551 |
| Liars' | Gemma-27B | 4333 | 0.3% | 0.000 | 0.011 |
| Liars' | Llama-70B | 2000 | 0.3% | 0.000 | 0.023 |
| OpenPromptInjection | Gemma-12B | 800 | 0.7% | 0.000 | 0.171 |
| OpenPromptInjection | Gemma-27B | 805 | 0.7% | 0.000 | 0.185 |
| OpenPromptInjection | Llama-70B | 800 | 0.6% | 0.009 | 0.130 |
| OpenPromptInjection | Qwen-7B | 800 | 0.7% | 0.000 | 0.256 |
| taboo | Gemma-12B | 96 | 1.8% | 0.000 | 0.281 |
| taboo | Gemma-27B | 96 | 1.7% | 0.000 | 0.306 |
| taboo | Llama-70B | 96 | 1.1% | 0.000 | 0.164 |
| taboo | Qwen-7B | 96 | 1.3% | 0.000 | 0.215 |
| Tensor Trust | Gemma-12B | 1963 | 0.4% | 0.051 | 0.852 |
| Tensor Trust | Gemma-27B | 2073 | 0.4% | 0.998 | 0.863 |
| Tensor Trust | Llama-70B | 1550 | 0.3% | 0.907 | 0.680 |
| Tensor Trust | Qwen-7B | 2303 | 0.4% | 0.231 | 0.839 |

Where the target is sparse, an explanation at a spike token is essentially never
on-task, at 0.000 across OpenPromptInjection, taboo and Liars' Bench. Discarding
those positions before writing any explanation costs nothing and never loses
anything worth having. The saving is small, since they are under 2% of positions,
so this is a tidy result rather than a large one.

Where the target is dense, the pattern does not hold and reverses on two of the
Tensor Trust models, which read 0.998 and 0.907 against their own baselines of
0.863 and 0.680. The rule to take away is that discarding spike tokens is safe
when explanations are scarce and unsafe when they are plentiful.


## Spending a fixed budget

AUROC scores a ranking, and it has no notion of cost. An auditor has a cost: a
number of explanations affordable for one transcript, which we call the
**budget**. The measure that matters at a budget is **precision at a budget**,
meaning the share of the explanations actually bought that turn out to be
on-task. The base rate is again the floor, since spending the budget at random
achieves it. The tables are in `bridge-pooling-all.md` at a budget of one
position in ten, and `bridge-pooling-all-1pct.md` at one in a hundred.

**A smaller budget buys more per explanation.** Because a signal ranks
positions, cutting the budget keeps only the positions it is most confident
about, and those are more often on-task. On injected input, `sink_drain` gives a
precision of 0.291 to 0.332 on the Gemma models at a budget of one position in
ten, and 0.303 to 0.337 at one in a hundred, against base rates of 0.17 and 0.18.
Selection is therefore worth most to an auditor who can afford least, which is
the opposite of what the AUROC tables suggest on their own, since AUROC has no
budget in it.

**Combining two signals does not help.** Given the same budget, the positions
chosen by two signals can be pooled in three ways: take the best from each in
turn, merge the two rankings by score, or keep only positions both rank highly.
None of them beat the better of the two signals on any real benchmark, and
pooling loses outright in most cells. On injected input with Gemma-27B, for
example, `sink_drain` alone gives 0.332 while the pool gives 0.291 and the
agreement of the two gives 0.144.

The reason is worth stating, because the opposite looks true at first. The
positions both signals agree on do have a high precision. That turns out to be an
effect of size rather than of agreement: the agreed positions are few, and any
smaller selection is a stricter one. Giving a single signal a budget matched to
the size of that agreed set reproduces its precision, so the second signal
contributed nothing. The deeper reason is that the strongest signals are not
independent evidence. `sink_drain` and `head_disagreement` both measure how many
attention heads have stopped parking, so they choose overlapping positions.


## Two confounds, measured

Because the pool now contains every position, two effects could produce a
flattering number for reasons that have nothing to do with finding interesting
tokens. Both were measured rather than assumed.

**Template positions.** The chat template scaffolding is where spike tokens
mostly live, and its on-task rate differs from the message content around it. A
signal that merely separated scaffolding from content would collect AUROC for
free. Every table below therefore has a companion computed on message content
alone, with the scaffolding dropped, in the `-content` files. Comparing the two
shows how much of a result is scaffolding.

**Gemma's attention window.** Gemma-3 alternates two kinds of layer, and the
local ones can only see 1024 tokens back. Past that distance those layers cannot
see the first token at all, so they contribute nothing to the measured sink, the
average across layers falls, and `sink_drain` drifts upward with position. The
effect is large and specific:

| benchmark and model | 0 to 256 | 256 to 512 | 512 to 1024 | 1024 to 2048 |
|---|---|---|---|---|
| Tensor Trust, Gemma-12B | -0.582 | -0.522 | -0.422 | **-0.059** |
| Tensor Trust, Gemma-27B | -0.621 | -0.568 | -0.447 | **-0.065** |
| Tensor Trust, Llama-70B | -0.843 | -0.802 | -0.787 | -0.771 |
| Tensor Trust, Qwen-7B | -0.430 | -0.345 | -0.319 | -0.284 |
| Liars', Gemma-27B | -0.630 | -0.559 | -0.314 | **-0.059** |
| Liars', Llama-70B | -0.860 | -0.802 | -0.785 | -0.769 |

The two Gemma models collapse by a factor of about ten past the window, and the
other two barely move. The on-task rate does not follow the same shape, so this
is contamination rather than signal. It affects `sink_drain` on the two Gemma
models in the two benchmarks with long transcripts, which are Tensor Trust and
Liars' Bench. The other three benchmarks have transcripts short enough to sit
inside the window, so they are unaffected. Reading `sink_drain` on Gemma over
long transcripts should use the layers with unrestricted attention only.

## What we recommend

Ordered by what an auditor knows.

- **Measure the base rate first.** Explain a sample of positions, count how many
  come back on-task. Above roughly a half, no signal will help and the budget is
  better spent explaining broadly. Below that, choosing is worth doing.
- **For an instruction hidden in the input, use `resid_jump_nla` on Gemma models
  and `sink_drain` on Llama.** Both need one forward pass. The numbers are 0.635
  to 0.685 and 0.673 respectively.
- **For a word the model is concealing, use `dominant_mass`.** It reaches 0.796,
  0.681 and 0.651 on three of the four models, with `resid_jump_nla` at 0.723 on
  the fourth.
- **For lies, expect little.** `head_disagreement` reversed gives 0.280 and
  0.318, and `varentropy` 0.693 on Gemma-27B, but with a base rate near 0.01 an
  auditor is looking for very few positions among very many. A classifier trained
  for deception is the better tool for this case.
- **Fix the direction of the signal before using it,** by measuring it on a
  handful of labelled transcripts for the threat at hand. Several signals point
  the wrong way on some threats, and using one with the wrong sign is worse than
  choosing at random.
- **Do not combine two signals.** Pooling the positions two signals pick never
  beat the better of the two in any test here. Spending a smaller budget on one
  signal is the better move.

## How far to trust this

- Four of the five benchmarks are external and real. The 27 authored cases are a
  controlled check on the method, with intervals wide enough that they should not
  carry an argument on their own.
- The on-task label is one judge model's answer per position. The controls sit at
  0.5 in every table, and the weak results on lie detection sit alongside the
  benchmark's own trained detectors meeting the same wall, so those weak numbers
  look like a property of the problem rather than a fault in the setup.
- Tensor Trust has almost no room for a selector, since most positions are
  on-task. It is a useful boundary case rather than a failure.
- The signals read from the internal state raise a question this study does not
  settle. They are computed from the same vector the NLA reads, so their strength
  may partly reflect their ability to spot a state the NLA cannot say anything
  useful about, rather than a position where something interesting happens. The
  two readings have different consequences for an auditor, and separating them
  needs an experiment that is not run here.
- Intervals on the large benchmarks are very narrow, because hundreds of
  thousands of positions leave little room for sampling error. Narrow intervals
  say the number is measured precisely. They say nothing about whether the
  judge's notion of on-task is the right target.

## Signals worth trying next

Ranked by how much they would add, with the first two needing no new model
machinery.

- **Restrict the attention signals to the heads that actually park attention.**
  Sink behaviour belongs to individual heads, and averaging across all of them
  mixes the heads carrying the effect with heads that never had it. Labelling the
  heads once per model needs no task data.
- **Count how many heads have stopped parking, rather than averaging attention.**
  If the mechanism is a switch per head, counting switches should be steadier
  than averaging a quantity that one unusual head can pull around.
- **PIShield** (Zou et al. 2025,
  [arXiv:2510.14005](https://arxiv.org/abs/2510.14005)): a trained classifier for
  whether a token sits inside an injected instruction.
- **Deception probe** (Goldowsky-Dill et al. 2025,
  [arXiv:2502.03407](https://arxiv.org/abs/2502.03407)): a direction in the
  internal state for strategic dishonesty, which is the on target tool for the
  two deception benchmarks, though it can be evaded
  ([arXiv:2507.12691](https://arxiv.org/abs/2507.12691)).
- **Semantic entropy probes** (Kossen et al. 2024,
  [arXiv:2406.15927](https://arxiv.org/abs/2406.15927)): uncertainty over
  meanings rather than over individual tokens.

Two ideas were considered and set aside. Sampling methods built on entropy and
varentropy measure quantities already in the tables. Attention traced back
through the layers has no notion of an unusual value without something to compare
against, so it works only when the suspicious span is already known.

## Engineering notes

**Getting full vocabulary distributions from a serving stack.** Several signals
need entropy and related quantities over the whole vocabulary, and serving stacks
return only the top few probabilities. Asking for the hidden states and
multiplying by the output projection reconstructs the full distribution exactly.
The attention signals need the attention maps themselves, which serving stacks do
not return, so those come from a plain teacher forced forward pass.

**Everything resumes.** Each stage writes its output in shards and skips any
shard already present, so an interrupted run continues where it stopped rather
than starting again.

**Where the numbers live.** `paper_results/bridge/all_{benchmark}_{model}.parquet`
holds one row per position with every signal, the region, the judge's answer and
the identifiers needed to group by transcript. Every table in this document can
be recomputed from those files with no model and no GPU.
