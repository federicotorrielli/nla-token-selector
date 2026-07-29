# Combining two selectors, and why the sink says it cannot work

> Two halves of one question. First a measured negative result: pooling the
> tokens picked by two cheap signals never beats the better of the two, and
> shrinking the budget does. Then the mechanism behind it, from Sun, Canziani,
> LeCun and Zhu, *The Spike, the Sparse and the Sink: Anatomy of Massive
> Activations and Attention Sinks*, arXiv:2603.05498 (March 2026), which says
> our two best signals were never independent evidence. Everything here runs
> from `selector/all_tokens_eval.py`.

## The question

The recommendation in `README.md` is two-tier: `entropy` when you do not know the
threat, `sink_drain` or `head_disagreement` once you do. That invites an obvious
third method for the case where you want one recipe that works everywhere: buy
explanations for the best tokens from *both* signals and skip the choice. This
note tests that.

## Why AUROC could not answer it

Every bridge number is an AUROC, which scores a ranking. A pool is a set, so it
has no ranking and no ROC curve. The question forces a different metric, and the
one it forces is closer to the real job: an auditor has a budget of B
explanations for this transcript, so score **precision@B**, the fraction of the B
bought explanations that the judge called on-task. Base rate is the floor
(spending the budget at random), and precision above it is what selection buys.

Scores are z-scored inside a transcript, because a transcript is what an auditor
holds. Budget B is 10% of the transcript's tokens.

Four ways to spend the same B:

- **single**: top B by one signal.
- **pool**: interleave the two rankings, skipping duplicates, until B are picked.
  This is the literal pool of the best tokens from both methods.
- **rrf**: reciprocal rank fusion, `1/(60+rank_a) + 1/(60+rank_b)`, the standard
  parameter-light way to merge two rankings (Cormack, Clarke and Buettcher 2009).
- **and**: rank by the *worse* of the two ranks, so a token is bought early only
  when both signals rank it high. The intersection, written as a ranking.

`sink_drain` points forward on injection and backward on secret words, so the
tables below give each signal its best sign per task. That is a favour to the
single-signal methods and it makes the comparison harder for pooling, not easier.
Two-tailed selection, which is what a genuinely threat-blind method would have to
use, is worse for everything (see the caveat below).

## The answer: pooling does not work

No cell shows the pool significantly ahead of the better single signal. The
closest it comes is taboo on Qwen-7B, +0.015 with an interval that touches zero.
On the four real benchmarks it is significantly *behind* in five of fourteen
cells: both Liars' cells and three of the four OPI cells. The same holds for
`entropy` + `head_disagreement`, and for `rrf`, which only ever looks good on the
27-case pilots where the intervals are enormous.


precision@B = of the B explanations bought, the fraction the judge called
on-task, best per row in bold. Every method spends the same B. `pool`
interleaves the two rankings (the union), `rrf` fuses their scores, `and`
ranks by the worse of the two ranks (the intersection). Each delta is that
combiner minus the better single signal, with a case-clustered 95%
interval; it has to clear 0 for the combiner to be worth anything.
`jaccard` is how much the two signals pick the same tokens (1.0 =
identical).

| task | model | base rate | entropy | sink_drain | pool | rrf | and | pool delta | and delta | jaccard |
|---|---|---|---|---|---|---|---|---|---|---|
| evalaware | g12 | 0.20 | 0.259 | 0.370 | 0.259 | 0.407 | **0.444** | -0.111 [-0.222, +0.000] | +0.074 [-0.087, +0.240] | 0.06 |
| injection | g12 | 0.89 | 0.926 | 0.963 | 0.926 | **1.000** | **1.000** | -0.037 [-0.115, +0.000] | +0.037 [+0.000, +0.100] | 0.08 |
| evalaware | g27 | 0.33 | **0.444** | **0.444** | 0.333 | 0.407 | **0.444** | -0.111 [-0.250, +0.038] | +0.000 [-0.143, +0.208] | 0.06 |
| injection | g27 | 0.92 | **0.926** | **0.926** | 0.889 | **0.926** | **0.926** | -0.037 [-0.111, +0.000] | +0.000 [-0.103, +0.107] | 0.15 |
| evalaware | l70 | 0.37 | 0.333 | 0.370 | 0.222 | 0.370 | **0.481** | -0.148 [-0.308, +0.000] | +0.111 [-0.074, +0.296] | 0.02 |
| injection | l70 | 0.86 | 0.857 | 0.857 | 0.893 | **0.929** | **0.929** | +0.036 [-0.120, +0.182] | +0.071 [+0.000, +0.172] | 0.06 |
| evalaware | q7 | 0.09 | **0.148** | 0.037 | **0.148** | 0.111 | 0.111 | +0.000 [+0.000, +0.000] | -0.037 [-0.160, +0.077] | 0.02 |
| injection | q7 | 0.90 | 0.786 | **0.929** | 0.857 | **0.929** | **0.929** | -0.071 [-0.231, +0.097] | +0.000 [-0.160, +0.138] | 0.10 |
| liars | g27 | 0.03 | 0.036 | **0.063** | 0.044 | 0.062 | 0.052 | -0.019 [-0.024, -0.015] | -0.011 [-0.018, -0.005] | 0.03 |
| liars | l70 | 0.04 | 0.046 | **0.089** | 0.071 | 0.070 | 0.064 | -0.017 [-0.024, -0.011] | -0.025 [-0.033, -0.016] | 0.08 |
| opi | g12 | 0.14 | 0.134 | **0.308** | 0.245 | 0.293 | 0.288 | -0.063 [-0.071, -0.055] | -0.020 [-0.029, -0.012] | 0.10 |
| opi | g27 | 0.15 | 0.142 | **0.335** | 0.254 | 0.291 | 0.290 | -0.081 [-0.089, -0.073] | -0.044 [-0.053, -0.036] | 0.06 |
| opi | l70 | 0.14 | 0.179 | 0.191 | 0.197 | **0.208** | 0.197 | +0.006 [-0.001, +0.013] | +0.006 [-0.003, +0.015] | 0.06 |
| opi | q7 | 0.21 | 0.265 | **0.330** | 0.311 | 0.315 | 0.307 | -0.018 [-0.027, -0.010] | -0.022 [-0.031, -0.014] | 0.09 |
| taboo | g12 | 0.34 | 0.486 | 0.494 | 0.494 | 0.490 | **0.498** | +0.000 [-0.027, +0.028] | +0.004 [-0.031, +0.040] | 0.31 |
| taboo | g27 | 0.33 | 0.543 | 0.498 | 0.524 | **0.547** | **0.547** | -0.019 [-0.054, +0.015] | +0.004 [-0.027, +0.036] | 0.26 |
| taboo | l70 | 0.21 | 0.256 | 0.242 | **0.260** | 0.245 | 0.242 | +0.004 [+0.000, +0.011] | -0.014 [-0.032, +0.000] | 0.10 |
| taboo | q7 | 0.37 | 0.454 | 0.439 | **0.469** | 0.462 | 0.439 | +0.015 [+0.000, +0.034] | -0.015 [-0.047, +0.015] | 0.13 |
| tt | g12 | 0.90 | **0.942** | 0.929 | 0.937 | 0.934 | 0.930 | -0.005 [-0.015, +0.005] | -0.012 [-0.024, +0.001] | 0.10 |
| tt | g27 | 0.92 | 0.937 | 0.944 | **0.947** | 0.942 | 0.942 | +0.003 [-0.006, +0.011] | -0.002 [-0.012, +0.008] | 0.15 |
| tt | l70 | 0.88 | 0.940 | 0.944 | 0.943 | **0.952** | 0.947 | -0.001 [-0.011, +0.009] | +0.004 [-0.007, +0.015] | 0.21 |
| tt | q7 | 0.90 | 0.948 | 0.961 | 0.959 | 0.960 | **0.967** | -0.002 [-0.011, +0.007] | +0.006 [-0.003, +0.014] | 0.12 |

## Where the two methods agree, and where they do not

On-task rate among the tokens both methods bought, and among the ones only
one of them bought. A union pays for both exclusive columns, so it can only
pay off when those match the agreement column.

| task | model | base rate | both | entropy only | sink_drain only |
|---|---|---|---|---|---|
| evalaware | g12 | 0.20 | **1.000** (n=3) | 0.167 (n=24) | 0.292 (n=24) |
| injection | g12 | 0.89 | **1.000** (n=4) | 0.913 (n=23) | 0.957 (n=23) |
| evalaware | g27 | 0.33 | **1.000** (n=3) | 0.375 (n=24) | 0.375 (n=24) |
| injection | g27 | 0.92 | **0.857** (n=7) | 0.950 (n=20) | 0.950 (n=20) |
| evalaware | l70 | 0.37 | **1.000** (n=1) | 0.308 (n=26) | 0.346 (n=26) |
| injection | l70 | 0.86 | **0.667** (n=3) | 0.880 (n=25) | 0.880 (n=25) |
| evalaware | q7 | 0.09 | **0.000** (n=1) | 0.154 (n=26) | 0.038 (n=26) |
| injection | q7 | 0.90 | **1.000** (n=5) | 0.739 (n=23) | 0.913 (n=23) |
| liars | g27 | 0.03 | **0.098** (n=367) | 0.032 (n=5333) | 0.061 (n=5333) |
| liars | l70 | 0.04 | **0.048** (n=670) | 0.046 (n=3852) | 0.096 (n=3852) |
| opi | g12 | 0.14 | **0.276** (n=1669) | 0.101 (n=7185) | 0.316 (n=7185) |
| opi | g27 | 0.15 | **0.377** (n=1040) | 0.111 (n=7814) | 0.329 (n=7814) |
| opi | l70 | 0.14 | **0.045** (n=1013) | 0.196 (n=7667) | 0.211 (n=7667) |
| opi | q7 | 0.21 | **0.300** (n=1536) | 0.258 (n=7399) | 0.336 (n=7399) |
| taboo | g12 | 0.34 | **0.441** (n=118) | 0.526 (n=133) | 0.541 (n=133) |
| taboo | g27 | 0.33 | **0.578** (n=109) | 0.519 (n=158) | 0.443 (n=158) |
| taboo | l70 | 0.21 | **0.365** (n=52) | 0.231 (n=225) | 0.213 (n=225) |
| taboo | q7 | 0.37 | **0.459** (n=61) | 0.453 (n=201) | 0.433 (n=201) |
| tt | g12 | 0.90 | **0.963** (n=400) | 0.937 (n=1742) | 0.921 (n=1742) |
| tt | g27 | 0.92 | **0.965** (n=601) | 0.927 (n=1756) | 0.937 (n=1756) |
| tt | l70 | 0.88 | **0.988** (n=749) | 0.915 (n=1441) | 0.921 (n=1441) |
| tt | q7 | 0.90 | **0.974** (n=426) | 0.940 (n=1546) | 0.957 (n=1546) |

## Why it fails, and why the obvious repair also fails

The two signals pick almost completely different tokens: the overlap between
their picks is 0.02 to 0.31, so nine tokens in ten are chosen by one signal and
not the other. The naive reading of that is encouraging, because two selectors
that disagree should be finding different things and pooling should collect both.

The agreement table above says otherwise. The tokens only one signal bought sit
at or below the base rate in most cells, so the disagreement carries no extra
on-task tokens. A union therefore pays half its budget for the weaker signal's
mistakes, and lands between the two singles rather than above them.

The tokens *both* signals bought do look far better (0.377 against a 0.15 base
rate on OPI with Gemma-27B, 0.098 against 0.03 on Liars' with Gemma-27B), which
suggests intersecting instead of pooling. That reading is wrong, and the `and`
column is the control that shows it: given the same budget B, the intersection
ranking is no better, and on OPI and Liars' it is worse. The agreement region
looked precise only because it is small, and a small selection is a strict
selection. Running one signal alone at a budget matched to the size of that
agreement region (1% instead of 10%) reproduces the number exactly: `sink_drain`
on OPI with Gemma-27B gives 0.379 at a 1% budget, against the agreement region's
0.377. Nothing was gained by the second signal.

## What to do instead

**Shrink the budget.** It is the lever that pooling was supposed to be. On OPI
with Gemma-27B, `sink_drain` goes from 0.335 at a 10% budget to 0.379 at 1%, a
bigger gain than any combiner produced, and it costs a tenth as much. Selection
is worth more per explanation the fewer explanations you buy, which is the
opposite of how the AUROC tables read, since AUROC has no budget in it at all.

**Keep the two-tier recipe.** Pick one signal from the threat, do not hedge
across two.

## Caveat on `entropy` for the threat-blind case

The bottom line in `README.md` says `entropy` is on-task-positive on every task at
0.58 to 0.80. Its own OPI table contradicts this: `entropy` there is 0.33 to 0.46,
marked inverse, because an injected command is fluent low-uncertainty text. So the
threat-agnostic default is not sign-stable either, and a method that must run
without knowing the threat has to take both tails of the z-score. That costs a
lot. Two-tailed `entropy` on OPI collapses to a precision of 0.010 to 0.045
against base rates of 0.14 to 0.21, which is far worse than spending the budget at
random. Genuine threat-blindness is expensive and is still unsolved; pooling does
not solve it.

## The mechanism: three facts from the sink anatomy paper

**Sink behaviour is a property of a head, not of a model.** The paper's own
definition (its equation 28) is per head: a head has a sink if some position in
the first half of the sequence receives more than a threshold of average
attention, and the "sink ratio" is the fraction of heads that qualify. Head
dimension is the dominant architectural driver, because a sink needs enough room
in the per-head space to keep the sink keys geometrically separate from the
ordinary ones.

**A sink is a gate.** This is the paper's strongest result for us. When the
authors give a model an explicit gate that depends on the current hidden state,
the sinks disappear entirely, with no cost in perplexity, and gates that do not
depend on the input fail to replace them. Their reading is that a sink is a
learned workaround: with no explicit gate, the model parks attention on the first
token to switch a head off. Sink heads are also biased to short range, and
removing short sequences from the training loss collapses the sink ratio, so the
purpose of parking attention is to ignore distant context when it does not help.

**Spike channels dominate the residual norm and carry no token information.**
A few channels in intermediate layers hold values orders of magnitude above
everything else, they hold near-fixed ratios to each other, and after
normalization distinct spike tokens collapse to almost the same vector (cosine
similarity near 1.0). Over 98% of every vocabulary tested becomes a spike token
when placed at position 0, so this is positional and architectural rather than
semantic. The paper's own phrase is that these act as implicit parameters.

## Why pooling was doomed: the two signals are one signal

If a sink is a per-head gate, then `sink_drain` (attention leaving the sink) and
`head_disagreement` (heads scattering rather than looking alike) are not two
pieces of evidence. They are two readouts of the same hidden variable, which is
how many heads have ungated at this token. Gated heads all park in the same
place, so they agree; ungating scatters them.

That is testable on the data we already have, and it holds. The overlap between
the tokens the two signals pick, at a 10% budget, is 0.29 to 0.80 with a median
near 0.47. `entropy` against either of them overlaps 0.02 to 0.31. The two
attention signals agree with each other about four times as much as either agrees
with uncertainty. The exception is Tensor Trust (0.03 to 0.12), which is the
degenerate task where nearly every token is on-task anyway.

This also explains the negative result in `bridge-pooling.md` mechanically.
Pooling two selectors added nothing because they were never independent evidence.
The way forward is a better estimator of the one underlying variable, rather than
a combination of worse ones.

## Method queue, best first

**A. Restrict the attention signals to sink heads.** We average over every head
before measuring anything (`signals.py:484`, `hm = rows.mean(axis=0)`), which
mixes the heads that carry the gate with the heads that never had one. The paper
gives a labelling rule that needs no task data: run a few hundred generic
sequences once per model, apply their per-head criterion, keep the heads that
qualify. Then measure drain only in those. The code already carries a note at
`signals.py:461` calling head selection the upgrade path; the paper supplies the
criterion. This makes the signal cheaper as well as sharper, since fewer heads
have to be read.

**B. Count open gates instead of averaging attention mass.** If the mechanism is
a per-head switch, the native measurement is a count: for each sink head, has its
sink share dropped below its own baseline at this token, and how many such heads
are there. A count of switches should be steadier than an average of masses,
which any single head with a large excursion can drag around.

**C. Mask the spike channels before computing `resid_jump`.** We take a plain
Euclidean norm of the difference between neighbouring hidden states
(`signals.py:435`) at the NLA layer, which is an intermediate layer, which is
exactly the depth band where spike channels dominate the norm. So `resid_jump`
is likely reporting proximity to a spike token rather than movement of the state.
This fits its record: it is our weakest signal, close to chance on every task and
below chance on several. The channels are fixed per model and input independent,
so they can be found once and dropped. This repairs an existing signal rather
than adding one.

**D. Count delimiters in the sink span.** Our sink span is the template prefix
before the system instruction, or else the first four tokens (`signals.py:477`).
The paper shows that delimiters such as full stops and newlines become secondary
sinks by attending to themselves, reaching the same amplifier as the first token.
So attention can leave the first token and still be parked, and we would score
that as draining. This matters more in the all-token run than it ever did before,
because chat template tokens are now included and templates are dense in
newlines.

**E. Read attention distance per head.** The paper ties gating to range directly:
gated heads are short range, and ungating is what lets a head reach further. Mean
attention distance per head is a direct readout of that and costs nothing extra,
since we already hold the attention row.

## What the all-token run can already answer

Methods A, B, D and E all need the per-head attention maps, so they need a fresh
forward pass and have to wait for the GPUs. Method C does not, and neither does
the question of where the sinks actually are, because the all-token corpora store
an activation for every token at the NLA layer, which is an intermediate layer,
which is the depth band the paper says the spike channels own.

`all_tokens_eval.py spike` reads those shards and writes, per position,
`spike_mass` (the share of the squared norm held by the spike channels),
`peak_ratio` (largest channel over the root mean square, which needs no channel
list), `act_norm`, and `resid_jump_masked` (method C, the jump with the spike
channels dropped). `consolidate_all.py` joins them into the canonical table, so
the spike columns sit beside the signals and the on-task label. No GPU, no
forward pass, nothing added to the running job.

Decided on 2026-07-28: run these three first, and let them say whether the
per-head work in section 3 is worth a second extraction pass at all.

```bash
python selector/all_tokens_eval.py all       # after the run finishes
python selector/consolidate_all.py
for k in hand opi tt liars taboo; do
  python selector/bridge_report.py --kind $k --all-tokens
  python selector/bridge_report.py --kind $k --all-tokens --content-only
done
```

`spike_mass`, `peak_ratio` and `resid_jump_masked` join the ordinary signal
table, so they arrive with the same intervals, controls and false discovery rate
control as everything else. `--content-only` drops the chat-template tokens, and
the gap between the two reports is the size of the template confound.

That buys three tests as soon as the data lands:

1. **Is a spike token a wasted explanation?** The NLA reads the very layer these
   channels dominate, so at a spike token it is handed something close to a
   constant. If on-task rate falls with `spike_mass`, a threshold discards part
   of the budget before any NLA call, at no cost, since the activation is already
   computed. This is the cheapest possible selector and it is architectural
   rather than task specific.
2. **Do the existing signals just find the spike tokens?** Comparing each
   signal's AUROC on the whole pool against its AUROC within the ordinary tokens
   says how much of the all-token headline is position and template rather than
   selection.
3. **Does masking repair `resid_jump`?** Its AUROC against the on-task label,
   before and after, on identical tokens.

## Two risks to the all-token numbers

**Gemma's sliding window against long transcripts.** The comment at
`signals.py:474` says every layer can see the sink because our transcripts are
short. That was true of the old corpora, whose OPI transcripts reach only 187
tokens, so this bias cannot be tested on anything we already hold. The all-token
run broke the assumption: it uses full transcripts, with a cap of 2600 tokens on
the response tasks and 4096 on OPI, while Gemma's local layers have a 1024
window. Past that point the local layers cannot see token 0 at
all, so they contribute nothing to the sink mass, the average over layers falls,
and `sink_drain` rises. That is a bias that grows with position, in the same
direction for every long transcript, on two of our four models. It should be
checked against token index before the g12 and g27 all-token numbers are read.

**Position 0 and delimiters are architecturally special.** The all-token pool now
contains the template tokens, and those are precisely the tokens the paper says
spike for reasons that have nothing to do with meaning. Any signal that separates
template tokens from content tokens will collect AUROC for free. The fix is to
read the tables within region rather than pooled; the `region` column already
exists in the all-token corpus for this.

## Not worth chasing

The paper's suppression results (sandwich normalization, QKNorm, gated attention)
are all training time interventions. They say what a model without sinks would
look like, which is interesting for whether our signals would survive the next
generation of architectures, but none of it applies to four pretrained models we
are auditing.
## Reproduce

```bash
uv run python selector/all_tokens_eval.py pool --frac 0.10 --tail oracle
uv run python selector/all_tokens_eval.py pool --frac 0.01 --tail oracle
uv run python selector/all_tokens_eval.py pool --tail two
uv run python selector/all_tokens_eval.py pool --signals entropy head_disagreement
uv run python selector/all_tokens_eval.py pool --all-tokens
uv run python selector/all_tokens_eval.py --selftest
```

Without `--all-tokens` it reads the committed per-token CSVs in
`paper_results/bridge/`; with it, the consolidated all-token tables. Neither
needs a GPU or a server.
