# Does pooling two selectors beat either one alone?

> A negative result, and the budget-shaped metric it needed. Pooling the tokens
> picked by two cheap signals never beats the better of the two. Shrinking the
> budget does. Reproduce with `python selector/pool_selector.py`.

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

## Reproduce

```bash
uv run python selector/pool_selector.py --frac 0.10 --tail oracle
uv run python selector/pool_selector.py --frac 0.01 --tail oracle   # budget control
uv run python selector/pool_selector.py --tail two                  # threat-blind
uv run python selector/pool_selector.py --signals entropy head_disagreement
uv run python selector/pool_selector.py --selftest
```

Reads the committed per-token CSVs in `paper_results/bridge/`, so it needs no GPU
and no server.
