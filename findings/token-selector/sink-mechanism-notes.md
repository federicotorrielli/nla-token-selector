# What the sink anatomy paper implies for our signals

> Scouting note on Sun, Canziani, LeCun and Zhu, *The Spike, the Sparse and the
> Sink: Anatomy of Massive Activations and Attention Sinks*, arXiv:2603.05498
> (March 2026). Our best signal, `sink_drain`, measures a thing this paper takes
> apart mechanically. Nothing here is measured yet beyond the one check in
> section 2; the rest is a design queue.

## 1. The three facts that matter to us

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

## 2. First consequence: three of our signals are one signal

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

## 3. Method queue, best first

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

## 3b. What the all-token run can already answer

Methods A, B, D and E all need the per-head attention maps, so they need a fresh
forward pass and have to wait for the GPUs. Method C does not, and neither does
the question of where the sinks actually are, because the all-token corpora store
an activation for every token at the NLA layer, which is an intermediate layer,
which is the depth band the paper says the spike channels own.

`selector/spike_stats.py` reads those shards and writes, per position,
`spike_mass` (the share of the squared norm held by the spike channels),
`peak_ratio` (largest channel over the root mean square, which needs no channel
list), `act_norm`, and `resid_jump_masked` (method C, the jump with the spike
channels dropped). `consolidate_all.py` joins them into the canonical table, so
the spike columns sit beside the signals and the on-task label. No GPU, no
forward pass, nothing added to the running job.

Decided on 2026-07-28: run these three first, and let them say whether the
per-head work in section 3 is worth a second extraction pass at all.

```bash
python selector/spike_stats.py                        # after the run finishes
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

## 4. Two risks to the run now in flight

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

## 5. Not worth chasing

The paper's suppression results (sandwich normalization, QKNorm, gated attention)
are all training time interventions. They say what a model without sinks would
look like, which is interesting for whether our signals would survive the next
generation of architectures, but none of it applies to four pretrained models we
are auditing.
