# Bridge all-token input audit

Generated from the canonical `paper_results/bridge/all_*.parquet` files. The four requested datasets exclude the authored pilot.

**Status:** PASS — 4,705,657 rows across 14 dataset/model combinations.

## Inventory and integrity

| dataset | model | rows | cases | columns | modes | regions | selected non-finite |
|---|---|---|---|---|---|---|---|
| opi | q7 | 111370 | 800 | 24 | injection_input | system,template,user | 0 |
| opi | g12 | 107750 | 800 | 24 | injection_input | system,template,user | 800 |
| opi | g27 | 107750 | 800 | 24 | injection_input | system,template,user | 800 |
| opi | l70 | 126340 | 800 | 24 | injection_input | system,template,user | 0 |
| tt | q7 | 528355 | 1552 | 23 | injection | assistant,system,template,user | 1552 |
| tt | g12 | 543780 | 1544 | 23 | injection | assistant,system,template,user | 0 |
| tt | g27 | 550420 | 1548 | 23 | injection | assistant,system,template,user | 1548 |
| tt | l70 | 564579 | 1550 | 23 | injection | assistant,system,template,user | 1550 |
| liars | g27 | 1444733 | 2000 | 23 | deception | assistant,assistant_prior,system,template,user | 2000 |
| liars | l70 | 593896 | 2000 | 23 | deception | assistant,assistant_prior,system,template,user | 0 |
| taboo | q7 | 7310 | 96 | 23 | taboo | assistant,template,user | 0 |
| taboo | g12 | 5260 | 96 | 23 | taboo | assistant,template,user | 0 |
| taboo | g27 | 5531 | 96 | 23 | taboo | assistant,template,user | 0 |
| taboo | l70 | 8583 | 96 | 23 | taboo | assistant,template,user | 96 |

Ordinary files have 23 columns. OPI has 24 because it additionally stores `attn_rollout`. Every file stores the persisted, transcript-z-scored `w`.

### Schema and dtypes

| dataset | model | schema variant | validation |
|---|---|---|---|
| opi | q7 | OPI + attn_rollout | exact match |
| opi | g12 | OPI + attn_rollout | exact match |
| opi | g27 | OPI + attn_rollout | exact match |
| opi | l70 | OPI + attn_rollout | exact match |
| tt | q7 | ordinary | exact match |
| tt | g12 | ordinary | exact match |
| tt | g27 | ordinary | exact match |
| tt | l70 | ordinary | exact match |
| liars | g27 | ordinary | exact match |
| liars | l70 | ordinary | exact match |
| taboo | q7 | ordinary | exact match |
| taboo | g12 | ordinary | exact match |
| taboo | g27 | ordinary | exact match |
| taboo | l70 | ordinary | exact match |

Ordinary schema: `position_id:Int64, case_id:String, mode:String, tok_idx:Int64, token:String, label:Int64, region:String, probe_tok_idx:Int64, surprisal:Float32, entropy:Float32, varentropy:Float32, temporal_kl:Float32, resid_jump:Float32, lookback_ratio:Float32, sink_drain:Float32, head_disagreement:Float32, on_task:Int64, dominant_mass:Float64, peak_ratio:Float64, act_norm:Float64, norm_ratio:Float64, resid_jump_nla:Float64, w:Float32`.

OPI schema: `position_id:Int64, case_id:String, mode:String, tok_idx:Int64, token:String, label:Int64, region:String, probe_tok_idx:Int64, surprisal:Float32, entropy:Float32, varentropy:Float32, temporal_kl:Float32, resid_jump:Float32, lookback_ratio:Float32, sink_drain:Float32, head_disagreement:Float32, attn_rollout:Float32, on_task:Int64, dominant_mass:Float64, peak_ratio:Float64, act_norm:Float64, norm_ratio:Float64, resid_jump_nla:Float64, w:Float32`.

All nine forward-pass signals, four Q2 activation-derived metrics, `act_norm`, and persisted `w` are present in every file. Position IDs and `(case_id, tok_idx)` are unique; each case is zero-based and contiguous; `token`, `label`, and `on_task` are complete.

### Selected all-token winner availability

| dataset | model | metric | published AUROC | relevant direction | non-finite |
|---|---|---|---|---|---|
| opi | q7 | peak_ratio | 0.239 | lower | 0 |
| opi | g12 | resid_jump_nla | 0.685 | higher | 800 |
| opi | g27 | resid_jump_nla | 0.635 | higher | 800 |
| opi | l70 | sink_drain | 0.673 | higher | 0 |
| tt | q7 | resid_jump | 0.639 | higher | 1552 |
| tt | g12 | norm_ratio | 0.719 | higher | 0 |
| tt | g27 | resid_jump_nla | 0.672 | higher | 1548 |
| tt | l70 | resid_jump_nla | 0.695 | higher | 1550 |
| liars | g27 | varentropy | 0.693 | higher | 2000 |
| liars | l70 | sink_drain | 0.337 | lower | 0 |
| taboo | q7 | dominant_mass | 0.796 | higher | 0 |
| taboo | g12 | dominant_mass | 0.681 | higher | 0 |
| taboo | g27 | dominant_mass | 0.651 | higher | 0 |
| taboo | l70 | resid_jump_nla | 0.723 | higher | 96 |

Per-model pooled-AUROC all-token winners are used.

Below-chance metrics are retained and direction-reversed only in derived aligned scores.

### Persisted `w` validation

| dataset | model | maximum absolute error |
|---|---|---|
| opi | q7 | 0.000 |
| opi | g12 | 0.000 |
| opi | g27 | 0.000 |
| opi | l70 | 0.000 |
| tt | q7 | 0.000 |
| tt | g12 | 0.000 |
| tt | g27 | 0.000 |
| tt | l70 | 0.000 |
| liars | g27 | 0.000 |
| liars | l70 | 0.000 |
| taboo | q7 | 0.000 |
| taboo | g12 | 0.000 |
| taboo | g27 | 0.000 |
| taboo | l70 | 0.000 |

The comparison recomputes `z(sink_drain) - z(lookback_ratio)` over eight deterministically selected complete transcripts per file.

### Null, NaN, and infinite metric values

| dataset | model | metric | null | NaN | +inf | -inf |
|---|---|---|---|---|---|---|
| opi | q7 | surprisal | 0 | 800 | 0 | 0 |
| opi | q7 | entropy | 0 | 800 | 0 | 0 |
| opi | q7 | varentropy | 0 | 800 | 0 | 0 |
| opi | q7 | temporal_kl | 0 | 1600 | 0 | 0 |
| opi | q7 | resid_jump | 0 | 800 | 0 | 0 |
| opi | q7 | lookback_ratio | 0 | 0 | 0 | 0 |
| opi | q7 | sink_drain | 0 | 0 | 0 | 0 |
| opi | q7 | head_disagreement | 0 | 0 | 0 | 0 |
| opi | q7 | w | 0 | 0 | 0 | 0 |
| opi | q7 | act_norm | 0 | 0 | 0 | 0 |
| opi | q7 | norm_ratio | 0 | 0 | 0 | 0 |
| opi | q7 | peak_ratio | 0 | 0 | 0 | 0 |
| opi | q7 | dominant_mass | 0 | 0 | 0 | 0 |
| opi | q7 | resid_jump_nla | 794 | 6 | 0 | 0 |
| opi | q7 | attn_rollout | 0 | 0 | 0 | 0 |
| opi | g12 | surprisal | 0 | 800 | 0 | 0 |
| opi | g12 | entropy | 0 | 800 | 0 | 0 |
| opi | g12 | varentropy | 0 | 800 | 0 | 0 |
| opi | g12 | temporal_kl | 0 | 1600 | 0 | 0 |
| opi | g12 | resid_jump | 0 | 800 | 0 | 0 |
| opi | g12 | lookback_ratio | 0 | 0 | 0 | 0 |
| opi | g12 | sink_drain | 0 | 0 | 0 | 0 |
| opi | g12 | head_disagreement | 0 | 0 | 0 | 0 |
| opi | g12 | w | 0 | 0 | 0 | 0 |
| opi | g12 | act_norm | 0 | 0 | 0 | 0 |
| opi | g12 | norm_ratio | 0 | 0 | 0 | 0 |
| opi | g12 | peak_ratio | 0 | 0 | 0 | 0 |
| opi | g12 | dominant_mass | 0 | 0 | 0 | 0 |
| opi | g12 | resid_jump_nla | 794 | 6 | 0 | 0 |
| opi | g12 | attn_rollout | 0 | 0 | 0 | 0 |
| opi | g27 | surprisal | 0 | 800 | 0 | 0 |
| opi | g27 | entropy | 0 | 800 | 0 | 0 |
| opi | g27 | varentropy | 0 | 800 | 0 | 0 |
| opi | g27 | temporal_kl | 0 | 1600 | 0 | 0 |
| opi | g27 | resid_jump | 0 | 800 | 0 | 0 |
| opi | g27 | lookback_ratio | 0 | 0 | 0 | 0 |
| opi | g27 | sink_drain | 0 | 0 | 0 | 0 |
| opi | g27 | head_disagreement | 0 | 0 | 0 | 0 |
| opi | g27 | w | 0 | 0 | 0 | 0 |
| opi | g27 | act_norm | 0 | 0 | 0 | 0 |
| opi | g27 | norm_ratio | 0 | 0 | 0 | 0 |
| opi | g27 | peak_ratio | 0 | 0 | 0 | 0 |
| opi | g27 | dominant_mass | 0 | 0 | 0 | 0 |
| opi | g27 | resid_jump_nla | 794 | 6 | 0 | 0 |
| opi | g27 | attn_rollout | 0 | 0 | 0 | 0 |
| opi | l70 | surprisal | 0 | 800 | 0 | 0 |
| opi | l70 | entropy | 0 | 800 | 0 | 0 |
| opi | l70 | varentropy | 0 | 800 | 0 | 0 |
| opi | l70 | temporal_kl | 0 | 1600 | 0 | 0 |
| opi | l70 | resid_jump | 0 | 800 | 0 | 0 |
| opi | l70 | lookback_ratio | 0 | 0 | 0 | 0 |
| opi | l70 | sink_drain | 0 | 0 | 0 | 0 |
| opi | l70 | head_disagreement | 0 | 0 | 0 | 0 |
| opi | l70 | w | 0 | 0 | 0 | 0 |
| opi | l70 | act_norm | 0 | 0 | 0 | 0 |
| opi | l70 | norm_ratio | 0 | 0 | 0 | 0 |
| opi | l70 | peak_ratio | 0 | 0 | 0 | 0 |
| opi | l70 | dominant_mass | 0 | 0 | 0 | 0 |
| opi | l70 | resid_jump_nla | 793 | 7 | 0 | 0 |
| opi | l70 | attn_rollout | 0 | 0 | 0 | 0 |
| tt | q7 | surprisal | 0 | 1552 | 0 | 0 |
| tt | q7 | entropy | 0 | 1552 | 0 | 0 |
| tt | q7 | varentropy | 0 | 1552 | 0 | 0 |
| tt | q7 | temporal_kl | 0 | 3104 | 0 | 0 |
| tt | q7 | resid_jump | 0 | 1552 | 0 | 0 |
| tt | q7 | lookback_ratio | 0 | 0 | 0 | 0 |
| tt | q7 | sink_drain | 0 | 0 | 0 | 0 |
| tt | q7 | head_disagreement | 0 | 0 | 0 | 0 |
| tt | q7 | w | 0 | 0 | 0 | 0 |
| tt | q7 | act_norm | 0 | 0 | 0 | 0 |
| tt | q7 | norm_ratio | 0 | 0 | 0 | 0 |
| tt | q7 | peak_ratio | 0 | 0 | 0 | 0 |
| tt | q7 | dominant_mass | 0 | 0 | 0 | 0 |
| tt | q7 | resid_jump_nla | 1525 | 27 | 0 | 0 |
| tt | g12 | surprisal | 0 | 1544 | 0 | 0 |
| tt | g12 | entropy | 0 | 1544 | 0 | 0 |
| tt | g12 | varentropy | 0 | 1544 | 0 | 0 |
| tt | g12 | temporal_kl | 0 | 3088 | 0 | 0 |
| tt | g12 | resid_jump | 0 | 1544 | 0 | 0 |
| tt | g12 | lookback_ratio | 0 | 0 | 0 | 0 |
| tt | g12 | sink_drain | 0 | 0 | 0 | 0 |
| tt | g12 | head_disagreement | 0 | 0 | 0 | 0 |
| tt | g12 | w | 0 | 0 | 0 | 0 |
| tt | g12 | act_norm | 0 | 0 | 0 | 0 |
| tt | g12 | norm_ratio | 0 | 0 | 0 | 0 |
| tt | g12 | peak_ratio | 0 | 0 | 0 | 0 |
| tt | g12 | dominant_mass | 0 | 0 | 0 | 0 |
| tt | g12 | resid_jump_nla | 1517 | 27 | 0 | 0 |
| tt | g27 | surprisal | 0 | 1548 | 0 | 0 |
| tt | g27 | entropy | 0 | 1548 | 0 | 0 |
| tt | g27 | varentropy | 0 | 1548 | 0 | 0 |
| tt | g27 | temporal_kl | 0 | 3096 | 0 | 0 |
| tt | g27 | resid_jump | 0 | 1548 | 0 | 0 |
| tt | g27 | lookback_ratio | 0 | 0 | 0 | 0 |
| tt | g27 | sink_drain | 0 | 0 | 0 | 0 |
| tt | g27 | head_disagreement | 0 | 0 | 0 | 0 |
| tt | g27 | w | 0 | 0 | 0 | 0 |
| tt | g27 | act_norm | 0 | 0 | 0 | 0 |
| tt | g27 | norm_ratio | 0 | 0 | 0 | 0 |
| tt | g27 | peak_ratio | 0 | 0 | 0 | 0 |
| tt | g27 | dominant_mass | 0 | 0 | 0 | 0 |
| tt | g27 | resid_jump_nla | 1520 | 28 | 0 | 0 |
| tt | l70 | surprisal | 0 | 1550 | 0 | 0 |
| tt | l70 | entropy | 0 | 1550 | 0 | 0 |
| tt | l70 | varentropy | 0 | 1550 | 0 | 0 |
| tt | l70 | temporal_kl | 0 | 3100 | 0 | 0 |
| tt | l70 | resid_jump | 0 | 1550 | 0 | 0 |
| tt | l70 | lookback_ratio | 0 | 0 | 0 | 0 |
| tt | l70 | sink_drain | 0 | 0 | 0 | 0 |
| tt | l70 | head_disagreement | 0 | 0 | 0 | 0 |
| tt | l70 | w | 0 | 0 | 0 | 0 |
| tt | l70 | act_norm | 0 | 0 | 0 | 0 |
| tt | l70 | norm_ratio | 0 | 0 | 0 | 0 |
| tt | l70 | peak_ratio | 0 | 0 | 0 | 0 |
| tt | l70 | dominant_mass | 0 | 0 | 0 | 0 |
| tt | l70 | resid_jump_nla | 1522 | 28 | 0 | 0 |
| liars | g27 | surprisal | 0 | 2000 | 0 | 0 |
| liars | g27 | entropy | 0 | 2000 | 0 | 0 |
| liars | g27 | varentropy | 0 | 2000 | 0 | 0 |
| liars | g27 | temporal_kl | 0 | 4000 | 0 | 0 |
| liars | g27 | resid_jump | 0 | 2000 | 0 | 0 |
| liars | g27 | lookback_ratio | 0 | 0 | 0 | 0 |
| liars | g27 | sink_drain | 0 | 0 | 0 | 0 |
| liars | g27 | head_disagreement | 0 | 0 | 0 | 0 |
| liars | g27 | w | 0 | 0 | 0 | 0 |
| liars | g27 | act_norm | 0 | 0 | 0 | 0 |
| liars | g27 | norm_ratio | 0 | 0 | 0 | 0 |
| liars | g27 | peak_ratio | 0 | 0 | 0 | 0 |
| liars | g27 | dominant_mass | 0 | 0 | 0 | 0 |
| liars | g27 | resid_jump_nla | 1930 | 70 | 0 | 0 |
| liars | l70 | surprisal | 0 | 2000 | 0 | 0 |
| liars | l70 | entropy | 0 | 2000 | 0 | 0 |
| liars | l70 | varentropy | 0 | 2000 | 0 | 0 |
| liars | l70 | temporal_kl | 0 | 4000 | 0 | 0 |
| liars | l70 | resid_jump | 0 | 2000 | 0 | 0 |
| liars | l70 | lookback_ratio | 0 | 0 | 0 | 0 |
| liars | l70 | sink_drain | 0 | 0 | 0 | 0 |
| liars | l70 | head_disagreement | 0 | 0 | 0 | 0 |
| liars | l70 | w | 0 | 0 | 0 | 0 |
| liars | l70 | act_norm | 0 | 0 | 0 | 0 |
| liars | l70 | norm_ratio | 0 | 0 | 0 | 0 |
| liars | l70 | peak_ratio | 0 | 0 | 0 | 0 |
| liars | l70 | dominant_mass | 0 | 0 | 0 | 0 |
| liars | l70 | resid_jump_nla | 1970 | 30 | 0 | 0 |
| taboo | q7 | surprisal | 0 | 96 | 0 | 0 |
| taboo | q7 | entropy | 0 | 96 | 0 | 0 |
| taboo | q7 | varentropy | 0 | 96 | 0 | 0 |
| taboo | q7 | temporal_kl | 0 | 192 | 0 | 0 |
| taboo | q7 | resid_jump | 0 | 96 | 0 | 0 |
| taboo | q7 | lookback_ratio | 0 | 0 | 0 | 0 |
| taboo | q7 | sink_drain | 0 | 0 | 0 | 0 |
| taboo | q7 | head_disagreement | 0 | 0 | 0 | 0 |
| taboo | q7 | w | 0 | 0 | 0 | 0 |
| taboo | q7 | act_norm | 0 | 0 | 0 | 0 |
| taboo | q7 | norm_ratio | 0 | 0 | 0 | 0 |
| taboo | q7 | peak_ratio | 0 | 0 | 0 | 0 |
| taboo | q7 | dominant_mass | 0 | 0 | 0 | 0 |
| taboo | q7 | resid_jump_nla | 95 | 1 | 0 | 0 |
| taboo | g12 | surprisal | 0 | 96 | 0 | 0 |
| taboo | g12 | entropy | 0 | 96 | 0 | 0 |
| taboo | g12 | varentropy | 0 | 96 | 0 | 0 |
| taboo | g12 | temporal_kl | 0 | 192 | 0 | 0 |
| taboo | g12 | resid_jump | 0 | 96 | 0 | 0 |
| taboo | g12 | lookback_ratio | 0 | 0 | 0 | 0 |
| taboo | g12 | sink_drain | 0 | 0 | 0 | 0 |
| taboo | g12 | head_disagreement | 0 | 0 | 0 | 0 |
| taboo | g12 | w | 0 | 0 | 0 | 0 |
| taboo | g12 | act_norm | 0 | 0 | 0 | 0 |
| taboo | g12 | norm_ratio | 0 | 0 | 0 | 0 |
| taboo | g12 | peak_ratio | 0 | 0 | 0 | 0 |
| taboo | g12 | dominant_mass | 0 | 0 | 0 | 0 |
| taboo | g12 | resid_jump_nla | 95 | 1 | 0 | 0 |
| taboo | g27 | surprisal | 0 | 96 | 0 | 0 |
| taboo | g27 | entropy | 0 | 96 | 0 | 0 |
| taboo | g27 | varentropy | 0 | 96 | 0 | 0 |
| taboo | g27 | temporal_kl | 0 | 192 | 0 | 0 |
| taboo | g27 | resid_jump | 0 | 96 | 0 | 0 |
| taboo | g27 | lookback_ratio | 0 | 0 | 0 | 0 |
| taboo | g27 | sink_drain | 0 | 0 | 0 | 0 |
| taboo | g27 | head_disagreement | 0 | 0 | 0 | 0 |
| taboo | g27 | w | 0 | 0 | 0 | 0 |
| taboo | g27 | act_norm | 0 | 0 | 0 | 0 |
| taboo | g27 | norm_ratio | 0 | 0 | 0 | 0 |
| taboo | g27 | peak_ratio | 0 | 0 | 0 | 0 |
| taboo | g27 | dominant_mass | 0 | 0 | 0 | 0 |
| taboo | g27 | resid_jump_nla | 95 | 1 | 0 | 0 |
| taboo | l70 | surprisal | 0 | 96 | 0 | 0 |
| taboo | l70 | entropy | 0 | 96 | 0 | 0 |
| taboo | l70 | varentropy | 0 | 96 | 0 | 0 |
| taboo | l70 | temporal_kl | 0 | 192 | 0 | 0 |
| taboo | l70 | resid_jump | 0 | 96 | 0 | 0 |
| taboo | l70 | lookback_ratio | 0 | 0 | 0 | 0 |
| taboo | l70 | sink_drain | 0 | 0 | 0 | 0 |
| taboo | l70 | head_disagreement | 0 | 0 | 0 | 0 |
| taboo | l70 | w | 0 | 0 | 0 | 0 |
| taboo | l70 | act_norm | 0 | 0 | 0 | 0 |
| taboo | l70 | norm_ratio | 0 | 0 | 0 | 0 |
| taboo | l70 | peak_ratio | 0 | 0 | 0 | 0 |
| taboo | l70 | dominant_mass | 0 | 0 | 0 | 0 |
| taboo | l70 | resid_jump_nla | 95 | 1 | 0 | 0 |

Difference signals are expected to be unavailable at the first token because there is no predecessor. Source values are preserved; downstream derived scores remain null.

## Evaluation and base-rate differences

| dataset | available sequence | label meaning | judge target | interpretation |
|---|---|---|---|---|
| OPI | input plus generation boundary; no response | injected span | embedded instruction | sparse selector task |
| Tensor Trust | prompt, boundary, response | attack/access-code case | following input steering | dense boundary case |
| Liars | full on-policy transcript | deceptive transcript | lying/concealment | extremely sparse |
| taboo | prompt, boundary, response | no negative span | secret word/concept | sparse selector task |

| dataset | model | pooled rate | case q25 | case median | case q75 | regime |
|---|---|---|---|---|---|---|
| opi | q7 | 0.254 | 0.222 | 0.254 | 0.289 | sparse |
| opi | g12 | 0.170 | 0.144 | 0.170 | 0.198 | sparse |
| opi | g27 | 0.184 | 0.162 | 0.180 | 0.210 | sparse |
| opi | l70 | 0.129 | 0.110 | 0.129 | 0.149 | sparse |
| tt | q7 | 0.837 | 0.818 | 0.887 | 0.926 | dense boundary |
| tt | g12 | 0.849 | 0.864 | 0.919 | 0.947 | dense boundary |
| tt | g27 | 0.864 | 0.857 | 0.922 | 0.952 | dense boundary |
| tt | l70 | 0.681 | 0.680 | 0.789 | 0.844 | dense boundary |
| liars | g27 | 0.011 | 0.001 | 0.005 | 0.104 | sparse |
| liars | l70 | 0.023 | 0.000 | 0.006 | 0.100 | sparse |
| taboo | q7 | 0.212 | 0.073 | 0.201 | 0.381 | sparse |
| taboo | g12 | 0.276 | 0.136 | 0.233 | 0.508 | sparse |
| taboo | g27 | 0.300 | 0.120 | 0.270 | 0.552 | sparse |
| taboo | l70 | 0.162 | 0.088 | 0.115 | 0.317 | sparse |

All files use the same deterministic judge model but different threat-specific questions. Tensor Trust remains measurable but has little practical room for selection because most positions are already on-task.

Model NLA layers/dimensions: q7 layer 20/3584; g12 layer 32/3840; g27 layer 41/5376; l70 layer 53/8192.

## Cross-model case coverage

| dataset | union | intersection | intersection/union | per-model cases |
|---|---|---|---|---|
| opi | 800 | 800 | 1.000 | q7:800, g12:800, g27:800, l70:800 |
| tt | 1552 | 1543 | 0.994 | q7:1552, g12:1544, g27:1548, l70:1550 |
| liars | 4000 | 0 | 0.000 | g27:2000, l70:2000 |
| taboo | 96 | 96 | 1.000 | q7:96, g12:96, g27:96, l70:96 |

Liars is disjoint on-policy data: the loader first filters each benchmark subset by generator model and only then samples, so each probed model sees transcripts that same model generated. Gemma-27B uses instructed-deception, insider-trading, and convincing-game; Llama-70B uses those three plus harm-pressure-choice and harm-pressure-knowledge-report. The two 2,000-case sets have zero shared IDs, so comparisons are distributional rather than paired.

### Liars subdataset and class composition

| model | subdataset | label | cases |
|---|---|---|---|
| g27 | convincing-game | 0 | 64 |
| g27 | insider-trading | 0 | 412 |
| g27 | instructed-deception | 0 | 524 |
| g27 | convincing-game | 1 | 69 |
| g27 | insider-trading | 1 | 364 |
| g27 | instructed-deception | 1 | 567 |
| l70 | convincing-game | 0 | 18 |
| l70 | harm-pressure-choice | 0 | 271 |
| l70 | harm-pressure-knowledge-report | 0 | 299 |
| l70 | insider-trading | 0 | 71 |
| l70 | instructed-deception | 0 | 341 |
| l70 | convincing-game | 1 | 41 |
| l70 | harm-pressure-choice | 1 | 32 |
| l70 | harm-pressure-knowledge-report | 1 | 37 |
| l70 | insider-trading | 1 | 113 |
| l70 | instructed-deception | 1 | 777 |

### Tensor Trust cases not present for all four models

| case_id | models present |
|---|---|
| 111639085031708_attack | q7 |
| 136365211771134_attack | q7, g12, g27 |
| 169217416594394_access_code | q7, l70 |
| 189021510777565_access_code | q7, l70 |
| 27900107631126_access_code | q7, l70 |
| 44809393836593_access_code | q7, g27, l70 |
| 8731668513451_access_code | q7, g27, l70 |
| 8731668513451_attack | q7, g27, l70 |
| 9010841403965_attack | q7, g27, l70 |

Published parquets do not retain exact skip reasons. Possible extraction paths include an unavailable original response, tokenizer round-trip rejection, span reconstruction failure, or a scoring exception.

## Tokenization and sequence structure

| dataset | model | full median | input median | boundary median | output median | trailer median | boundary example |
|---|---|---|---|---|---|---|---|
| opi | q7 | 140.000 | 135.000 | 5.000 | 0.000 | 0.000 | <\|im_end\|> Ċ <\|im_start\|> assistant Ċ |
| opi | g12 | 136.000 | 131.000 | 5.000 | 0.000 | 0.000 | <end_of_turn> \n <start_of_turn> model \n |
| opi | g27 | 136.000 | 131.000 | 5.000 | 0.000 | 0.000 | <end_of_turn> \n <start_of_turn> model \n |
| opi | l70 | 159.000 | 154.000 | 5.000 | 0.000 | 0.000 | <\|eot_id\|> <\|start_header_id\|> assistant <\|end_header_id\|> ĊĊ |
| tt | q7 | 253.500 | 240.000 | 5.000 | 2.000 | 0.000 | <\|im_end\|> Ċ <\|im_start\|> assistant Ċ |
| tt | g12 | 261.000 | 246.500 | 5.000 | 3.000 | 0.000 | <end_of_turn> \n <start_of_turn> model \n |
| tt | g27 | 262.500 | 246.000 | 5.000 | 3.000 | 0.000 | <end_of_turn> \n <start_of_turn> model \n |
| tt | l70 | 279.000 | 260.000 | 5.000 | 2.000 | 0.000 | <\|eot_id\|> <\|start_header_id\|> assistant <\|end_header_id\|> ĊĊ |
| liars | g27 | 120.000 | 73.000 | 5.000 | 44.000 | 2.000 | <end_of_turn> \n <start_of_turn> model |
| liars | l70 | 145.000 | 100.000 | 5.000 | 26.000 | 1.000 | <\|eot_id\|> <\|start_header_id\|> assistant <\|end_header_id\|> ĊĊ |
| taboo | q7 | 75.000 | 39.500 | 5.000 | 29.000 | 0.000 | <\|im_end\|> Ċ <\|im_start\|> assistant Ċ |
| taboo | g12 | 58.500 | 19.500 | 5.000 | 34.000 | 0.000 | <end_of_turn> \n <start_of_turn> model \n |
| taboo | g27 | 63.500 | 19.500 | 5.000 | 39.000 | 0.000 | <end_of_turn> \n <start_of_turn> model \n |
| taboo | l70 | 97.000 | 45.500 | 5.000 | 46.000 | 0.000 | <\|eot_id\|> <\|start_header_id\|> assistant <\|end_header_id\|> ĊĊ |

### Derived boundary corrections

| dataset | model | case_id | source boundary | analysis boundary | relabeled content tokens |
|---|---|---|---|---|---|
| tt | q7 | 170140834572836_access_code | 11 | 5 | 1 |
| tt | l70 | 170140834572836_access_code | 11 | 5 | 1 |

The canonical source `region` values are preserved as `source_region`. For these TT cases, single-character user content was left as `template` by substring/offset span matching, causing the final user turn to merge with the assistant-generation boundary. The derived table retains the source length, assigns the final five template tokens to `boundary`, moves the preceding user turn to `input`, and records the corrected content role in `analysis_region`.

| dataset | model | `Ġ` | `Ċ` | `▁` | `âĢ…` | leading `<` |
|---|---|---|---|---|---|---|
| opi | q7 | 71510 | 4510 | 0 | 10 | 4000 |
| opi | g12 | 0 | 0 | 70870 | 0 | 3200 |
| opi | g27 | 0 | 0 | 70870 | 0 | 3200 |
| opi | l70 | 77510 | 4510 | 0 | 10 | 7200 |
| tt | q7 | 367612 | 25090 | 0 | 1558 | 8709 |
| tt | g12 | 0 | 0 | 365721 | 0 | 7270 |
| tt | g27 | 0 | 0 | 368782 | 0 | 7332 |
| tt | l70 | 383100 | 24654 | 0 | 1548 | 14398 |
| liars | g27 | 0 | 0 | 830131 | 0 | 35844 |
| liars | l70 | 337187 | 29741 | 0 | 15 | 29650 |
| taboo | q7 | 4635 | 483 | 0 | 30 | 480 |
| taboo | g12 | 0 | 0 | 3331 | 0 | 420 |
| taboo | g27 | 0 | 0 | 3530 | 0 | 432 |
| taboo | l70 | 4945 | 480 | 0 | 20 | 864 |

The strings come directly from `tokenizer.convert_ids_to_tokens(ids)`. `Ġ` and `Ċ` are byte-level BPE whitespace/newline markers; `▁` is SentencePiece metaspace; forms such as `âĢĿ` are byte-to-Unicode renderings of UTF-8 bytes; angle-bracket values are model-specific chat-template special tokens. They are preserved verbatim as `token_raw`, not lossily decoded.

Shared case IDs do not imply shared token positions. Model comparisons should use segment-normalized positions or model-specific bins.

## Response provenance

| dataset | model | output tokens | reconstructed cases | reconstructed tokens |
|---|---|---|---|---|
| opi | q7 | 0 | 0 | 0 |
| opi | g12 | 0 | 0 | 0 |
| opi | g27 | 0 | 0 | 0 |
| opi | l70 | 0 | 0 | 0 |
| tt | q7 | 13607 | 142 | 3853 |
| tt | g12 | 18596 | 212 | 6376 |
| tt | g27 | 24532 | 316 | 9153 |
| tt | l70 | 19557 | 245 | 7000 |
| liars | g27 | 208381 | 0 | 0 |
| liars | l70 | 70616 | 0 | 0 |
| taboo | q7 | 3062 | 45 | 679 |
| taboo | g12 | 2932 | 55 | 625 |
| taboo | g27 | 3203 | 60 | 720 |
| taboo | l70 | 3759 | 74 | 1078 |

TT/taboo stored at most 30 historical response tokens. A deterministic continuation reconstructed from a stored prefix means replaying that exact prefix and greedily generating the missing suffix (no sampling), stopping at a special token or the original 64-token TT / 48-token taboo budget. It recovers a reproducible completion, not necessarily history: for temperature-sampled taboo cases the unavailable historical sampled tail cannot be reconstructed.

## Template sensitivity and fixed budgets

| dataset | model | all winner | content winner | all base | content base | all AUROC | content AUROC | winner changes | top 1% precision | top 1% template | top 10% precision | top 10% template |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| opi | q7 | peak_ratio | peak_ratio | 0.254 | 0.248 | 0.239 | 0.221 | False | 0.534 | 0.000 | 0.520 | 0.000 |
| opi | g12 | resid_jump_nla | resid_jump_nla | 0.170 | 0.166 | 0.685 | 0.693 | False | 0.006 | 1.000 | 0.286 | 0.510 |
| opi | g27 | resid_jump_nla | peak_ratio | 0.184 | 0.174 | 0.635 | 0.626 | True | 0.258 | 0.995 | 0.284 | 0.361 |
| opi | l70 | sink_drain | peak_ratio | 0.129 | 0.161 | 0.673 | 0.605 | True | 0.249 | 0.189 | 0.183 | 0.171 |
| tt | q7 | resid_jump | resid_jump | 0.837 | 0.836 | 0.639 | 0.648 | False | 0.921 | 0.006 | 0.901 | 0.023 |
| tt | g12 | norm_ratio | norm_ratio | 0.849 | 0.850 | 0.719 | 0.740 | False | 0.653 | 0.426 | 0.891 | 0.067 |
| tt | g27 | resid_jump_nla | norm_ratio | 0.864 | 0.860 | 0.672 | 0.666 | True | 0.963 | 0.572 | 0.925 | 0.134 |
| tt | l70 | resid_jump_nla | resid_jump_nla | 0.681 | 0.687 | 0.695 | 0.729 | False | 0.486 | 0.692 | 0.721 | 0.200 |
| liars | g27 | varentropy | varentropy | 0.011 | 0.011 | 0.693 | 0.711 | False | 0.004 | 0.106 | 0.014 | 0.035 |
| liars | l70 | sink_drain | sink_drain | 0.023 | 0.026 | 0.337 | 0.270 | False | 0.000 | 0.934 | 0.000 | 0.494 |
| taboo | q7 | dominant_mass | dominant_mass | 0.212 | 0.266 | 0.796 | 0.760 | False | 0.000 | 1.000 | 0.513 | 0.360 |
| taboo | g12 | dominant_mass | dominant_mass | 0.276 | 0.266 | 0.681 | 0.696 | False | 0.000 | 1.000 | 0.284 | 0.377 |
| taboo | g27 | dominant_mass | dominant_mass | 0.300 | 0.299 | 0.651 | 0.715 | False | 0.000 | 1.000 | 0.289 | 0.319 |
| taboo | l70 | resid_jump_nla | norm_ratio | 0.162 | 0.193 | 0.723 | 0.711 | True | 0.000 | 1.000 | 0.195 | 0.376 |

Content-only means `region != "template"`, matching `bridge_report.py`. Budget rows rank the selected all-token winner independently within each case. No signal pooling is performed.
### Selected-score quantiles

Columns are the 1st, 25th, 50th, 75th, and 99th percentiles. Aligned scores are sign-flipped when lower raw values are relevant.

| dataset | model | raw p01 | raw p25 | raw p50 | raw p75 | raw p99 | aligned p01 | aligned p25 | aligned p50 | aligned p75 | aligned p99 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| opi | q7 | 9.614 | 13.891 | 17.643 | 23.055 | 34.719 | -34.719 | -23.055 | -17.643 | -13.891 | -9.614 |
| opi | g12 | 6354.973 | 10014.812 | 11350.854 | 12826.494 | 35585.634 | 6354.973 | 10014.812 | 11350.854 | 12826.494 | 35585.634 |
| opi | g27 | 7248.672 | 11020.698 | 12431.006 | 13813.486 | 32465.031 | 7248.672 | 11020.698 | 12431.006 | 13813.486 | 32465.031 |
| opi | l70 | -1.000 | -0.891 | -0.864 | -0.834 | -0.785 | -1.000 | -0.891 | -0.864 | -0.834 | -0.785 |
| tt | q7 | 22.320 | 205.508 | 277.516 | 326.041 | 467.006 | 22.320 | 205.508 | 277.516 | 326.041 | 467.006 |
| tt | g12 | 0.608 | 0.789 | 1.000 | 1.192 | 1.562 | 0.608 | 0.789 | 1.000 | 1.192 | 1.562 |
| tt | g27 | 2438.179 | 9742.938 | 11920.100 | 13671.283 | 20260.879 | 2438.179 | 9742.938 | 11920.100 | 13671.283 | 20260.879 |
| tt | l70 | 1.762 | 18.658 | 21.488 | 23.800 | 31.195 | 1.762 | 18.658 | 21.488 | 23.800 | 31.195 |
| liars | g27 | 0.000 | 0.004 | 0.461 | 1.320 | 4.705 | 0.000 | 0.004 | 0.461 | 1.320 | 4.705 |
| liars | l70 | -1.000 | -0.862 | -0.824 | -0.785 | -0.735 | 0.735 | 0.785 | 0.824 | 0.862 | 1.000 |
| taboo | q7 | 0.128 | 0.234 | 0.312 | 0.369 | 0.888 | 0.128 | 0.234 | 0.312 | 0.369 | 0.888 |
| taboo | g12 | 0.010 | 0.966 | 0.971 | 0.975 | 1.000 | 0.010 | 0.966 | 0.971 | 0.975 | 1.000 |
| taboo | g27 | 0.004 | 0.931 | 0.942 | 0.952 | 1.000 | 0.004 | 0.931 | 0.942 | 0.952 | 1.000 |
| taboo | l70 | 11.557 | 21.789 | 25.237 | 29.080 | 791.640 | 11.557 | 21.789 | 25.237 | 29.080 | 791.640 |

### Direction-aligned selected score by derived segment

| dataset | model | segment | tokens | mean | std |
|---|---|---|---|---|---|
| opi | q7 | boundary | 4000 | -19.887 | 3.600 |
| opi | q7 | input | 107370 | -18.832 | 6.372 |
| opi | g12 | boundary | 4000 | 16148.503 | 3487.402 |
| opi | g12 | input | 103750 | 11718.133 | 3663.769 |
| opi | g27 | boundary | 4000 | 15712.572 | 3432.764 |
| opi | g27 | input | 103750 | 12651.486 | 3369.891 |
| opi | l70 | boundary | 4000 | -0.816 | 0.024 |
| opi | l70 | input | 122340 | -0.867 | 0.043 |
| tt | q7 | boundary | 7760 | 293.505 | 54.203 |
| tt | q7 | input | 506988 | 254.436 | 109.841 |
| tt | q7 | output | 13607 | 303.866 | 70.792 |
| tt | g12 | boundary | 7720 | 1.004 | 0.338 |
| tt | g12 | input | 517464 | 1.036 | 0.687 |
| tt | g12 | output | 18596 | 1.057 | 0.358 |
| tt | g27 | boundary | 7740 | 15505.646 | 3378.730 |
| tt | g27 | input | 518148 | 11474.979 | 3922.125 |
| tt | g27 | output | 24532 | 11395.121 | 2956.330 |
| tt | l70 | boundary | 7750 | 21.831 | 3.036 |
| tt | l70 | input | 537272 | 22.443 | 41.709 |
| tt | l70 | output | 19557 | 21.569 | 3.406 |
| liars | g27 | boundary | 9144 | 0.460 | 0.624 |
| liars | g27 | input | 1223208 | 0.950 | 1.475 |
| liars | g27 | output | 208381 | 0.552 | 0.689 |
| liars | g27 | trailer | 4000 | 0.178 | 0.234 |
| liars | l70 | boundary | 10000 | 0.818 | 0.029 |
| liars | l70 | input | 511280 | 0.829 | 0.055 |
| liars | l70 | output | 70616 | 0.810 | 0.039 |
| liars | l70 | trailer | 2000 | 0.827 | 0.013 |
| taboo | q7 | boundary | 480 | 0.369 | 0.100 |
| taboo | q7 | input | 3768 | 0.309 | 0.131 |
| taboo | q7 | output | 3062 | 0.306 | 0.080 |
| taboo | g12 | boundary | 480 | 0.953 | 0.027 |
| taboo | g12 | input | 1848 | 0.923 | 0.214 |
| taboo | g12 | output | 2932 | 0.969 | 0.011 |
| taboo | g27 | boundary | 480 | 0.904 | 0.037 |
| taboo | g27 | input | 1848 | 0.892 | 0.209 |
| taboo | g27 | output | 3203 | 0.943 | 0.017 |
| taboo | l70 | boundary | 480 | 28.528 | 3.333 |
| taboo | l70 | input | 4344 | 40.860 | 114.425 |
| taboo | l70 | output | 3759 | 26.968 | 5.211 |

### Direction-aligned selected score by raw repository region

| dataset | model | source region | tokens | mean | std |
|---|---|---|---|---|---|
| opi | q7 | system | 15200 | -16.687 | 4.420 |
| opi | q7 | template | 10400 | -21.385 | 7.997 |
| opi | q7 | user | 85770 | -18.951 | 6.217 |
| opi | g12 | system | 14800 | 12023.732 | 2081.207 |
| opi | g12 | template | 8000 | 20361.838 | 8731.720 |
| opi | g12 | user | 84950 | 11140.899 | 1966.340 |
| opi | g27 | system | 14800 | 12900.809 | 1517.986 |
| opi | g27 | template | 8000 | 19343.171 | 8157.411 |
| opi | g27 | user | 84950 | 12185.026 | 2166.573 |
| opi | l70 | system | 15200 | -0.889 | 0.020 |
| opi | l70 | template | 28000 | -0.903 | 0.053 |
| opi | l70 | user | 83140 | -0.848 | 0.031 |
| tt | q7 | assistant | 13607 | 303.866 | 70.792 |
| tt | q7 | system | 213672 | 294.808 | 77.626 |
| tt | q7 | template | 20337 | 302.360 | 49.219 |
| tt | q7 | user | 280739 | 221.582 | 120.926 |
| tt | g12 | assistant | 18596 | 1.057 | 0.358 |
| tt | g12 | system | 219324 | 1.085 | 0.210 |
| tt | g12 | template | 15428 | 2.183 | 3.399 |
| tt | g12 | user | 290432 | 0.937 | 0.341 |
| tt | g27 | assistant | 24532 | 11395.121 | 2956.330 |
| tt | g27 | system | 219530 | 12691.982 | 2471.063 |
| tt | g27 | template | 15468 | 19078.691 | 8329.047 |
| tt | g27 | user | 290890 | 10299.913 | 3855.167 |
| tt | l70 | assistant | 19557 | 21.569 | 3.406 |
| tt | l70 | system | 212103 | 21.875 | 3.473 |
| tt | l70 | template | 54249 | 44.196 | 129.746 |
| tt | l70 | user | 278670 | 18.745 | 6.973 |
| liars | g27 | assistant | 208381 | 0.552 | 0.689 |
| liars | g27 | assistant_prior | 474018 | 0.711 | 0.857 |
| liars | g27 | system | 204001 | 1.354 | 1.875 |
| liars | g27 | template | 87249 | 0.938 | 3.088 |
| liars | g27 | user | 471084 | 1.002 | 1.240 |
| liars | l70 | assistant | 70616 | 0.810 | 0.039 |
| liars | l70 | assistant_prior | 89643 | 0.772 | 0.023 |
| liars | l70 | system | 81896 | 0.856 | 0.030 |
| liars | l70 | template | 87570 | 0.879 | 0.068 |
| liars | l70 | user | 264171 | 0.823 | 0.041 |
| taboo | q7 | assistant | 3062 | 0.306 | 0.080 |
| taboo | q7 | template | 2784 | 0.302 | 0.146 |
| taboo | q7 | user | 1464 | 0.342 | 0.082 |
| taboo | g12 | assistant | 2932 | 0.969 | 0.011 |
| taboo | g12 | template | 864 | 0.859 | 0.302 |
| taboo | g12 | user | 1464 | 0.971 | 0.007 |
| taboo | g27 | assistant | 3203 | 0.943 | 0.017 |
| taboo | g27 | template | 864 | 0.826 | 0.294 |
| taboo | g27 | user | 1464 | 0.934 | 0.019 |
| taboo | l70 | assistant | 3759 | 26.968 | 5.211 |
| taboo | l70 | template | 3360 | 46.065 | 130.084 |
| taboo | l70 | user | 1464 | 25.210 | 3.857 |

## Extreme activation proxy

| dataset | model | `norm_ratio > 5` | README count | share | on-task extreme | on-task ordinary | at position 0 | position median | position max | segment counts |
|---|---|---|---|---|---|---|---|---|---|---|
| opi | q7 | 800 | 800 | 0.007 | 0.000 | 0.256 | 0 | 2.000 | 2 | input:800 |
| opi | g12 | 800 | 800 | 0.007 | 0.000 | 0.171 | 800 | 0.000 | 0 | input:800 |
| opi | g27 | 805 | 805 | 0.007 | 0.000 | 0.185 | 800 | 0.000 | 166 | input:805 |
| opi | l70 | 800 | 800 | 0.006 | 0.009 | 0.130 | 800 | 0.000 | 0 | input:800 |
| tt | q7 | 2303 | 2303 | 0.004 | 0.231 | 0.839 | 0 | 2.000 | 543 | input:2303 |
| tt | g12 | 1962 | 1963 | 0.004 | 0.051 | 0.852 | 1544 | 0.000 | 1569 | boundary:16, input:1926, output:20 |
| tt | g27 | 2073 | 2073 | 0.004 | 0.998 | 0.863 | 1548 | 0.000 | 1520 | boundary:22, input:2022, output:29 |
| tt | l70 | 1550 | 1550 | 0.003 | 0.907 | 0.680 | 1550 | 0.000 | 0 | input:1550 |
| liars | g27 | 4333 | 4333 | 0.003 | 0.000 | 0.011 | 2000 | 518.000 | 2026 | input:3989, output:344 |
| liars | l70 | 2000 | 2000 | 0.003 | 0.000 | 0.023 | 2000 | 0.000 | 0 | input:2000 |
| taboo | q7 | 96 | 96 | 0.013 | 0.000 | 0.215 | 0 | 2.000 | 2 | input:96 |
| taboo | g12 | 96 | 96 | 0.018 | 0.000 | 0.281 | 96 | 0.000 | 0 | input:96 |
| taboo | g27 | 96 | 96 | 0.017 | 0.000 | 0.306 | 96 | 0.000 | 0 | input:96 |
| taboo | l70 | 96 | 96 | 0.011 | 0.000 | 0.164 | 96 | 0.000 | 0 | input:96 |

`is_extreme_norm` is a token-level analysis proxy, not the factor-of-10 spike-channel definition in `all_tokens_eval.py`. No token is removed. Sparse tasks and dense Tensor Trust must be interpreted separately.

## Position and length diagnostics

Bin means are ordered 0–256, 256–512, 512–1024, 1024–2048, and 2048+.

| dataset | model | metric | corr absolute position | corr normalized position | corr full length | corr segment length | within-case z mean | within-case z std | percentile min | percentile max | aligned bin means |
|---|---|---|---|---|---|---|---|---|---|---|---|
| opi | q7 | peak_ratio | 0.225 | 0.182 | 0.108 | 0.102 | -0.000 | 0.996 | 0.000 | 1.000 | -18.870, —, —, —, — |
| opi | g12 | resid_jump_nla | -0.120 | -0.131 | -0.047 | -0.165 | 0.000 | 0.996 | 0.000 | 1.000 | 11883.832, —, —, —, — |
| opi | g27 | resid_jump_nla | -0.130 | -0.137 | -0.044 | -0.132 | 0.000 | 0.996 | 0.000 | 1.000 | 12765.973, —, —, —, — |
| opi | l70 | sink_drain | 0.660 | 0.689 | 0.105 | -0.045 | 0.000 | 0.997 | 0.000 | 1.000 | -0.865, —, —, —, — |
| tt | q7 | resid_jump | -0.157 | -0.007 | -0.235 | -0.245 | -0.000 | 0.999 | 0.000 | 1.000 | 273.948, 223.371, 245.855, 263.342, — |
| tt | g12 | norm_ratio | -0.117 | -0.108 | -0.101 | -0.095 | -0.000 | 0.999 | 0.000 | 1.000 | 1.098, 0.946, 0.969, 0.920, — |
| tt | g27 | resid_jump_nla | -0.210 | -0.087 | -0.242 | -0.238 | 0.000 | 0.999 | 0.000 | 1.000 | 12251.031, 10517.221, 10672.962, 10637.714, — |
| tt | l70 | resid_jump_nla | -0.076 | -0.085 | -0.058 | -0.053 | -0.000 | 0.999 | 0.000 | 1.000 | 24.800, 18.671, 19.900, 21.830, — |
| liars | g27 | varentropy | -0.054 | -0.072 | -0.128 | -0.049 | -0.000 | 0.999 | 0.000 | 1.000 | 1.243, 0.342, 0.822, 0.924, 0.716 |
| liars | l70 | sink_drain | -0.728 | -0.525 | -0.634 | -0.530 | -0.000 | 0.998 | 0.000 | 1.000 | 0.860, 0.802, 0.785, 0.769, — |
| taboo | q7 | dominant_mass | 0.023 | 0.077 | -0.081 | -0.213 | -0.000 | 0.993 | 0.000 | 1.000 | 0.312, —, —, —, — |
| taboo | g12 | dominant_mass | 0.192 | 0.211 | 0.042 | 0.099 | 0.000 | 0.991 | 0.000 | 1.000 | 0.952, —, —, —, — |
| taboo | g27 | dominant_mass | 0.214 | 0.233 | 0.032 | 0.132 | -0.000 | 0.991 | 0.000 | 1.000 | 0.922, —, —, —, — |
| taboo | l70 | resid_jump_nla | -0.152 | -0.154 | -0.025 | 0.015 | 0.000 | 0.994 | 0.000 | 1.000 | 34.009, —, —, —, — |

Within-case z-scores use sample standard deviation with the repository epsilon; percentiles run from 0 (least relevant) to 1 (most relevant) after direction alignment. They are retained for later fixed-budget analyses.

### Gemma sink-window diagnostic

| dataset | model | sink before 1024 | sink at/after 1024 | tokens after |
|---|---|---|---|---|
| tt | g12 | -0.541 | -0.059 | 6252 |
| tt | g27 | -0.580 | -0.065 | 6307 |
| liars | g27 | -0.479 | -0.059 | 515526 |

The post-1024 flag documents contamination of Gemma `sink_drain`; it is not applied to unrelated metrics.

## Interpretation constraints

- Internal-state metrics may partly identify states the NLA verbalizes poorly, rather than positions that are intrinsically meaningful.
- Narrow intervals quantify sampling precision, not validity of the judge's definition of on-task.
- Template, extreme-activation, dense-task, and post-window rows are retained and explicitly labelled rather than filtered.

## Audit findings

No hard integrity failures were found.

### Expected warnings

- tt/g12: norm_ratio > 5 gives 1962 extreme rows; README table reports 1963.
