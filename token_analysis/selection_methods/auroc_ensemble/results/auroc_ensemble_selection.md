# Exhaustive AUROC rank-ensemble selection

This is the blind primary analysis over the 13 deployable metrics.

Finite values are converted to pooled fractional midranks, component directions are aligned using pooled AUROC, and every original plus every 25/75, 50/50, and 75/25 unordered pair competes in the same pool. Full-data winners are exploratory; grouped held-out diagnostics estimate selection generalization.

## Model-specific all-token winners

| dataset | model | candidate | type | AUROC | gain vs single | held-out AUROC | held-out macro delta | common-support change |
|---|---|---|---|---|---|---|---|---|
| liars | g27 | mix::head_disagreement@0.50+w@0.50 | ensemble | 0.8142 | 0.0943 | 0.8142 | 0.0798 | False |
| liars | l70 | mix::head_disagreement@0.50+norm_ratio@0.50 | ensemble | 0.7091 | 0.0269 | 0.7090 | 0.0922 | True |
| opi | g12 | mix::lookback_ratio@0.50+sink_drain@0.50 | ensemble | 0.8098 | 0.1248 | 0.8098 | 0.1337 | False |
| opi | g27 | mix::lookback_ratio@0.50+sink_drain@0.50 | ensemble | 0.8101 | 0.1754 | 0.8101 | 0.1941 | False |
| opi | l70 | mix::peak_ratio@0.50+sink_drain@0.50 | ensemble | 0.7292 | 0.0558 | 0.7292 | 0.0595 | False |
| opi | q7 | mix::lookback_ratio@0.50+sink_drain@0.50 | ensemble | 0.7856 | 0.0250 | 0.7856 | 0.0303 | True |
| taboo | g12 | mix::dominant_mass@0.75+resid_jump@0.25 | ensemble | 0.7328 | 0.0520 | 0.7302 | 0.0388 | True |
| taboo | g27 | mix::dominant_mass@0.50+norm_ratio@0.50 | ensemble | 0.7710 | 0.1197 | 0.7719 | 0.1535 | False |
| taboo | l70 | mix::lookback_ratio@0.50+w@0.50 | ensemble | 0.8440 | 0.1213 | 0.8436 | 0.1373 | False |
| taboo | q7 | mix::dominant_mass@0.75+w@0.25 | ensemble | 0.8352 | 0.0396 | 0.8311 | 0.0048 | False |
| tt | g12 | mix::norm_ratio@0.75+resid_jump@0.25 | ensemble | 0.7398 | 0.0210 | 0.7398 | 0.0413 | False |
| tt | g27 | mix::peak_ratio@0.50+resid_jump@0.50 | ensemble | 0.6904 | 0.0182 | 0.6904 | 0.0365 | False |
| tt | l70 | mix::head_disagreement@0.50+sink_drain@0.50 | ensemble | 0.7826 | 0.0873 | 0.7826 | 0.1454 | False |
| tt | q7 | mix::resid_jump@0.50+w@0.50 | ensemble | 0.6754 | 0.0364 | 0.6754 | 0.1116 | False |

## Dataset-shared all-token winner

| dataset | model | candidate | equal-model AUROC | model AUROC |
|---|---|---|---|---|
| liars | g27 | mix::dominant_mass@0.50+head_disagreement@0.50 | 0.7216 | 0.7698 |
| liars | l70 | mix::dominant_mass@0.50+head_disagreement@0.50 | 0.7216 | 0.6734 |
| opi | g12 | mix::lookback_ratio@0.50+sink_drain@0.50 | 0.7540 | 0.8098 |
| opi | g27 | mix::lookback_ratio@0.50+sink_drain@0.50 | 0.7540 | 0.8101 |
| opi | l70 | mix::lookback_ratio@0.50+sink_drain@0.50 | 0.7540 | 0.6105 |
| opi | q7 | mix::lookback_ratio@0.50+sink_drain@0.50 | 0.7540 | 0.7856 |
| taboo | g12 | mix::dominant_mass@0.50+norm_ratio@0.50 | 0.7496 | 0.6914 |
| taboo | g27 | mix::dominant_mass@0.50+norm_ratio@0.50 | 0.7496 | 0.7710 |
| taboo | l70 | mix::dominant_mass@0.50+norm_ratio@0.50 | 0.7496 | 0.7792 |
| taboo | q7 | mix::dominant_mass@0.50+norm_ratio@0.50 | 0.7496 | 0.7568 |
| tt | g12 | mix::resid_jump_nla@0.75+w@0.25 | 0.6824 | 0.7048 |
| tt | g27 | mix::resid_jump_nla@0.75+w@0.25 | 0.6824 | 0.6790 |
| tt | l70 | mix::resid_jump_nla@0.75+w@0.25 | 0.6824 | 0.7013 |
| tt | q7 | mix::resid_jump_nla@0.75+w@0.25 | 0.6824 | 0.6444 |

## Dataset-shared grouped held-out validation

| dataset | held-out macro AUROC | single macro AUROC | delta | paired 95% CI | held-out macro AP | modal fold winner | modal frequency | full-winner frequency |
|---|---|---|---|---|---|---|---|---|
| liars | 0.5407 | 0.4567 | 0.0840 | [0.0780, 0.0901] | 0.0643 | mix::dominant_mass@0.50+head_disagreement@0.50 | 0.8000 | 0.8000 |
| opi | 0.7613 | 0.6556 | 0.1057 | [0.1026, 0.1091] | 0.4056 | mix::lookback_ratio@0.50+sink_drain@0.50 | 1.0000 | 1.0000 |
| taboo | 0.7752 | 0.7222 | 0.0530 | [0.0410, 0.0651] | 0.5211 | mix::dominant_mass@0.50+norm_ratio@0.50 | 1.0000 | 1.0000 |
| tt | 0.5669 | 0.5363 | 0.0306 | [0.0287, 0.0323] | 0.8858 | mix::resid_jump_nla@0.75+w@0.25 | 1.0000 | 1.0000 |

The complete shared candidate and the shared original-only comparator are reselected inside every training fold. Models receive equal weight. The interval resamples union case-ID clusters and is not a token bootstrap.

## Leave-one-model-out transfer

| dataset | held-out model | protocol | candidate | case-macro AUROC | case-macro AP | delta vs single | paired 95% CI | top-1% precision | weak evidence |
|---|---|---|---|---|---|---|---|---|---|
| liars | g27 | label_free | mix::head_disagreement@0.50+norm_ratio@0.50 | 0.3802 | 0.0657 | -0.0925 | [-0.0977, -0.0872] | 0.0010 | True |
| liars | g27 | target_calibrated | mix::head_disagreement@0.50+norm_ratio@0.50 | 0.5322 | 0.0560 | 0.0595 | [0.0542, 0.0650] | 0.0029 | True |
| liars | l70 | label_free | mix::head_disagreement@0.50+w@0.50 | 0.5604 | 0.0630 | 0.1197 | [0.1094, 0.1292] | 0.0162 | True |
| liars | l70 | target_calibrated | mix::head_disagreement@0.50+w@0.50 | 0.4264 | 0.0457 | -0.0142 | [-0.0168, -0.0117] | 0.0000 | True |
| opi | g12 | label_free | mix::lookback_ratio@0.50+sink_drain@0.50 | 0.8162 | 0.4279 | 0.2200 | [0.2137, 0.2264] | 0.3496 | False |
| opi | g12 | target_calibrated | mix::lookback_ratio@0.50+sink_drain@0.50 | 0.8162 | 0.4279 | 0.2200 | [0.2137, 0.2264] | 0.3496 | False |
| opi | g27 | label_free | mix::lookback_ratio@0.50+sink_drain@0.50 | 0.8196 | 0.4901 | 0.2000 | [0.1949, 0.2047] | 0.5492 | False |
| opi | g27 | target_calibrated | mix::lookback_ratio@0.50+sink_drain@0.50 | 0.8196 | 0.4901 | 0.2000 | [0.1949, 0.2047] | 0.5492 | False |
| opi | l70 | label_free | mix::lookback_ratio@0.50+sink_drain@0.50 | 0.7776 | 0.3154 | 0.1355 | [0.1286, 0.1425] | 0.1808 | False |
| opi | l70 | target_calibrated | mix::lookback_ratio@0.50+sink_drain@0.50 | 0.6147 | 0.1923 | -0.0274 | [-0.0355, -0.0194] | 0.0867 | False |
| opi | q7 | label_free | mix::lookback_ratio@0.50+sink_drain@0.50 | 0.7948 | 0.5119 | 0.0303 | [0.0251, 0.0356] | 0.4288 | False |
| opi | q7 | target_calibrated | mix::lookback_ratio@0.50+sink_drain@0.50 | 0.7948 | 0.5119 | 0.1482 | [0.1429, 0.1540] | 0.4288 | False |
| taboo | g12 | label_free | mix::dominant_mass@0.75+w@0.25 | 0.6934 | 0.4702 | -0.0045 | [-0.0190, 0.0102] | 0.4479 | False |
| taboo | g12 | target_calibrated | mix::dominant_mass@0.50+norm_ratio@0.50 | 0.7562 | 0.5557 | 0.0582 | [0.0211, 0.0947] | 0.0833 | False |
| taboo | g27 | label_free | mix::dominant_mass@0.50+w@0.50 | 0.5517 | 0.4554 | -0.0651 | [-0.0977, -0.0316] | 0.4375 | False |
| taboo | g27 | target_calibrated | mix::dominant_mass@0.50+w@0.50 | 0.5517 | 0.4554 | -0.0651 | [-0.0977, -0.0316] | 0.4375 | False |
| taboo | l70 | label_free | mix::dominant_mass@0.75+varentropy@0.25 | 0.7087 | 0.3330 | -0.0448 | [-0.0546, -0.0353] | 0.0000 | False |
| taboo | l70 | target_calibrated | mix::dominant_mass@0.75+varentropy@0.25 | 0.7087 | 0.3330 | -0.0448 | [-0.0546, -0.0353] | 0.0000 | False |
| taboo | q7 | label_free | mix::lookback_ratio@0.50+w@0.50 | 0.6359 | 0.3884 | -0.1845 | [-0.2080, -0.1606] | 0.4688 | False |
| taboo | q7 | target_calibrated | mix::dominant_mass@0.50+norm_ratio@0.50 | 0.7992 | 0.5514 | -0.0211 | [-0.0344, -0.0071] | 0.7396 | False |
| tt | g12 | label_free | mix::head_disagreement@0.50+sink_drain@0.50 | 0.5959 | 0.9257 | 0.0461 | [0.0374, 0.0546] | 0.9530 | False |
| tt | g12 | target_calibrated | mix::resid_jump_nla@0.50+w@0.50 | 0.5616 | 0.9216 | 0.0118 | [0.0049, 0.0186] | 0.9425 | False |
| tt | g27 | label_free | mix::head_disagreement@0.50+sink_drain@0.50 | 0.5679 | 0.9219 | 0.0051 | [-0.0032, 0.0125] | 0.9437 | False |
| tt | g27 | target_calibrated | mix::dominant_mass@0.50+resid_jump@0.50 | 0.5825 | 0.9224 | 0.0196 | [0.0120, 0.0275] | 0.9687 | False |
| tt | l70 | label_free | mix::resid_jump_nla@0.75+w@0.25 | 0.5872 | 0.8034 | 0.0366 | [0.0346, 0.0387] | 0.8381 | False |
| tt | l70 | target_calibrated | mix::peak_ratio@0.50+resid_jump@0.50 | 0.5607 | 0.7755 | 0.0101 | [0.0048, 0.0153] | 0.5613 | False |
| tt | q7 | label_free | mix::head_disagreement@0.50+sink_drain@0.50 | 0.6440 | 0.9136 | 0.1621 | [0.1529, 0.1709] | 0.9545 | False |
| tt | q7 | target_calibrated | mix::head_disagreement@0.50+sink_drain@0.50 | 0.6432 | 0.9086 | 0.1613 | [0.1501, 0.1723] | 0.9730 | False |

'label_free' fits target empirical CDFs without target judgment labels and uses directions learned from the remaining models. 'target_calibrated' keeps candidate identity and weights fixed but learns directions on target-training cases. Liars has only one training model per holdout, so its transfer result is descriptive weak evidence.

## Segment-specific model winners

| dataset | model | segment | candidate | AUROC |
|---|---|---|---|---|
| liars | g27 | boundary | mix::norm_ratio@0.50+resid_jump@0.50 | 0.8223 |
| liars | g27 | input | mix::sink_drain@0.50+w@0.50 | 0.8463 |
| liars | g27 | output | mix::dominant_mass@0.50+lookback_ratio@0.50 | 0.7264 |
| liars | l70 | boundary | mix::resid_jump_nla@0.50+varentropy@0.50 | 0.8296 |
| liars | l70 | input | mix::head_disagreement@0.75+temporal_kl@0.25 | 0.7659 |
| liars | l70 | output | mix::dominant_mass@0.50+peak_ratio@0.50 | 0.6582 |
| opi | g12 | boundary | mix::peak_ratio@0.50+sink_drain@0.50 | 0.7735 |
| opi | g12 | input | mix::lookback_ratio@0.50+sink_drain@0.50 | 0.8192 |
| opi | g27 | boundary | mix::dominant_mass@0.50+sink_drain@0.50 | 0.7475 |
| opi | g27 | input | mix::lookback_ratio@0.50+sink_drain@0.50 | 0.8175 |
| opi | l70 | boundary | mix::resid_jump@0.75+resid_jump_nla@0.25 | 0.8514 |
| opi | l70 | input | mix::peak_ratio@0.50+sink_drain@0.50 | 0.7428 |
| opi | q7 | boundary | mix::peak_ratio@0.75+resid_jump_nla@0.25 | 0.7852 |
| opi | q7 | input | mix::lookback_ratio@0.50+sink_drain@0.50 | 0.7945 |
| taboo | g12 | boundary | mix::dominant_mass@0.75+temporal_kl@0.25 | 0.9860 |
| taboo | g12 | input | mix::entropy@0.50+peak_ratio@0.50 | 0.7920 |
| taboo | g12 | output | mix::sink_drain@0.50+w@0.50 | 0.8786 |
| taboo | g27 | boundary | mix::dominant_mass@0.75+peak_ratio@0.25 | 0.9418 |
| taboo | g27 | input | mix::dominant_mass@0.25+head_disagreement@0.75 | 0.7421 |
| taboo | g27 | output | mix::dominant_mass@0.50+sink_drain@0.50 | 0.8752 |
| taboo | l70 | boundary | mix::norm_ratio@0.50+resid_jump@0.50 | 0.9720 |
| taboo | l70 | input | mix::dominant_mass@0.25+sink_drain@0.75 | 0.8463 |
| taboo | l70 | output | mix::lookback_ratio@0.50+norm_ratio@0.50 | 0.9123 |
| taboo | q7 | boundary | mix::resid_jump_nla@0.75+sink_drain@0.25 | 0.9214 |
| taboo | q7 | input | mix::dominant_mass@0.75+norm_ratio@0.25 | 0.8295 |
| taboo | q7 | output | mix::head_disagreement@0.50+w@0.50 | 0.9126 |
| tt | g12 | boundary | mix::resid_jump@0.25+surprisal@0.75 | 0.6901 |
| tt | g12 | input | mix::norm_ratio@0.75+resid_jump@0.25 | 0.7459 |
| tt | g12 | output | mix::lookback_ratio@0.75+resid_jump_nla@0.25 | 0.6406 |
| tt | g27 | boundary | mix::dominant_mass@0.75+resid_jump@0.25 | 0.6985 |
| tt | g27 | input | mix::peak_ratio@0.50+resid_jump@0.50 | 0.6940 |
| tt | g27 | output | mix::lookback_ratio@0.75+resid_jump@0.25 | 0.6270 |
| tt | l70 | boundary | mix::dominant_mass@0.50+varentropy@0.50 | 0.6873 |
| tt | l70 | input | mix::head_disagreement@0.50+sink_drain@0.50 | 0.7858 |
| tt | l70 | output | mix::head_disagreement@0.50+sink_drain@0.50 | 0.6610 |
| tt | q7 | boundary | mix::resid_jump@0.50+w@0.50 | 0.6439 |
| tt | q7 | input | mix::resid_jump@0.50+w@0.50 | 0.6736 |
| tt | q7 | output | mix::lookback_ratio@0.50+sink_drain@0.50 | 0.7642 |

## Segment-specific shared winners

| dataset | segment | candidate | equal-model AUROC |
|---|---|---|---|
| liars | boundary | mix::head_disagreement@0.50+varentropy@0.50 | 0.7787 |
| liars | input | mix::entropy@0.50+head_disagreement@0.50 | 0.7724 |
| liars | output | mix::dominant_mass@0.50+peak_ratio@0.50 | 0.6731 |
| opi | boundary | mix::dominant_mass@0.50+resid_jump@0.50 | 0.7235 |
| opi | input | mix::lookback_ratio@0.50+sink_drain@0.50 | 0.7622 |
| taboo | boundary | mix::dominant_mass@0.75+norm_ratio@0.25 | 0.8939 |
| taboo | input | mix::dominant_mass@0.50+sink_drain@0.50 | 0.7721 |
| taboo | output | mix::sink_drain@0.50+w@0.50 | 0.8777 |
| tt | boundary | mix::dominant_mass@0.50+surprisal@0.50 | 0.6307 |
| tt | input | mix::peak_ratio@0.25+resid_jump_nla@0.75 | 0.6842 |
| tt | output | mix::lookback_ratio@0.50+varentropy@0.50 | 0.6430 |

Boundary tokens are structural strings, but their hidden states are conditioned on the full preceding input and can therefore be informative aggregation points. Segment comparisons are descriptive associations with judge labels, not causal tests of NLA verbalization utility. Liars trailers are excluded from every segment summary and figure.
