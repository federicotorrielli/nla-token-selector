# Input–boundary–output importance analysis

All values below use the frozen all-token winner, including its all-token rank mapping, component directions, weights, and final direction. Thus segment AUROCs are comparable. Mean relevance and enrichment are averaged within cases first.

## Model-specific fixed-winner estimates

| dataset | model | segment | candidate | tokens | cases | AUROC cases | on-task rate | template fraction | pooled AUROC | case-macro AUROC | mean percentile | top-1% enrichment | top-10% enrichment |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| liars | g27 | boundary | mix::head_disagreement@0.50+w@0.50 | 9144 | 2000 | 272 | 0.0360 | 1.0000 | 0.6344 | 0.4219 | 0.4837 | 0.0000 | 0.0347 |
| liars | g27 | input | mix::head_disagreement@0.50+w@0.50 | 1223208 | 2000 | 1521 | 0.0090 | 0.0606 | 0.8441 | 0.6154 | 0.3675 | 0.0736 | 0.2326 |
| liars | g27 | output | mix::head_disagreement@0.50+w@0.50 | 208381 | 2000 | 879 | 0.0172 | 0.0000 | 0.6386 | 0.4218 | 0.8270 | 4.8340 | 3.5991 |
| liars | l70 | boundary | mix::head_disagreement@0.50+norm_ratio@0.50 | 10000 | 2000 | 390 | 0.0423 | 1.0000 | 0.4253 | 0.3376 | 0.5721 | 0.5341 | 1.6170 |
| liars | l70 | input | mix::head_disagreement@0.50+norm_ratio@0.50 | 511280 | 2000 | 1123 | 0.0216 | 0.1478 | 0.7674 | 0.4550 | 0.5738 | 1.2992 | 1.2514 |
| liars | l70 | output | mix::head_disagreement@0.50+norm_ratio@0.50 | 70616 | 2000 | 769 | 0.0322 | 0.0000 | 0.4914 | 0.5293 | 0.3116 | 0.0015 | 0.0461 |
| opi | g12 | boundary | mix::lookback_ratio@0.50+sink_drain@0.50 | 4000 | 800 | 583 | 0.2055 | 1.0000 | 0.4904 | 0.5240 | 0.7567 | 2.2360 | 2.2766 |
| opi | g12 | input | mix::lookback_ratio@0.50+sink_drain@0.50 | 103750 | 800 | 800 | 0.1685 | 0.0386 | 0.8203 | 0.8264 | 0.4894 | 0.9553 | 0.9582 |
| opi | g27 | boundary | mix::lookback_ratio@0.50+sink_drain@0.50 | 4000 | 800 | 699 | 0.2988 | 1.0000 | 0.5256 | 0.5663 | 0.7122 | 2.8791 | 1.8826 |
| opi | g27 | input | mix::lookback_ratio@0.50+sink_drain@0.50 | 103750 | 800 | 800 | 0.1796 | 0.0386 | 0.8182 | 0.8271 | 0.4911 | 0.9261 | 0.9733 |
| opi | l70 | boundary | mix::peak_ratio@0.50+sink_drain@0.50 | 4000 | 800 | 356 | 0.0983 | 1.0000 | 0.7203 | 0.7912 | 0.8939 | 20.9841 | 5.9106 |
| opi | l70 | input | mix::peak_ratio@0.50+sink_drain@0.50 | 122340 | 800 | 800 | 0.1302 | 0.1962 | 0.7404 | 0.7469 | 0.4863 | 0.3266 | 0.8289 |
| opi | q7 | boundary | mix::lookback_ratio@0.50+sink_drain@0.50 | 4000 | 800 | 733 | 0.3553 | 1.0000 | 0.6703 | 0.7923 | 0.8829 | 8.3110 | 5.2045 |
| opi | q7 | input | mix::lookback_ratio@0.50+sink_drain@0.50 | 107370 | 800 | 800 | 0.2505 | 0.0596 | 0.7941 | 0.8051 | 0.4849 | 0.7383 | 0.8363 |
| taboo | g12 | boundary | mix::dominant_mass@0.75+resid_jump@0.25 | 480 | 96 | 96 | 0.5854 | 1.0000 | 0.9151 | 0.9306 | 0.4788 | 0.0000 | 1.7126 |
| taboo | g12 | input | mix::dominant_mass@0.75+resid_jump@0.25 | 1848 | 96 | 96 | 0.1607 | 0.2078 | 0.6316 | 0.6280 | 0.4952 | 2.9053 | 1.4727 |
| taboo | g12 | output | mix::dominant_mass@0.75+resid_jump@0.25 | 2932 | 96 | 69 | 0.2981 | 0.0000 | 0.7767 | 0.6810 | 0.5116 | 0.0165 | 0.6270 |
| taboo | g27 | boundary | mix::dominant_mass@0.50+norm_ratio@0.50 | 480 | 96 | 94 | 0.5500 | 1.0000 | 0.8029 | 0.8333 | 0.5246 | 0.0000 | 1.1992 |
| taboo | g27 | input | mix::dominant_mass@0.50+norm_ratio@0.50 | 1848 | 96 | 96 | 0.2560 | 0.2078 | 0.6802 | 0.7177 | 0.6057 | 2.7207 | 1.7109 |
| taboo | g27 | output | mix::dominant_mass@0.50+norm_ratio@0.50 | 3203 | 96 | 60 | 0.2885 | 0.0000 | 0.8414 | 0.8381 | 0.4619 | 0.1858 | 0.6840 |
| taboo | l70 | boundary | mix::lookback_ratio@0.50+w@0.50 | 480 | 96 | 96 | 0.5771 | 1.0000 | 0.7366 | 0.7318 | 0.9300 | 13.4792 | 7.2324 |
| taboo | l70 | input | mix::lookback_ratio@0.50+w@0.50 | 4344 | 96 | 88 | 0.1123 | 0.6630 | 0.7811 | 0.8345 | 0.4312 | 0.4827 | 1.0232 |
| taboo | l70 | output | mix::lookback_ratio@0.50+w@0.50 | 3759 | 96 | 31 | 0.1660 | 0.0000 | 0.8908 | 0.7034 | 0.5287 | 0.0000 | 0.3015 |
| taboo | q7 | boundary | mix::dominant_mass@0.75+w@0.25 | 480 | 96 | 96 | 0.4313 | 1.0000 | 0.8191 | 0.8594 | 0.6364 | 0.4875 | 2.6714 |
| taboo | q7 | input | mix::dominant_mass@0.75+w@0.25 | 3768 | 96 | 96 | 0.1056 | 0.6115 | 0.8056 | 0.7865 | 0.4134 | 0.0122 | 0.3930 |
| taboo | q7 | output | mix::dominant_mass@0.75+w@0.25 | 3062 | 96 | 54 | 0.3080 | 0.0000 | 0.8258 | 0.7445 | 0.6250 | 2.3407 | 1.7888 |
| tt | g12 | boundary | mix::norm_ratio@0.75+resid_jump@0.25 | 7720 | 1544 | 222 | 0.9692 | 1.0000 | 0.5071 | 0.5364 | 0.5619 | 1.3514 | 1.4715 |
| tt | g12 | input | mix::norm_ratio@0.75+resid_jump@0.25 | 517464 | 1544 | 1514 | 0.8459 | 0.0149 | 0.7458 | 0.5736 | 0.4954 | 0.9826 | 0.9549 |
| tt | g12 | output | mix::norm_ratio@0.75+resid_jump@0.25 | 18596 | 1544 | 385 | 0.8859 | 0.0000 | 0.5055 | 0.5757 | 0.5249 | 0.8406 | 0.6999 |
| tt | g27 | boundary | mix::peak_ratio@0.50+resid_jump@0.50 | 7740 | 1548 | 113 | 0.9848 | 1.0000 | 0.3711 | 0.3355 | 0.6300 | 2.4252 | 1.5454 |
| tt | g27 | input | mix::peak_ratio@0.50+resid_jump@0.50 | 518148 | 1548 | 1525 | 0.8597 | 0.0149 | 0.6936 | 0.5959 | 0.4802 | 0.8603 | 0.8777 |
| tt | g27 | output | mix::peak_ratio@0.50+resid_jump@0.50 | 24532 | 1548 | 409 | 0.9125 | 0.0000 | 0.4885 | 0.5209 | 0.6057 | 2.0374 | 1.3876 |
| tt | l70 | boundary | mix::head_disagreement@0.50+sink_drain@0.50 | 7750 | 1550 | 787 | 0.8649 | 1.0000 | 0.5117 | 0.5158 | 0.7926 | 6.0513 | 3.6280 |
| tt | l70 | input | mix::head_disagreement@0.50+sink_drain@0.50 | 537272 | 1550 | 1550 | 0.6714 | 0.0865 | 0.7852 | 0.6957 | 0.4922 | 0.8428 | 0.9081 |
| tt | l70 | output | mix::head_disagreement@0.50+sink_drain@0.50 | 19557 | 1550 | 415 | 0.8724 | 0.0000 | 0.6740 | 0.6675 | 0.6179 | 0.5769 | 2.0318 |
| tt | q7 | boundary | mix::resid_jump@0.50+w@0.50 | 7760 | 1552 | 67 | 0.9907 | 1.0000 | 0.6260 | 0.6219 | 0.7407 | 0.2912 | 1.9863 |
| tt | q7 | input | mix::resid_jump@0.50+w@0.50 | 506988 | 1552 | 1551 | 0.8331 | 0.0248 | 0.6737 | 0.6436 | 0.4765 | 0.5386 | 0.7999 |
| tt | q7 | output | mix::resid_jump@0.50+w@0.50 | 13607 | 1552 | 276 | 0.8779 | 0.0000 | 0.6450 | 0.5626 | 0.9394 | 32.3321 | 7.8392 |

## Matched-case segment contrasts

| dataset | model | contrast | matched cases | pooled delta | case-macro delta | 95% low | 95% high |
|---|---|---|---|---|---|---|---|
| liars | g27 | boundary − input | 270 | -0.2097 | -0.0799 | -0.1109 | -0.0489 |
| liars | g27 | boundary − output | 258 | -0.0042 | 0.0688 | 0.0312 | 0.1032 |
| liars | g27 | output − input | 868 | -0.2055 | -0.1719 | -0.1870 | -0.1560 |
| liars | l70 | boundary − input | 379 | -0.3421 | -0.1424 | -0.1655 | -0.1184 |
| liars | l70 | boundary − output | 288 | -0.0661 | -0.1987 | -0.2349 | -0.1631 |
| liars | l70 | output − input | 720 | -0.2760 | 0.0927 | 0.0729 | 0.1120 |
| opi | g12 | boundary − input | 583 | -0.3299 | -0.3003 | -0.3249 | -0.2764 |
| opi | g27 | boundary − input | 699 | -0.2926 | -0.2626 | -0.2830 | -0.2418 |
| opi | l70 | boundary − input | 356 | -0.0201 | 0.0352 | 0.0073 | 0.0627 |
| opi | q7 | boundary − input | 733 | -0.1238 | -0.0104 | -0.0273 | 0.0060 |
| taboo | g12 | boundary − input | 96 | 0.2835 | 0.3026 | 0.2606 | 0.3418 |
| taboo | g12 | boundary − output | 69 | 0.1384 | 0.2538 | 0.1935 | 0.3120 |
| taboo | g12 | output − input | 69 | 0.1451 | 0.0577 | -0.0104 | 0.1253 |
| taboo | g27 | boundary − input | 94 | 0.1226 | 0.1138 | 0.0693 | 0.1581 |
| taboo | g27 | boundary − output | 59 | -0.0386 | 0.0474 | -0.0025 | 0.1004 |
| taboo | g27 | output − input | 60 | 0.1612 | 0.1354 | 0.0919 | 0.1782 |
| taboo | l70 | boundary − input | 88 | -0.0445 | -0.1006 | -0.1466 | -0.0567 |
| taboo | l70 | boundary − output | 31 | -0.1542 | 0.0063 | -0.1208 | 0.1379 |
| taboo | l70 | output − input | 25 | 0.1097 | -0.1900 | -0.2886 | -0.1036 |
| taboo | q7 | boundary − input | 96 | 0.0135 | 0.0729 | 0.0297 | 0.1129 |
| taboo | q7 | boundary − output | 54 | -0.0067 | 0.1274 | 0.0649 | 0.1874 |
| taboo | q7 | output − input | 54 | 0.0202 | -0.0330 | -0.1008 | 0.0271 |
| tt | g12 | boundary − input | 222 | -0.2387 | -0.0644 | -0.1133 | -0.0147 |
| tt | g12 | boundary − output | 88 | 0.0016 | -0.0274 | -0.1083 | 0.0536 |
| tt | g12 | output − input | 381 | -0.2403 | -0.0374 | -0.0675 | -0.0066 |
| tt | g27 | boundary − input | 113 | -0.3225 | -0.2443 | -0.2981 | -0.1920 |
| tt | g27 | boundary − output | 42 | -0.1174 | -0.2361 | -0.3272 | -0.1454 |
| tt | g27 | output − input | 406 | -0.2051 | -0.0833 | -0.1086 | -0.0587 |
| tt | l70 | boundary − input | 787 | -0.2736 | -0.1846 | -0.2093 | -0.1595 |
| tt | l70 | boundary − output | 235 | -0.1623 | -0.1899 | -0.2416 | -0.1414 |
| tt | l70 | output − input | 415 | -0.1113 | -0.0515 | -0.0769 | -0.0276 |
| tt | q7 | boundary − input | 67 | -0.0477 | -0.0017 | -0.0844 | 0.0895 |
| tt | q7 | boundary − output | 21 | -0.0191 | 0.0164 | -0.1831 | 0.2066 |
| tt | q7 | output − input | 275 | -0.0286 | -0.0593 | -0.0915 | -0.0265 |

The corresponding dataset-shared table contains 38 model–segment estimates. A positive contrast means stronger measured discrimination in the first named segment. Intervals are descriptive, pointwise whole-case bootstrap intervals; they are not simultaneous tests.

Input contains earlier scaffolding and Liars prior-assistant turns. Boundary is the final generation-prompt run and is predominantly template, but its hidden states are conditioned on preceding content. Output is final assistant text. Liars trailers are excluded. These results do not establish causal usefulness for downstream NLA verbalization.
