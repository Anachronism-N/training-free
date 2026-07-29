# v120 paired metric analysis

Positive improvement means the candidate is better after applying the metric direction. Raw deltas always equal candidate minus reference.

## ours_prototype_retrieval_age24 vs sf_native

| Metric | n | Candidate | Reference | Raw delta | 95% CI | W/T/L | p |
|---|---:|---:|---:|---:|---:|---:|---:|
| vbench.subject_consistency | 128 | 0.97298 | 0.97183 | 0.00115 | [-0.00104, 0.00345] | 69/0/59 | 0.3241 |
| vbench.subject_consistency.inclip | 128 | 0.96769 | 0.96718 | 0.00052 | [-0.00258, 0.00380] | 58/0/70 | 0.7562 |
| vbench.subject_consistency.clip2clip | 128 | 0.81316 | 0.79872 | 0.01444 | [0.00204, 0.02718] | 81/0/47 | 0.0252 |
| vbench.background_consistency | 128 | 0.96386 | 0.96376 | 0.00010 | [-0.00120, 0.00137] | 68/0/60 | 0.8836 |
| vbench.background_consistency.inclip | 128 | 0.95929 | 0.96137 | -0.00207 | [-0.00426, 0.00006] | 51/0/77 | 0.0618 |
| vbench.background_consistency.clip2clip | 128 | 0.85113 | 0.82613 | 0.02500 | [0.01734, 0.03259] | 93/0/35 | 0.0000 |
| vbench.aesthetic_quality | 128 | 0.60582 | 0.59266 | 0.01316 | [0.00669, 0.01957] | 87/0/41 | 0.0001 |
| vbench.imaging_quality | 128 | 69.12691 | 68.17180 | 0.95511 | [0.14412, 1.79605] | 67/0/61 | 0.0255 |
| vbench.motion_smoothness | 128 | 0.98378 | 0.98567 | -0.00189 | [-0.00314, -0.00076] | 44/0/84 | 0.0010 |
| vbench.temporal_flickering | 128 | 0.96804 | 0.97672 | -0.00868 | [-0.01079, -0.00675] | 23/0/105 | 0.0000 |
| vbench.dynamic_degree | 128 | 0.61719 | 0.43281 | 0.18438 | [0.13490, 0.23542] | 73/46/9 | 0.0000 |
| vbench.background_consistency.mapped_clip2clip | 128 | 0.96843 | 0.96616 | 0.00227 | [0.00166, 0.00289] | 93/0/35 | 0.0000 |
| vbench.overall_consistency | 128 | 0.23703 | 0.23472 | 0.00231 | [-0.00189, 0.00648] | 71/0/57 | 0.2826 |
| vbench.subject_consistency.mapped_clip2clip | 128 | 0.97826 | 0.97648 | 0.00178 | [0.00004, 0.00358] | 81/0/47 | 0.0536 |

## ours_prototype_retrieval_age24 vs deep_forcing

| Metric | n | Candidate | Reference | Raw delta | 95% CI | W/T/L | p |
|---|---:|---:|---:|---:|---:|---:|---:|
| vbench.subject_consistency | 128 | 0.97298 | 0.97090 | 0.00208 | [0.00025, 0.00400] | 70/0/58 | 0.0338 |
| vbench.subject_consistency.inclip | 128 | 0.96769 | 0.96364 | 0.00405 | [0.00140, 0.00676] | 78/0/50 | 0.0033 |
| vbench.subject_consistency.clip2clip | 128 | 0.81316 | 0.81693 | -0.00378 | [-0.01556, 0.00829] | 48/0/80 | 0.5419 |
| vbench.background_consistency | 128 | 0.96386 | 0.96341 | 0.00045 | [-0.00067, 0.00157] | 67/0/61 | 0.4382 |
| vbench.background_consistency.inclip | 128 | 0.95929 | 0.95735 | 0.00195 | [0.00015, 0.00379] | 73/0/55 | 0.0392 |
| vbench.background_consistency.clip2clip | 128 | 0.85113 | 0.85980 | -0.00867 | [-0.01569, -0.00177] | 51/0/77 | 0.0158 |
| vbench.aesthetic_quality | 128 | 0.60582 | 0.60151 | 0.00431 | [-0.00078, 0.00939] | 69/0/59 | 0.0995 |
| vbench.imaging_quality | 128 | 69.12691 | 68.36438 | 0.76253 | [0.06231, 1.48344] | 70/0/58 | 0.0371 |
| vbench.motion_smoothness | 128 | 0.98378 | 0.98266 | 0.00112 | [-0.00017, 0.00234] | 71/0/57 | 0.0816 |
| vbench.temporal_flickering | 128 | 0.96804 | 0.96973 | -0.00169 | [-0.00373, 0.00028] | 52/0/76 | 0.0980 |
| vbench.dynamic_degree | 128 | 0.61719 | 0.55521 | 0.06198 | [0.02344, 0.10312] | 46/60/22 | 0.0028 |
| vbench.background_consistency.mapped_clip2clip | 128 | 0.96843 | 0.96948 | -0.00105 | [-0.00164, -0.00048] | 51/0/77 | 0.0004 |
| vbench.overall_consistency | 128 | 0.23703 | 0.23811 | -0.00107 | [-0.00498, 0.00301] | 55/0/73 | 0.6091 |
| vbench.subject_consistency.mapped_clip2clip | 128 | 0.97826 | 0.97815 | 0.00010 | [-0.00142, 0.00175] | 48/0/80 | 0.9003 |

## ours_prototype_retrieval_age24 vs rolling_forcing

| Metric | n | Candidate | Reference | Raw delta | 95% CI | W/T/L | p |
|---|---:|---:|---:|---:|---:|---:|---:|
| vbench.subject_consistency | 128 | 0.97298 | 0.98053 | -0.00755 | [-0.00999, -0.00522] | 42/0/86 | 0.0000 |
| vbench.subject_consistency.inclip | 128 | 0.96769 | 0.97801 | -0.01032 | [-0.01367, -0.00718] | 44/0/84 | 0.0000 |
| vbench.subject_consistency.clip2clip | 128 | 0.81316 | 0.85562 | -0.04246 | [-0.06013, -0.02486] | 43/0/85 | 0.0000 |
| vbench.background_consistency | 128 | 0.96386 | 0.96879 | -0.00493 | [-0.00646, -0.00346] | 44/0/84 | 0.0000 |
| vbench.background_consistency.inclip | 128 | 0.95929 | 0.96524 | -0.00595 | [-0.00833, -0.00364] | 52/0/76 | 0.0000 |
| vbench.background_consistency.clip2clip | 128 | 0.85113 | 0.89733 | -0.04620 | [-0.05794, -0.03500] | 31/0/97 | 0.0000 |
| vbench.aesthetic_quality | 128 | 0.60582 | 0.60902 | -0.00321 | [-0.01162, 0.00490] | 64/0/64 | 0.4501 |
| vbench.imaging_quality | 128 | 69.12691 | 71.05817 | -1.93126 | [-2.92751, -0.97695] | 51/0/77 | 0.0001 |
| vbench.motion_smoothness | 128 | 0.98378 | 0.98802 | -0.00424 | [-0.00615, -0.00253] | 50/0/78 | 0.0000 |
| vbench.temporal_flickering | 128 | 0.96804 | 0.97809 | -0.01005 | [-0.01305, -0.00730] | 38/0/90 | 0.0000 |
| vbench.dynamic_degree | 128 | 0.61719 | 0.34583 | 0.27135 | [0.20990, 0.33385] | 79/36/13 | 0.0000 |
| vbench.background_consistency.mapped_clip2clip | 128 | 0.96843 | 0.97235 | -0.00392 | [-0.00488, -0.00298] | 31/0/97 | 0.0000 |
| vbench.overall_consistency | 128 | 0.23703 | 0.24074 | -0.00371 | [-0.00943, 0.00176] | 62/0/66 | 0.1954 |
| vbench.subject_consistency.mapped_clip2clip | 128 | 0.97826 | 0.98305 | -0.00479 | [-0.00690, -0.00274] | 43/0/85 | 0.0000 |

## ours_prototype_retrieval_age24 vs longlive

| Metric | n | Candidate | Reference | Raw delta | 95% CI | W/T/L | p |
|---|---:|---:|---:|---:|---:|---:|---:|
| vbench.subject_consistency | 128 | 0.97298 | 0.97818 | -0.00521 | [-0.00716, -0.00334] | 36/0/92 | 0.0000 |
| vbench.subject_consistency.inclip | 128 | 0.96769 | 0.97478 | -0.00708 | [-0.00988, -0.00445] | 39/0/89 | 0.0000 |
| vbench.subject_consistency.clip2clip | 128 | 0.81316 | 0.84453 | -0.03137 | [-0.04530, -0.01754] | 41/0/87 | 0.0000 |
| vbench.background_consistency | 128 | 0.96386 | 0.96693 | -0.00307 | [-0.00436, -0.00181] | 42/0/86 | 0.0000 |
| vbench.background_consistency.inclip | 128 | 0.95929 | 0.96433 | -0.00504 | [-0.00719, -0.00295] | 46/0/82 | 0.0000 |
| vbench.background_consistency.clip2clip | 128 | 0.85113 | 0.86745 | -0.01632 | [-0.02447, -0.00846] | 46/0/82 | 0.0001 |
| vbench.aesthetic_quality | 128 | 0.60582 | 0.61090 | -0.00508 | [-0.01348, 0.00315] | 62/0/66 | 0.2403 |
| vbench.imaging_quality | 128 | 69.12691 | 69.31927 | -0.19237 | [-1.10682, 0.70749] | 63/0/65 | 0.6771 |
| vbench.motion_smoothness | 128 | 0.98378 | 0.98694 | -0.00317 | [-0.00484, -0.00169] | 45/0/83 | 0.0000 |
| vbench.temporal_flickering | 128 | 0.96804 | 0.97506 | -0.00702 | [-0.00959, -0.00454] | 33/0/95 | 0.0000 |
| vbench.dynamic_degree | 128 | 0.61719 | 0.41979 | 0.19740 | [0.14792, 0.24896] | 76/37/15 | 0.0000 |
| vbench.background_consistency.mapped_clip2clip | 128 | 0.96843 | 0.96952 | -0.00110 | [-0.00172, -0.00049] | 46/0/82 | 0.0008 |
| vbench.overall_consistency | 128 | 0.23703 | 0.24141 | -0.00438 | [-0.00923, 0.00017] | 60/0/68 | 0.0697 |
| vbench.subject_consistency.mapped_clip2clip | 128 | 0.97826 | 0.98159 | -0.00333 | [-0.00509, -0.00160] | 41/0/87 | 0.0002 |

## ours_confidence_recent vs sf_native

| Metric | n | Candidate | Reference | Raw delta | 95% CI | W/T/L | p |
|---|---:|---:|---:|---:|---:|---:|---:|
| vbench.subject_consistency | 128 | 0.97310 | 0.97183 | 0.00127 | [-0.00085, 0.00352] | 67/0/61 | 0.2630 |
| vbench.subject_consistency.inclip | 128 | 0.96773 | 0.96718 | 0.00055 | [-0.00257, 0.00393] | 54/0/74 | 0.7437 |
| vbench.subject_consistency.clip2clip | 128 | 0.81544 | 0.79872 | 0.01673 | [0.00500, 0.02881] | 83/0/45 | 0.0061 |
| vbench.background_consistency | 128 | 0.96390 | 0.96376 | 0.00013 | [-0.00113, 0.00137] | 64/0/64 | 0.8357 |
| vbench.background_consistency.inclip | 128 | 0.95936 | 0.96137 | -0.00201 | [-0.00410, 0.00010] | 57/0/71 | 0.0641 |
| vbench.background_consistency.clip2clip | 128 | 0.85115 | 0.82613 | 0.02503 | [0.01715, 0.03278] | 94/0/34 | 0.0000 |
| vbench.aesthetic_quality | 128 | 0.60690 | 0.59266 | 0.01424 | [0.00750, 0.02105] | 83/0/45 | 0.0000 |
| vbench.imaging_quality | 128 | 69.18193 | 68.17180 | 1.01013 | [0.18853, 1.87540] | 66/0/62 | 0.0201 |
| vbench.motion_smoothness | 128 | 0.98389 | 0.98567 | -0.00178 | [-0.00301, -0.00066] | 44/0/84 | 0.0018 |
| vbench.temporal_flickering | 128 | 0.96811 | 0.97672 | -0.00860 | [-0.01066, -0.00670] | 21/0/107 | 0.0000 |
| vbench.dynamic_degree | 128 | 0.59583 | 0.43281 | 0.16302 | [0.11302, 0.21302] | 66/49/13 | 0.0000 |
| vbench.background_consistency.mapped_clip2clip | 128 | 0.96843 | 0.96616 | 0.00228 | [0.00166, 0.00290] | 94/0/34 | 0.0000 |
| vbench.overall_consistency | 128 | 0.23622 | 0.23472 | 0.00150 | [-0.00254, 0.00555] | 65/0/63 | 0.4738 |
| vbench.subject_consistency.mapped_clip2clip | 128 | 0.97846 | 0.97648 | 0.00198 | [0.00037, 0.00368] | 83/0/45 | 0.0192 |

## ours_confidence_recent vs deep_forcing

| Metric | n | Candidate | Reference | Raw delta | 95% CI | W/T/L | p |
|---|---:|---:|---:|---:|---:|---:|---:|
| vbench.subject_consistency | 128 | 0.97310 | 0.97090 | 0.00220 | [0.00045, 0.00403] | 67/0/61 | 0.0171 |
| vbench.subject_consistency.inclip | 128 | 0.96773 | 0.96364 | 0.00409 | [0.00153, 0.00677] | 73/0/55 | 0.0023 |
| vbench.subject_consistency.clip2clip | 128 | 0.81544 | 0.81693 | -0.00149 | [-0.01247, 0.01016] | 48/0/80 | 0.7964 |
| vbench.background_consistency | 128 | 0.96390 | 0.96341 | 0.00048 | [-0.00062, 0.00162] | 60/0/68 | 0.4024 |
| vbench.background_consistency.inclip | 128 | 0.95936 | 0.95735 | 0.00201 | [0.00019, 0.00393] | 68/0/60 | 0.0376 |
| vbench.background_consistency.clip2clip | 128 | 0.85115 | 0.85980 | -0.00865 | [-0.01543, -0.00173] | 45/0/83 | 0.0147 |
| vbench.aesthetic_quality | 128 | 0.60690 | 0.60151 | 0.00539 | [0.00025, 0.01071] | 73/0/55 | 0.0470 |
| vbench.imaging_quality | 128 | 69.18193 | 68.36438 | 0.81755 | [0.13968, 1.51334] | 69/0/59 | 0.0216 |
| vbench.motion_smoothness | 128 | 0.98389 | 0.98266 | 0.00123 | [-0.00015, 0.00254] | 69/0/59 | 0.0694 |
| vbench.temporal_flickering | 128 | 0.96811 | 0.96973 | -0.00162 | [-0.00376, 0.00046] | 50/0/78 | 0.1340 |
| vbench.dynamic_degree | 128 | 0.59583 | 0.55521 | 0.04063 | [0.00417, 0.07760] | 37/64/27 | 0.0318 |
| vbench.background_consistency.mapped_clip2clip | 128 | 0.96843 | 0.96948 | -0.00104 | [-0.00161, -0.00047] | 45/0/83 | 0.0003 |
| vbench.overall_consistency | 128 | 0.23622 | 0.23811 | -0.00188 | [-0.00557, 0.00177] | 54/0/74 | 0.3270 |
| vbench.subject_consistency.mapped_clip2clip | 128 | 0.97846 | 0.97815 | 0.00031 | [-0.00116, 0.00186] | 48/0/80 | 0.6999 |

## ours_confidence_recent vs rolling_forcing

| Metric | n | Candidate | Reference | Raw delta | 95% CI | W/T/L | p |
|---|---:|---:|---:|---:|---:|---:|---:|
| vbench.subject_consistency | 128 | 0.97310 | 0.98053 | -0.00743 | [-0.00976, -0.00519] | 44/0/84 | 0.0000 |
| vbench.subject_consistency.inclip | 128 | 0.96773 | 0.97801 | -0.01028 | [-0.01341, -0.00729] | 41/0/87 | 0.0000 |
| vbench.subject_consistency.clip2clip | 128 | 0.81544 | 0.85562 | -0.04018 | [-0.05772, -0.02277] | 45/0/83 | 0.0001 |
| vbench.background_consistency | 128 | 0.96390 | 0.96879 | -0.00490 | [-0.00637, -0.00347] | 43/0/85 | 0.0000 |
| vbench.background_consistency.inclip | 128 | 0.95936 | 0.96524 | -0.00588 | [-0.00814, -0.00367] | 52/0/76 | 0.0000 |
| vbench.background_consistency.clip2clip | 128 | 0.85115 | 0.89733 | -0.04618 | [-0.05790, -0.03488] | 29/0/99 | 0.0000 |
| vbench.aesthetic_quality | 128 | 0.60690 | 0.60902 | -0.00212 | [-0.01080, 0.00648] | 63/0/65 | 0.6327 |
| vbench.imaging_quality | 128 | 69.18193 | 71.05817 | -1.87624 | [-2.88078, -0.93579] | 51/0/77 | 0.0003 |
| vbench.motion_smoothness | 128 | 0.98389 | 0.98802 | -0.00412 | [-0.00605, -0.00243] | 53/0/75 | 0.0000 |
| vbench.temporal_flickering | 128 | 0.96811 | 0.97809 | -0.00997 | [-0.01298, -0.00717] | 36/0/92 | 0.0000 |
| vbench.dynamic_degree | 128 | 0.59583 | 0.34583 | 0.25000 | [0.18906, 0.31198] | 74/39/15 | 0.0000 |
| vbench.background_consistency.mapped_clip2clip | 128 | 0.96843 | 0.97235 | -0.00391 | [-0.00486, -0.00298] | 29/0/99 | 0.0000 |
| vbench.overall_consistency | 128 | 0.23622 | 0.24074 | -0.00452 | [-0.01033, 0.00111] | 62/0/66 | 0.1255 |
| vbench.subject_consistency.mapped_clip2clip | 128 | 0.97846 | 0.98305 | -0.00459 | [-0.00676, -0.00250] | 45/0/83 | 0.0001 |

## ours_confidence_recent vs longlive

| Metric | n | Candidate | Reference | Raw delta | 95% CI | W/T/L | p |
|---|---:|---:|---:|---:|---:|---:|---:|
| vbench.subject_consistency | 128 | 0.97310 | 0.97818 | -0.00509 | [-0.00704, -0.00319] | 39/0/89 | 0.0000 |
| vbench.subject_consistency.inclip | 128 | 0.96773 | 0.97478 | -0.00704 | [-0.00981, -0.00440] | 45/0/83 | 0.0000 |
| vbench.subject_consistency.clip2clip | 128 | 0.81544 | 0.84453 | -0.02909 | [-0.04305, -0.01559] | 41/0/87 | 0.0000 |
| vbench.background_consistency | 128 | 0.96390 | 0.96693 | -0.00303 | [-0.00427, -0.00183] | 41/0/87 | 0.0000 |
| vbench.background_consistency.inclip | 128 | 0.95936 | 0.96433 | -0.00497 | [-0.00706, -0.00296] | 43/0/85 | 0.0000 |
| vbench.background_consistency.clip2clip | 128 | 0.85115 | 0.86745 | -0.01630 | [-0.02455, -0.00834] | 49/0/79 | 0.0001 |
| vbench.aesthetic_quality | 128 | 0.60690 | 0.61090 | -0.00400 | [-0.01244, 0.00430] | 64/0/64 | 0.3523 |
| vbench.imaging_quality | 128 | 69.18193 | 69.31927 | -0.13734 | [-1.02893, 0.73861] | 69/0/59 | 0.7626 |
| vbench.motion_smoothness | 128 | 0.98389 | 0.98694 | -0.00305 | [-0.00473, -0.00154] | 46/0/82 | 0.0002 |
| vbench.temporal_flickering | 128 | 0.96811 | 0.97506 | -0.00695 | [-0.00963, -0.00434] | 33/0/95 | 0.0000 |
| vbench.dynamic_degree | 128 | 0.59583 | 0.41979 | 0.17604 | [0.12500, 0.22813] | 66/46/16 | 0.0000 |
| vbench.background_consistency.mapped_clip2clip | 128 | 0.96843 | 0.96952 | -0.00109 | [-0.00171, -0.00048] | 49/0/79 | 0.0005 |
| vbench.overall_consistency | 128 | 0.23622 | 0.24141 | -0.00519 | [-0.01019, -0.00031] | 58/0/70 | 0.0405 |
| vbench.subject_consistency.mapped_clip2clip | 128 | 0.97846 | 0.98159 | -0.00313 | [-0.00488, -0.00146] | 41/0/87 | 0.0004 |

## ours_prototype_retrieval_motion vs sf_native

| Metric | n | Candidate | Reference | Raw delta | 95% CI | W/T/L | p |
|---|---:|---:|---:|---:|---:|---:|---:|
| vbench.subject_consistency | 128 | 0.97231 | 0.97183 | 0.00048 | [-0.00178, 0.00285] | 62/0/66 | 0.6983 |
| vbench.subject_consistency.inclip | 128 | 0.96677 | 0.96718 | -0.00041 | [-0.00365, 0.00306] | 54/0/74 | 0.8174 |
| vbench.subject_consistency.clip2clip | 128 | 0.80941 | 0.79872 | 0.01069 | [-0.00145, 0.02300] | 83/0/45 | 0.0881 |
| vbench.background_consistency | 128 | 0.96337 | 0.96376 | -0.00039 | [-0.00163, 0.00084] | 66/0/62 | 0.5417 |
| vbench.background_consistency.inclip | 128 | 0.95846 | 0.96137 | -0.00290 | [-0.00503, -0.00082] | 52/0/76 | 0.0074 |
| vbench.background_consistency.clip2clip | 128 | 0.84944 | 0.82613 | 0.02331 | [0.01560, 0.03095] | 93/0/35 | 0.0000 |
| vbench.aesthetic_quality | 128 | 0.60814 | 0.59266 | 0.01548 | [0.00883, 0.02209] | 86/0/42 | 0.0000 |
| vbench.imaging_quality | 128 | 69.04725 | 68.17180 | 0.87545 | [0.04415, 1.73360] | 69/0/59 | 0.0428 |
| vbench.motion_smoothness | 128 | 0.98314 | 0.98567 | -0.00253 | [-0.00394, -0.00126] | 42/0/86 | 0.0001 |
| vbench.temporal_flickering | 128 | 0.96737 | 0.97672 | -0.00934 | [-0.01153, -0.00737] | 20/0/108 | 0.0000 |
| vbench.dynamic_degree | 128 | 0.59948 | 0.43281 | 0.16667 | [0.11615, 0.21823] | 74/44/10 | 0.0000 |
| vbench.background_consistency.mapped_clip2clip | 128 | 0.96828 | 0.96616 | 0.00212 | [0.00152, 0.00274] | 93/0/35 | 0.0000 |
| vbench.overall_consistency | 128 | 0.23775 | 0.23472 | 0.00302 | [-0.00115, 0.00719] | 73/0/55 | 0.1575 |
| vbench.subject_consistency.mapped_clip2clip | 128 | 0.97784 | 0.97648 | 0.00136 | [-0.00032, 0.00313] | 83/0/45 | 0.1266 |

## ours_prototype_retrieval_motion vs deep_forcing

| Metric | n | Candidate | Reference | Raw delta | 95% CI | W/T/L | p |
|---|---:|---:|---:|---:|---:|---:|---:|
| vbench.subject_consistency | 128 | 0.97231 | 0.97090 | 0.00141 | [-0.00037, 0.00325] | 69/0/59 | 0.1322 |
| vbench.subject_consistency.inclip | 128 | 0.96677 | 0.96364 | 0.00313 | [0.00057, 0.00574] | 72/0/56 | 0.0183 |
| vbench.subject_consistency.clip2clip | 128 | 0.80941 | 0.81693 | -0.00753 | [-0.01912, 0.00442] | 51/0/77 | 0.2154 |
| vbench.background_consistency | 128 | 0.96337 | 0.96341 | -0.00004 | [-0.00112, 0.00101] | 58/0/70 | 0.9420 |
| vbench.background_consistency.inclip | 128 | 0.95846 | 0.95735 | 0.00112 | [-0.00066, 0.00285] | 62/0/66 | 0.2178 |
| vbench.background_consistency.clip2clip | 128 | 0.84944 | 0.85980 | -0.01036 | [-0.01746, -0.00313] | 45/0/83 | 0.0048 |
| vbench.aesthetic_quality | 128 | 0.60814 | 0.60151 | 0.00663 | [0.00166, 0.01157] | 70/0/58 | 0.0096 |
| vbench.imaging_quality | 128 | 69.04725 | 68.36438 | 0.68287 | [-0.04158, 1.41910] | 70/0/58 | 0.0677 |
| vbench.motion_smoothness | 128 | 0.98314 | 0.98266 | 0.00048 | [-0.00093, 0.00181] | 65/0/63 | 0.5033 |
| vbench.temporal_flickering | 128 | 0.96737 | 0.96973 | -0.00236 | [-0.00452, -0.00026] | 51/0/77 | 0.0295 |
| vbench.dynamic_degree | 128 | 0.59948 | 0.55521 | 0.04427 | [0.00781, 0.08281] | 44/54/30 | 0.0249 |
| vbench.background_consistency.mapped_clip2clip | 128 | 0.96828 | 0.96948 | -0.00120 | [-0.00181, -0.00059] | 45/0/83 | 0.0002 |
| vbench.overall_consistency | 128 | 0.23775 | 0.23811 | -0.00036 | [-0.00456, 0.00410] | 59/0/69 | 0.8742 |
| vbench.subject_consistency.mapped_clip2clip | 128 | 0.97784 | 0.97815 | -0.00031 | [-0.00180, 0.00124] | 51/0/77 | 0.6960 |

## ours_prototype_retrieval_motion vs rolling_forcing

| Metric | n | Candidate | Reference | Raw delta | 95% CI | W/T/L | p |
|---|---:|---:|---:|---:|---:|---:|---:|
| vbench.subject_consistency | 128 | 0.97231 | 0.98053 | -0.00823 | [-0.01057, -0.00593] | 41/0/87 | 0.0000 |
| vbench.subject_consistency.inclip | 128 | 0.96677 | 0.97801 | -0.01124 | [-0.01438, -0.00818] | 36/0/92 | 0.0000 |
| vbench.subject_consistency.clip2clip | 128 | 0.80941 | 0.85562 | -0.04622 | [-0.06397, -0.02847] | 42/0/86 | 0.0000 |
| vbench.background_consistency | 128 | 0.96337 | 0.96879 | -0.00542 | [-0.00692, -0.00396] | 40/0/88 | 0.0000 |
| vbench.background_consistency.inclip | 128 | 0.95846 | 0.96524 | -0.00678 | [-0.00907, -0.00455] | 43/0/85 | 0.0000 |
| vbench.background_consistency.clip2clip | 128 | 0.84944 | 0.89733 | -0.04789 | [-0.05942, -0.03668] | 29/0/99 | 0.0000 |
| vbench.aesthetic_quality | 128 | 0.60814 | 0.60902 | -0.00088 | [-0.00903, 0.00724] | 64/0/64 | 0.8356 |
| vbench.imaging_quality | 128 | 69.04725 | 71.05817 | -2.01092 | [-2.97222, -1.08959] | 48/0/80 | 0.0000 |
| vbench.motion_smoothness | 128 | 0.98314 | 0.98802 | -0.00487 | [-0.00679, -0.00310] | 43/0/85 | 0.0000 |
| vbench.temporal_flickering | 128 | 0.96737 | 0.97809 | -0.01071 | [-0.01361, -0.00795] | 34/0/94 | 0.0000 |
| vbench.dynamic_degree | 128 | 0.59948 | 0.34583 | 0.25365 | [0.19271, 0.31615] | 82/31/15 | 0.0000 |
| vbench.background_consistency.mapped_clip2clip | 128 | 0.96828 | 0.97235 | -0.00407 | [-0.00505, -0.00311] | 29/0/99 | 0.0000 |
| vbench.overall_consistency | 128 | 0.23775 | 0.24074 | -0.00299 | [-0.00858, 0.00236] | 59/0/69 | 0.2882 |
| vbench.subject_consistency.mapped_clip2clip | 128 | 0.97784 | 0.98305 | -0.00521 | [-0.00728, -0.00316] | 42/0/86 | 0.0000 |

## ours_prototype_retrieval_motion vs longlive

| Metric | n | Candidate | Reference | Raw delta | 95% CI | W/T/L | p |
|---|---:|---:|---:|---:|---:|---:|---:|
| vbench.subject_consistency | 128 | 0.97231 | 0.97818 | -0.00588 | [-0.00779, -0.00401] | 33/0/95 | 0.0000 |
| vbench.subject_consistency.inclip | 128 | 0.96677 | 0.97478 | -0.00801 | [-0.01070, -0.00545] | 36/0/92 | 0.0000 |
| vbench.subject_consistency.clip2clip | 128 | 0.80941 | 0.84453 | -0.03512 | [-0.04935, -0.02137] | 47/0/81 | 0.0000 |
| vbench.background_consistency | 128 | 0.96337 | 0.96693 | -0.00356 | [-0.00479, -0.00236] | 42/0/86 | 0.0000 |
| vbench.background_consistency.inclip | 128 | 0.95846 | 0.96433 | -0.00587 | [-0.00791, -0.00389] | 43/0/85 | 0.0000 |
| vbench.background_consistency.clip2clip | 128 | 0.84944 | 0.86745 | -0.01801 | [-0.02611, -0.00995] | 42/0/86 | 0.0001 |
| vbench.aesthetic_quality | 128 | 0.60814 | 0.61090 | -0.00276 | [-0.01104, 0.00539] | 61/0/67 | 0.5150 |
| vbench.imaging_quality | 128 | 69.04725 | 69.31927 | -0.27202 | [-1.14259, 0.57013] | 64/0/64 | 0.5423 |
| vbench.motion_smoothness | 128 | 0.98314 | 0.98694 | -0.00380 | [-0.00550, -0.00226] | 41/0/87 | 0.0000 |
| vbench.temporal_flickering | 128 | 0.96737 | 0.97506 | -0.00769 | [-0.01027, -0.00517] | 31/0/97 | 0.0000 |
| vbench.dynamic_degree | 128 | 0.59948 | 0.41979 | 0.17969 | [0.13125, 0.22969] | 76/36/16 | 0.0000 |
| vbench.background_consistency.mapped_clip2clip | 128 | 0.96828 | 0.96952 | -0.00124 | [-0.00187, -0.00062] | 42/0/86 | 0.0001 |
| vbench.overall_consistency | 128 | 0.23775 | 0.24141 | -0.00366 | [-0.00844, 0.00087] | 57/0/71 | 0.1296 |
| vbench.subject_consistency.mapped_clip2clip | 128 | 0.97784 | 0.98159 | -0.00375 | [-0.00546, -0.00209] | 47/0/81 | 0.0000 |

## ours_confidence_motion vs sf_native

| Metric | n | Candidate | Reference | Raw delta | 95% CI | W/T/L | p |
|---|---:|---:|---:|---:|---:|---:|---:|
| vbench.subject_consistency | 128 | 0.97209 | 0.97183 | 0.00026 | [-0.00206, 0.00268] | 62/0/66 | 0.8297 |
| vbench.subject_consistency.inclip | 128 | 0.96668 | 0.96718 | -0.00050 | [-0.00373, 0.00297] | 57/0/71 | 0.7715 |
| vbench.subject_consistency.clip2clip | 128 | 0.80625 | 0.79872 | 0.00753 | [-0.00514, 0.02022] | 77/0/51 | 0.2464 |
| vbench.background_consistency | 128 | 0.96306 | 0.96376 | -0.00070 | [-0.00204, 0.00063] | 60/0/68 | 0.3060 |
| vbench.background_consistency.inclip | 128 | 0.95804 | 0.96137 | -0.00332 | [-0.00558, -0.00107] | 49/0/79 | 0.0049 |
| vbench.background_consistency.clip2clip | 128 | 0.84738 | 0.82613 | 0.02125 | [0.01328, 0.02929] | 92/0/36 | 0.0000 |
| vbench.aesthetic_quality | 128 | 0.60654 | 0.59266 | 0.01388 | [0.00698, 0.02074] | 88/0/40 | 0.0002 |
| vbench.imaging_quality | 128 | 69.13735 | 68.17180 | 0.96555 | [0.15876, 1.81389] | 72/0/56 | 0.0252 |
| vbench.motion_smoothness | 128 | 0.98307 | 0.98567 | -0.00260 | [-0.00390, -0.00140] | 37/0/91 | 0.0001 |
| vbench.temporal_flickering | 128 | 0.96688 | 0.97672 | -0.00984 | [-0.01197, -0.00785] | 19/0/109 | 0.0000 |
| vbench.dynamic_degree | 128 | 0.62187 | 0.43281 | 0.18906 | [0.13750, 0.24115] | 77/41/10 | 0.0000 |
| vbench.background_consistency.mapped_clip2clip | 128 | 0.96808 | 0.96616 | 0.00192 | [0.00128, 0.00255] | 92/0/36 | 0.0000 |
| vbench.overall_consistency | 128 | 0.23694 | 0.23472 | 0.00222 | [-0.00131, 0.00590] | 64/0/64 | 0.2371 |
| vbench.subject_consistency.mapped_clip2clip | 128 | 0.97751 | 0.97648 | 0.00103 | [-0.00070, 0.00281] | 77/0/51 | 0.2556 |

## ours_confidence_motion vs deep_forcing

| Metric | n | Candidate | Reference | Raw delta | 95% CI | W/T/L | p |
|---|---:|---:|---:|---:|---:|---:|---:|
| vbench.subject_consistency | 128 | 0.97209 | 0.97090 | 0.00120 | [-0.00061, 0.00309] | 62/0/66 | 0.2080 |
| vbench.subject_consistency.inclip | 128 | 0.96668 | 0.96364 | 0.00304 | [0.00056, 0.00567] | 76/0/52 | 0.0208 |
| vbench.subject_consistency.clip2clip | 128 | 0.80625 | 0.81693 | -0.01068 | [-0.02289, 0.00154] | 47/0/81 | 0.0898 |
| vbench.background_consistency | 128 | 0.96306 | 0.96341 | -0.00035 | [-0.00151, 0.00079] | 57/0/71 | 0.5501 |
| vbench.background_consistency.inclip | 128 | 0.95804 | 0.95735 | 0.00070 | [-0.00118, 0.00255] | 64/0/64 | 0.4634 |
| vbench.background_consistency.clip2clip | 128 | 0.84738 | 0.85980 | -0.01242 | [-0.01978, -0.00506] | 47/0/81 | 0.0011 |
| vbench.aesthetic_quality | 128 | 0.60654 | 0.60151 | 0.00503 | [-0.00022, 0.01030] | 68/0/60 | 0.0636 |
| vbench.imaging_quality | 128 | 69.13735 | 68.36438 | 0.77297 | [0.07236, 1.48451] | 72/0/56 | 0.0332 |
| vbench.motion_smoothness | 128 | 0.98307 | 0.98266 | 0.00041 | [-0.00091, 0.00166] | 65/0/63 | 0.5441 |
| vbench.temporal_flickering | 128 | 0.96688 | 0.96973 | -0.00285 | [-0.00493, -0.00086] | 47/0/81 | 0.0062 |
| vbench.dynamic_degree | 128 | 0.62187 | 0.55521 | 0.06667 | [0.02656, 0.10885] | 50/55/23 | 0.0013 |
| vbench.background_consistency.mapped_clip2clip | 128 | 0.96808 | 0.96948 | -0.00140 | [-0.00204, -0.00078] | 47/0/81 | 0.0000 |
| vbench.overall_consistency | 128 | 0.23694 | 0.23811 | -0.00116 | [-0.00501, 0.00297] | 52/0/76 | 0.5748 |
| vbench.subject_consistency.mapped_clip2clip | 128 | 0.97751 | 0.97815 | -0.00064 | [-0.00218, 0.00096] | 47/0/81 | 0.4254 |

## ours_confidence_motion vs rolling_forcing

| Metric | n | Candidate | Reference | Raw delta | 95% CI | W/T/L | p |
|---|---:|---:|---:|---:|---:|---:|---:|
| vbench.subject_consistency | 128 | 0.97209 | 0.98053 | -0.00844 | [-0.01085, -0.00613] | 34/0/94 | 0.0000 |
| vbench.subject_consistency.inclip | 128 | 0.96668 | 0.97801 | -0.01133 | [-0.01445, -0.00838] | 33/0/95 | 0.0000 |
| vbench.subject_consistency.clip2clip | 128 | 0.80625 | 0.85562 | -0.04937 | [-0.06749, -0.03138] | 41/0/87 | 0.0000 |
| vbench.background_consistency | 128 | 0.96306 | 0.96879 | -0.00573 | [-0.00729, -0.00421] | 40/0/88 | 0.0000 |
| vbench.background_consistency.inclip | 128 | 0.95804 | 0.96524 | -0.00720 | [-0.00962, -0.00484] | 47/0/81 | 0.0000 |
| vbench.background_consistency.clip2clip | 128 | 0.84738 | 0.89733 | -0.04995 | [-0.06164, -0.03855] | 27/0/101 | 0.0000 |
| vbench.aesthetic_quality | 128 | 0.60654 | 0.60902 | -0.00248 | [-0.01067, 0.00564] | 65/0/63 | 0.5503 |
| vbench.imaging_quality | 128 | 69.13735 | 71.05817 | -1.92082 | [-2.90807, -0.96075] | 51/0/77 | 0.0001 |
| vbench.motion_smoothness | 128 | 0.98307 | 0.98802 | -0.00494 | [-0.00682, -0.00327] | 41/0/87 | 0.0000 |
| vbench.temporal_flickering | 128 | 0.96688 | 0.97809 | -0.01121 | [-0.01406, -0.00853] | 32/0/96 | 0.0000 |
| vbench.dynamic_degree | 128 | 0.62187 | 0.34583 | 0.27604 | [0.21302, 0.34010] | 82/34/12 | 0.0000 |
| vbench.background_consistency.mapped_clip2clip | 128 | 0.96808 | 0.97235 | -0.00427 | [-0.00526, -0.00330] | 27/0/101 | 0.0000 |
| vbench.overall_consistency | 128 | 0.23694 | 0.24074 | -0.00380 | [-0.00904, 0.00134] | 57/0/71 | 0.1561 |
| vbench.subject_consistency.mapped_clip2clip | 128 | 0.97751 | 0.98305 | -0.00554 | [-0.00769, -0.00341] | 41/0/87 | 0.0000 |

## ours_confidence_motion vs longlive

| Metric | n | Candidate | Reference | Raw delta | 95% CI | W/T/L | p |
|---|---:|---:|---:|---:|---:|---:|---:|
| vbench.subject_consistency | 128 | 0.97209 | 0.97818 | -0.00609 | [-0.00799, -0.00425] | 31/0/97 | 0.0000 |
| vbench.subject_consistency.inclip | 128 | 0.96668 | 0.97478 | -0.00810 | [-0.01065, -0.00564] | 36/0/92 | 0.0000 |
| vbench.subject_consistency.clip2clip | 128 | 0.80625 | 0.84453 | -0.03828 | [-0.05238, -0.02416] | 37/0/91 | 0.0000 |
| vbench.background_consistency | 128 | 0.96306 | 0.96693 | -0.00387 | [-0.00513, -0.00264] | 34/0/94 | 0.0000 |
| vbench.background_consistency.inclip | 128 | 0.95804 | 0.96433 | -0.00629 | [-0.00841, -0.00419] | 39/0/89 | 0.0000 |
| vbench.background_consistency.clip2clip | 128 | 0.84738 | 0.86745 | -0.02007 | [-0.02792, -0.01235] | 40/0/88 | 0.0000 |
| vbench.aesthetic_quality | 128 | 0.60654 | 0.61090 | -0.00436 | [-0.01231, 0.00366] | 58/0/70 | 0.2861 |
| vbench.imaging_quality | 128 | 69.13735 | 69.31927 | -0.18192 | [-1.05289, 0.66338] | 69/0/59 | 0.6834 |
| vbench.motion_smoothness | 128 | 0.98307 | 0.98694 | -0.00387 | [-0.00545, -0.00243] | 40/0/88 | 0.0000 |
| vbench.temporal_flickering | 128 | 0.96688 | 0.97506 | -0.00818 | [-0.01060, -0.00583] | 27/0/101 | 0.0000 |
| vbench.dynamic_degree | 128 | 0.62187 | 0.41979 | 0.20208 | [0.15208, 0.25417] | 84/30/14 | 0.0000 |
| vbench.background_consistency.mapped_clip2clip | 128 | 0.96808 | 0.96952 | -0.00145 | [-0.00207, -0.00083] | 40/0/88 | 0.0000 |
| vbench.overall_consistency | 128 | 0.23694 | 0.24141 | -0.00447 | [-0.00856, -0.00047] | 54/0/74 | 0.0338 |
| vbench.subject_consistency.mapped_clip2clip | 128 | 0.97751 | 0.98159 | -0.00408 | [-0.00579, -0.00239] | 37/0/91 | 0.0000 |
