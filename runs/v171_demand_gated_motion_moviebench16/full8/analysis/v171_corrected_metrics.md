# v171 Corrected VBench Decision

Mechanism gate: **True**
Recommendation: **reject_both_without_manual_review**
Selected candidate: `None`
Manual review requested: **False**

| Method | Quality | Identity/background | Temporal mechanics | Semantic | Visual | Dynamic |
|---|---:|---:|---:|---:|---:|---:|
| ours_middle10_reservoir2_multiscalemotion1 | 84.3425 | 0.968494 | 0.971138 | 0.234875 | 0.669263 | 0.770833 |
| ours_middle10_reservoir2_deficitquery1 | 84.3242 | 0.968713 | 0.970967 | 0.236358 | 0.671004 | 0.762500 |
| ours_middle10_reservoir2_deficitbaseline1 | 84.1826 | 0.967836 | 0.969964 | 0.235860 | 0.666335 | 0.779167 |

A candidate reaches matched confirmation only with a passing mechanism gate, nonnegative Quality, positive Dynamic Degree, and identity/temporal losses no larger than v170 same-policy replica noise. Strict nonnegative checks are reported separately.

This 16-prompt suite is adaptive development evidence. A passing result still requires matched and held-out confirmation.
