# v144 Context-conditioned Head-role Audit

| axis | residual split rho | residual context rho | layer eta2 | persistent top heads | interpretation |
|---|---:|---:|---:|---:|---|
| prompt_history_excess | 0.1664 | 0.1511 | 0.1668 | 32 | layer_or_state_conditioned |
| policy_prompt_score | 0.0972 | 0.1017 | 0.4094 | 20 | layer_or_state_conditioned |
| stale_a_mass | 0.5244 | -0.1078 | 0.0718 | 130 | layer_or_state_conditioned |
| persistent_content | 0.1089 | 0.2882 | 0.3151 | 54 | layer_or_state_conditioned |
| persistent_positioned | 0.1101 | 0.2745 | 0.2308 | 51 | layer_or_state_conditioned |
| persistent_output | 0.0988 | 0.2096 | 0.1943 | 45 | layer_or_state_conditioned |

Passing this audit permits the phrase `static head candidate`, not a functional role claim. Failed axes should be modeled as layer/timestep/episode-conditioned routing signals.
