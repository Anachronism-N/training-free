# v138 History Intervention Analysis

- Recommendation: `history_specificity_only`
- Maximum RoPE reconstruction error: `0.00680272`
- Maximum RoPE reconstruction RMS error: `3.01887e-05`
- Maximum recent-value preservation error: `0`
- History-specificity gate: `True`
- Order-axis gate: `False`

## Specificity

- `self_history_specific`: 201
- `no_self_history_preference`: 159
- Split-half Spearman: `0.9711`
- Bootstrap-reliable fraction: `0.9472`

## Order Intervention

- Reverse split-half Spearman: `0.9980`
- GMM BIC1-BIC2 / BIC3-BIC2: `-2.2159` / `12.0859`
- GMM threshold: `-1.042161`

## Evidence Boundary

- Cross-video specificity compares self history with unrelated and lexically similar wrong trajectories; it is not yet an identity-versus-scene decomposition.
- Reverse/phase/freeze interventions reposition cached layer features with corrected RoPE. They measure attention-level history sensitivity, not final-prediction causal utility.
- A generation map must not be constructed unless its gate passes and grouped top/bottom/random/reversed causal controls are run.
