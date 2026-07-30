# v143 Multi-axis Profile Summary

- Natural profiles: `128`
- A-B profiles: `32`
- A-B switch frame: `57`
- Natural policy split agreement: `0.9333`
- A-B prompt/history split rho: `0.4475`
- A-B persistent-A split rho: `0.2903`
- Correctness gate: `True`

The natural temporal axes cover the native 21-frame Self-Forcing window.
The A-B experiment measures scene plasticity and stale-A suppression. The
per-context tables separate switch type, episode, frame, and denoising-state
dependence instead of assuming a static head identity. Persistent A recall
after an intervening episode remains an A-B-A generation question.
