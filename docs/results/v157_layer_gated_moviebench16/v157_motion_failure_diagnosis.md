# v157 Motion-Failure Diagnosis

Mean primary - all-reservoir motion: **-0.3125**.

Motion deficits are localized to prompts `[0, 1, 5, 6, 7, 15]`; the other 10 prompts are ties and none favors the primary.

| Method | Severe prompt indices |
|---|---|
| `ours_layer_interleaved10_reservoir4` | `[0, 6, 8, 15]` |
| `ours_layer_middle10_reservoir4` | `[4, 6, 15]` |
| `ours_all_reservoir4_reference` | `[4, 6, 10]` |
| `ours_all_recent8_reference` | `[0, 5, 6, 8, 10, 15]` |

Shared hard prompts: `[0, 4, 6, 8, 10, 15]`. Primary-specific severe prompts relative to all-reservoir: `[0, 8, 15]`.

At fixed sink1+middle4+recent4 budget, replace two random reservoir frames with one semantically coherent adjacent motion pair; keep layer membership fixed.

The prompt subset is diagnosed after viewing v157 ratings. v159 therefore keeps all 16 prompts and is exploratory.
