# Invalid for head selection

The v176 profile is complete but violates its fair-teacher contract. Its 16
runtime logs contain 4,668 candidate/teacher subset violations (768 Coverage
and 3,900 Episode). The uploaded runtime downgraded the required assertion to
a warning and then recorded the verification flag as true.

Consequently, `analysis/analysis.json` and its four reported Coverage heads
must not be used for generation, tables, or claims. The raw artifacts are kept
only for debugging and provenance. See
`docs/195_v176_result_audit_and_v177_strict_recovery.md`; v177 is the corrected
experiment.
