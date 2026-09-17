# v210 LPHC Development Screen8

- Recommendation: `stop_v210_no_eligible_lphc_candidate`
- Selected for matched-random extension: `None`
- Provenance gate: `True`
- Dynamic Degree used for selection: `False`
- Significance used for selection: `False`
- Paper claim ready: `False`

## Provenance checks

- `claim_boundary`: `True`
- `dense_prompt_indices`: `True`
- `development_only`: `True`
- `dimensions`: `True`
- `effective_seeds`: `True`
- `evaluation_runtime_provenance`: `True`
- `experiment`: `True`
- `frame_count`: `True`
- `generation_times`: `True`
- `method_order`: `True`
- `pass`: `True`
- `primary_controls`: `True`
- `prompt_count`: `True`
- `read_only_materialization`: `True`
- `source_indices`: `True`
- `source_media_audit`: `True`
- `source_provenance`: `True`
- `split_provenance`: `True`
- `summary_complete`: `True`
- `summary_dimensions`: `True`
- `summary_experiment`: `True`
- `summary_methods`: `True`
- `temporal_contract`: `True`
- `vbench_part_provenance`: `True`
- `vbench_runtime_provenance`: `True`

| Candidate | Provenance | Dual SF21 NI | Dual SF21 positive | Temporal safe | Eligible |
|---|---:|---:|---:|---:|---:|
| lphc_e1_a002_correct | True | False | True | False | False |
| lphc_e1_a005_correct | True | False | True | False | False |
| lphc_e1_a010_correct | True | False | True | False | False |
| lphc_e2_a010_correct | True | False | True | False | False |
| lphc_full_a010_correct | True | False | True | False | False |

| Rank | Candidate | Full quality w/o DD | Full subject consistency | Generation seconds |
|---:|---|---:|---:|---:|

The eight-prompt screen is a development prefilter only. Bootstrap intervals, sign-test p-values, and BH q-values are descriptive and do not support a paper claim. Dynamic Degree and the sf_fifo25 capacity context do not affect selection. At most one method may advance to a separate matched-random extension.
