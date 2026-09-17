# v210 LPHC Screen8 results

- Generation commit: `789d860403c58449efb65067bd034ad2e6212d5d`
- Evaluation commit: `08fe9d72d4eb73130f7e65ae4f1a48d7cf02a64d`
- Generation root: `/apdcephfs_gy2/share_302533218/cedricnie/v210_runs/v210_lphc_789d8604_sixnode`
- Evaluation root: `/apdcephfs_gy2/share_302533218/cedricnie/v210_runs/v210_eval_08fe9d72_screen8`
- Completion: Gate0 4/4, smoke 1/1, Screen8 64/64, VBench core-9 72/72 jobs, temporal diagnostics 64/64 videos.
- Frozen recommendation: `stop_v210_no_eligible_lphc_candidate`.

The LPHC candidates improved mean quality excluding Dynamic Degree by approximately `+0.138` to `+0.234` over `sf_fifo21`, but none satisfied the pre-registered non-inferiority and temporal safety gates against both `sf_fifo21` and `sf_sink1_21`.

`generation/` contains the frozen input manifest and completion receipts. `evaluation/` contains aggregate reports, paired comparisons, VBench summaries, temporal diagnostics, the comparison manifest, and six-node split provenance. MP4, clips, model files, raw traces, and tensor dumps are excluded.
