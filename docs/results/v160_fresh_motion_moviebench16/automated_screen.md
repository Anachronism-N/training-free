# v160 Automated Diagnostic Screen

This report is for failure triage and adaptive review selection. It is not a paper metric or a promotion gate.

Automatic safety screen: **PASS**

## Flagged prompts

- Prompt 12: subject_consistency_drop
- Prompt 13: late_motion_collapse

## Wave1

- Prompt 12: highest_automatic_risk (tags=object_identity,tracking_camera,long_motion,abandoned_street)
- Prompt 7: largest_predicted_gain (tags=articulated_motion,festival,crowd,colorful)
- Prompt 1: largest_metric_disagreement (tags=multi_subject,animal,snow,camera_depth)
- Prompt 10: typical_case (tags=child_identity,bicycle,season_change,scene_evolution)

## Wave2

- Prompt 13: highest_automatic_risk (tags=fpv,scene_transition,rapid_camera,interior)
- Prompt 3: largest_predicted_gain (tags=multi_object,miniature_scale,water_motion,photorealistic)
- Prompt 6: largest_metric_disagreement (tags=vehicle,fast_motion,dust,tracking_camera)
- Prompt 15: typical_case (tags=transformation,animal_identity,lightning,scene_event)

Wave 1 contains 12 videos (4 prompts x 3 methods). Prepare Wave 2 only if the prespecified human decision is inconclusive.
