# v161 Automated Diagnostic Screen

This report is for failure triage and adaptive review selection. It is not a paper metric or a promotion gate.

Automatic safety screen: **FLAGGED**

## Flagged prompts

- Prompt 7: background_drift
- Prompt 11: subject_consistency_drop, background_drift
- Prompt 12: late_motion_collapse, temporal_discontinuity

## Wave1

- Prompt 12: highest_automatic_risk (tags=object_identity,tracking_camera,long_motion,abandoned_street)
- Prompt 3: largest_predicted_gain (tags=multi_object,miniature_scale,water_motion,photorealistic)
- Prompt 6: largest_metric_disagreement (tags=vehicle,fast_motion,dust,tracking_camera)
- Prompt 8: typical_case (tags=human_motion,running,cinematic,step_printing)

## Wave2

- Prompt 11: highest_automatic_risk (tags=vehicle,high_speed,turning,dynamic_background)
- Prompt 5: largest_predicted_gain (tags=rotating_camera,indoor,many_objects,screen_content)
- Prompt 13: largest_metric_disagreement (tags=fpv,scene_transition,rapid_camera,interior)
- Prompt 15: typical_case (tags=transformation,animal_identity,lightning,scene_event)

Wave 1 contains 12 videos (4 prompts x 3 methods). Prepare Wave 2 only if the prespecified human decision is inconclusive.
