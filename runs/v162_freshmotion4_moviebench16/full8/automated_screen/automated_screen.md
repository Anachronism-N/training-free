# v162 Automated Diagnostic Screen

This report is for failure triage and adaptive review selection. It is not a paper metric or a promotion gate.

Automatic safety screen: **FLAGGED**

## Flagged prompts

- Prompt 0: background_drift
- Prompt 1: subject_consistency_drop, background_drift
- Prompt 4: late_motion_collapse
- Prompt 5: temporal_discontinuity, background_drift
- Prompt 8: background_drift
- Prompt 9: background_drift
- Prompt 10: background_drift
- Prompt 13: late_motion_collapse

## Wave1

- Prompt 14: highest_automatic_risk (tags=human_identity,anime,ship,camera_facing)
- Prompt 2: largest_predicted_gain (tags=3d_animation,character_identity,object_interaction,fire)
- Prompt 9: largest_metric_disagreement (tags=two_subjects,animal_identity,running,neon_city)
- Prompt 13: typical_case (tags=fpv,scene_transition,rapid_camera,interior)

## Wave2

- Prompt 6: highest_automatic_risk (tags=vehicle,fast_motion,dust,tracking_camera)
- Prompt 3: largest_predicted_gain (tags=multi_object,miniature_scale,water_motion,photorealistic)
- Prompt 5: largest_metric_disagreement (tags=rotating_camera,indoor,many_objects,screen_content)
- Prompt 0: typical_case (tags=human_identity,walking,crowd,night_city)

Wave 1 contains 12 videos (4 prompts x 3 methods). Prepare Wave 2 only if the prespecified human decision is inconclusive.
