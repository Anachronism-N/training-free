# v166 Multi-scale Motion Automated Screen

This report is for failure triage and adaptive review selection. It is not a paper metric or a promotion gate.

Automatic safety screen: **FLAGGED**

## Flagged prompts

- Prompt 5: background_drift
- Prompt 6: background_drift
- Prompt 8: background_drift
- Prompt 10: background_drift

## Wave1

- Prompt 14: highest_automatic_risk (tags=human_identity,anime,ship,camera_facing)
- Prompt 3: largest_predicted_gain (tags=multi_object,miniature_scale,water_motion,photorealistic)
- Prompt 9: largest_metric_disagreement (tags=two_subjects,animal_identity,running,neon_city)
- Prompt 13: typical_case (tags=fpv,scene_transition,rapid_camera,interior)

## Wave2

- Prompt 11: highest_automatic_risk (tags=vehicle,high_speed,turning,dynamic_background)
- Prompt 5: largest_predicted_gain (tags=rotating_camera,indoor,many_objects,screen_content)
- Prompt 2: largest_metric_disagreement (tags=3d_animation,character_identity,object_interaction,fire)
- Prompt 15: typical_case (tags=transformation,animal_identity,lightning,scene_event)

Wave 1 contains 12 videos (4 prompts x 3 methods). Prepare Wave 2 only if the prespecified human decision is inconclusive.
