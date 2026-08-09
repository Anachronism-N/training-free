# v168 Cross-scale Consensus Automated Screen

This report is for failure triage and adaptive review selection. It is not a paper metric or a promotion gate.

Automatic safety screen: **FLAGGED**

## Flagged prompts

- Prompt 4: background_drift
- Prompt 10: background_drift
- Prompt 14: background_drift

## Wave1

- Prompt 3: highest_automatic_risk (tags=multi_object,miniature_scale,water_motion,photorealistic)
- Prompt 9: largest_predicted_gain (tags=two_subjects,animal_identity,running,neon_city)
- Prompt 2: largest_metric_disagreement (tags=3d_animation,character_identity,object_interaction,fire)
- Prompt 1: typical_case (tags=multi_subject,animal,snow,camera_depth)

## Wave2

- Prompt 14: highest_automatic_risk (tags=human_identity,anime,ship,camera_facing)
- Prompt 11: largest_predicted_gain (tags=vehicle,high_speed,turning,dynamic_background)
- Prompt 5: largest_metric_disagreement (tags=rotating_camera,indoor,many_objects,screen_content)
- Prompt 7: typical_case (tags=articulated_motion,festival,crowd,colorful)

Wave 1 contains 12 videos (4 prompts x 3 methods). Prepare Wave 2 only if the prespecified human decision is inconclusive.
