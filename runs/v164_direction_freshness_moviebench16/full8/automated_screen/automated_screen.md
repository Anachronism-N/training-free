# v164 Direction/Freshness Automated Screen

This report is for failure triage and adaptive review selection. It is not a paper metric or a promotion gate.

Automatic safety screen: **PASS**

## Flagged prompts

- Prompt 8: background_drift

## Wave1

- Prompt 14: highest_automatic_risk (tags=human_identity,anime,ship,camera_facing)
- Prompt 5: largest_predicted_gain (tags=rotating_camera,indoor,many_objects,screen_content)
- Prompt 3: largest_metric_disagreement (tags=multi_object,miniature_scale,water_motion,photorealistic)
- Prompt 6: typical_case (tags=vehicle,fast_motion,dust,tracking_camera)

## Wave2

- Prompt 13: highest_automatic_risk (tags=fpv,scene_transition,rapid_camera,interior)
- Prompt 9: largest_predicted_gain (tags=two_subjects,animal_identity,running,neon_city)
- Prompt 2: largest_metric_disagreement (tags=3d_animation,character_identity,object_interaction,fire)
- Prompt 11: typical_case (tags=vehicle,high_speed,turning,dynamic_background)

Wave 1 contains 12 videos (4 prompts x 3 methods). Prepare Wave 2 only if the prespecified human decision is inconclusive.
