# v167 State-conditioned Motion Automated Screen

This report is for failure triage and adaptive review selection. It is not a paper metric or a promotion gate.

Automatic safety screen: **PASS**

## Flagged prompts

- Prompt 1: background_drift

## Wave1

- Prompt 3: highest_automatic_risk (tags=multi_object,miniature_scale,water_motion,photorealistic)
- Prompt 14: largest_predicted_gain (tags=human_identity,anime,ship,camera_facing)
- Prompt 5: largest_metric_disagreement (tags=rotating_camera,indoor,many_objects,screen_content)
- Prompt 6: typical_case (tags=vehicle,fast_motion,dust,tracking_camera)

## Wave2

- Prompt 1: highest_automatic_risk (tags=multi_subject,animal,snow,camera_depth)
- Prompt 2: largest_predicted_gain (tags=3d_animation,character_identity,object_interaction,fire)
- Prompt 9: largest_metric_disagreement (tags=two_subjects,animal_identity,running,neon_city)
- Prompt 12: typical_case (tags=object_identity,tracking_camera,long_motion,abandoned_street)

Wave 1 contains 12 videos (4 prompts x 3 methods). Prepare Wave 2 only if the prespecified human decision is inconclusive.
