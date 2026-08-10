# v169 Soft Cross-scale Automated Screen

This report is for failure triage and adaptive review selection. It is not a paper metric or a promotion gate.

Automatic safety screen: **PASS**

## Flagged prompts

- Prompt 3: background_drift
- Prompt 11: background_drift

## Wave1

- Prompt 5: highest_automatic_risk (tags=rotating_camera,indoor,many_objects,screen_content)
- Prompt 3: largest_predicted_gain (tags=multi_object,miniature_scale,water_motion,photorealistic)
- Prompt 14: largest_metric_disagreement (tags=human_identity,anime,ship,camera_facing)
- Prompt 7: typical_case (tags=articulated_motion,festival,crowd,colorful)

## Wave2

- Prompt 2: highest_automatic_risk (tags=3d_animation,character_identity,object_interaction,fire)
- Prompt 11: largest_predicted_gain (tags=vehicle,high_speed,turning,dynamic_background)
- Prompt 9: largest_metric_disagreement (tags=two_subjects,animal_identity,running,neon_city)
- Prompt 4: typical_case (tags=multi_person,mobile_video,urban,documentary)

Wave 1 contains 12 videos (4 prompts x 3 methods). Prepare Wave 2 only if the prespecified human decision is inconclusive.
