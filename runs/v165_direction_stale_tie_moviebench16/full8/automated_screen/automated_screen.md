# v165 Direction Stale-Tie Automated Screen

This report is for failure triage and adaptive review selection. It is not a paper metric or a promotion gate.

Automatic safety screen: **PASS**

## Flagged prompts

- Prompt 6: subject_consistency_drop
- Prompt 10: background_drift

## Wave1

- Prompt 14: highest_automatic_risk (tags=human_identity,anime,ship,camera_facing)
- Prompt 7: largest_predicted_gain (tags=articulated_motion,festival,crowd,colorful)
- Prompt 5: largest_metric_disagreement (tags=rotating_camera,indoor,many_objects,screen_content)
- Prompt 12: typical_case (tags=object_identity,tracking_camera,long_motion,abandoned_street)

## Wave2

- Prompt 10: highest_automatic_risk (tags=child_identity,bicycle,season_change,scene_evolution)
- Prompt 6: largest_predicted_gain (tags=vehicle,fast_motion,dust,tracking_camera)
- Prompt 3: largest_metric_disagreement (tags=multi_object,miniature_scale,water_motion,photorealistic)
- Prompt 11: typical_case (tags=vehicle,high_speed,turning,dynamic_background)

Wave 1 contains 12 videos (4 prompts x 3 methods). Prepare Wave 2 only if the prespecified human decision is inconclusive.
