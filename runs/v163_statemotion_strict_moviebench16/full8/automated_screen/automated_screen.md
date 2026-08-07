# v163 Automated Diagnostic Screen

This report is for failure triage and adaptive review selection. It is not a paper metric or a promotion gate.

Automatic safety screen: **FLAGGED**

## Flagged prompts

- Prompt 0: background_drift
- Prompt 8: background_drift
- Prompt 9: background_drift
- Prompt 11: background_drift
- Prompt 12: background_drift
- Prompt 13: temporal_discontinuity
- Prompt 15: background_drift

## Wave1

- Prompt 13: highest_automatic_risk (tags=fpv,scene_transition,rapid_camera,interior)
- Prompt 7: largest_predicted_gain (tags=articulated_motion,festival,crowd,colorful)
- Prompt 6: largest_metric_disagreement (tags=vehicle,fast_motion,dust,tracking_camera)
- Prompt 4: typical_case (tags=multi_person,mobile_video,urban,documentary)

## Wave2

- Prompt 15: highest_automatic_risk (tags=transformation,animal_identity,lightning,scene_event)
- Prompt 12: largest_predicted_gain (tags=object_identity,tracking_camera,long_motion,abandoned_street)
- Prompt 14: largest_metric_disagreement (tags=human_identity,anime,ship,camera_facing)
- Prompt 5: typical_case (tags=rotating_camera,indoor,many_objects,screen_content)

Wave 1 contains 12 videos (4 prompts x 3 methods). Prepare Wave 2 only if the prespecified human decision is inconclusive.
