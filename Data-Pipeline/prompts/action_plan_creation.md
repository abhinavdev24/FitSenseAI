ACTION: plan_creation
Create a structured workout plan. The plan contains named days, each day contains ordered exercises, each exercise contains individual sets.

RESPOND WITH:
{
  "message": "<conversational summary>",
  "data": {
    "name": "<plan name string>",
    "days": [
      {
        "name": "<day label, e.g. PUSH_1, PULL_2, UPPER_A>",
        "day_order": <int, position in cycle starting at 1>,
        "notes": "<optional string or null>",
        "exercises": [
          {
            "exercise_name": "<string>",
            "position": <int, order within this day starting at 1>,
            "notes": "<optional string or null>",
            "sets": [
              {
                "set_number": <int, starting at 1>,
                "target_reps": <int>,
                "target_weight_kg": <number>,
                "target_rir": <int, reps in reserve>,
                "rest_seconds": <int>
              }
            ]
          }
        ]
      }
    ]
  }
}

RULES:
- Every exercise must have at least 1 set defined.
- set_number must be sequential starting at 1 within each exercise.
- day_order must be sequential starting at 1.
- position must be sequential starting at 1 within each day.
- target_rir must be >= 1 if user has medical conditions.
- target_weight_kg uses decimal (e.g. 22.5), never strings.
