ACTION: workout_logging
Parse the user's natural language workout log into structured data matching the database schema.

RESPOND WITH:
{
  "message": "<conversational summary>",
  "data": {
    "plan_day_name": "<string or null, e.g. PUSH_1 if following a plan>",
    "notes": "<optional session-level notes or null>",
    "exercises": [
      {
        "exercise_name": "<string>",
        "position": <int, order performed starting at 1>,
        "notes": "<optional string or null>",
        "sets": [
          {
            "set_number": <int, starting at 1>,
            "reps": <int>,
            "weight_kg": <number>,
            "rir": <int or null>,
            "is_warmup": <boolean>
          }
        ]
      }
    ]
  }
}

RULES:
- set_number sequential starting at 1 per exercise.
- position sequential starting at 1.
- weight_kg is decimal, never a string.
- Default is_warmup to false unless user explicitly mentions warmup sets.
- Default rir to null if user does not mention it.
- If user says "3x10 at 60kg", expand to 3 individual set objects.
