ACTION: plan_modification
Modify an existing workout plan. The current plan state is provided in the user message.

RESPOND WITH:
{
  "message": "<conversational summary>",
  "data": {
    "modifications": [
      {
        "action": "add | remove | update",
        "day_name": "<which plan day this targets, e.g. PUSH_1>",
        "exercise_name": "<string>",
        "position": <int or null, required for add>,
        "updates": {
          "sets": [
            {
              "set_number": <int>,
              "target_reps": <int>,
              "target_weight_kg": <number>,
              "target_rir": <int>,
              "rest_seconds": <int>
            }
          ],
          "notes": "<optional string or null>"
        }
      }
    ]
  }
}

RULES:
- action is one of: add, remove, update.
- For remove: only day_name, exercise_name, and action are required. updates can be null.
- For add: position and full sets array are required.
- For update: include only the fields being changed inside updates.
- Do not rename exercises — use remove + add instead.
