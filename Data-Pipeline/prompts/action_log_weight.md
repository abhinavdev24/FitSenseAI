ACTION: log_weight
Log a body weight measurement.

RESPOND WITH:
{
  "message": "<conversational summary>",
  "data": {
    "weight_kg": <number>,
    "body_fat_percentage": <number or null>,
    "logged_at": "<ISO 8601 datetime, e.g. 2025-03-08T07:30:00Z>",
    "notes": "<optional string or null>"
  }
}

RULES:
- weight_kg is required, must be a positive decimal.
- body_fat_percentage is null unless user explicitly provides it.
- logged_at defaults to current date/time if user does not specify.
- Convert lbs to kg if user provides pounds (1 lb = 0.4536 kg, round to 1 decimal).
