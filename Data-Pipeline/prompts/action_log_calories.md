ACTION: log_calories
Log daily calorie intake.

RESPOND WITH:
{
  "message": "<conversational summary>",
  "data": {
    "log_date": "<YYYY-MM-DD>",
    "calories_consumed": <int>,
    "notes": "<optional string or null>"
  }
}

RULES:
- calories_consumed must be a positive integer.
- log_date defaults to today if user does not specify.
