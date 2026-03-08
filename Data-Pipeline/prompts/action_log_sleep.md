ACTION: log_sleep
Log sleep duration.

RESPOND WITH:
{
  "message": "<conversational summary>",
  "data": {
    "log_date": "<YYYY-MM-DD>",
    "sleep_duration_hours": <decimal>,
    "notes": "<optional string or null>"
  }
}

RULES:
- sleep_duration_hours is a decimal (e.g. 7.5 for 7 hours 30 minutes).
- log_date defaults to today if user does not specify.
- If user says "slept from 11pm to 6:30am", calculate 7.5.
