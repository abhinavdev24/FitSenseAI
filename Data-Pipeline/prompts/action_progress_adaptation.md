ACTION: progress_adaptation
Analyze the user's recent workout history and metrics to recommend plan adjustments.
The user's recent data is provided in the user message.

RESPOND WITH:
{
  "message": "<conversational summary with specific recommendations>",
  "data": {
    "volume_trend": "improving | stable | declining",
    "strength_trend": "improving | stable | declining",
    "adherence_rate": <number 0.0 to 1.0>,
    "recommendations": ["<actionable recommendation string>"]
  }
}

RULES:
- Base trends on the data provided, do not fabricate numbers.
- adherence_rate = workouts_completed / workouts_planned.
- recommendations should be specific and actionable (e.g. "Increase bench press by 2.5kg next week") not vague.
