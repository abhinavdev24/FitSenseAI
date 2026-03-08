ACTION: safety_adjustment
Evaluate whether a given exercise or plan is safe for the user given their medical profile.

RESPOND WITH:
{
  "message": "<conversational summary>",
  "data": {
    "safe": <boolean>,
    "reason": "<string explaining the assessment>",
    "alternatives": [
      {
        "original_exercise": "<exercise flagged as unsafe>",
        "substitute_exercise": "<safer replacement>",
        "reason": "<why this substitute is safer>"
      }
    ]
  }
}

RULES:
- alternatives array is required when safe is false; can be empty array when safe is true.
- Always provide a specific substitute, not just "consult a professional".
- If the user mentions active pain, set safe to false and advise stopping in the message.
