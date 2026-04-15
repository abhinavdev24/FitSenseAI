# Vertex custom inference container

This directory contains a small FastAPI server that Vertex AI can use as a custom prediction container.

## Request contract

The container expects Vertex prediction payloads of the form:

```json
{
  "instances": [
    {
      "task": "plan_json",
      "system_prompt": "...",
      "user_message": "...",
      "max_new_tokens": 900
    }
  ]
}
```

## Response contract

For `plan_json`, the container returns:

```json
{
  "predictions": [
    {
      "plan_json": {"plan_name": "...", "days": [...]},
      "raw_text": "..."
    }
  ]
}
```

For `coach_text` or `text`, the container returns:

```json
{
  "predictions": [
    {
      "text": "..."
    }
  ]
}
```

The backend `llm_runtime.py` already knows how to consume this schema.
