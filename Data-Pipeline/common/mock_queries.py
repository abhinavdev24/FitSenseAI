"""Prompt templates and context requirements for each LLM action type in FitSenseAI.

This module documents:
  - CONTEXT_FIELDS: which synthetic data fields to embed in the prompt per action type
  - PROMPT_EXAMPLES and per-type lists: representative prompt texts for testing/inspection

Actual prompts are generated dynamically in generate_synthetic_queries.py using real
synthetic pipeline data. No PII (name, email, credentials) is included in any prompt.
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Context fields required per action type
# These fields come from synthetic pipeline data — NO PII (no name/email/id)
# ---------------------------------------------------------------------------

CONTEXT_FIELDS: dict[str, list[str]] = {
    "plan_creation": [
        "age_band",        # e.g. "25-34"
        "sex",             # "male" | "female" | "non_binary"
        "activity_level",  # "sedentary" | "light" | "moderate" | "active" | "very_active"
        "goal",            # e.g. "muscle_gain"
        "conditions",      # list of condition slugs, e.g. ["lower_back_pain"]
    ],
    "plan_modification": [
        "age_band",
        "sex",
        "goal",
        "conditions",
        "current_plan_exercises",  # list of (exercise_name, sets, reps, weight_kg)
        "avg_reps",                # float
        "avg_weight",              # float kg
        "avg_rir",                 # float
    ],
    "safety_adjustment": [
        "age_band",
        "sex",
        "conditions",
        "current_plan_exercises",  # list of exercise names to evaluate for safety
    ],
    "progress_adaptation": [
        "goal",
        "activity_level",
        "workout_count",   # int
        "avg_reps",        # float
        "avg_weight",      # float kg
        "avg_rir",         # float
        "trend",           # "plateau" | "improving" | "fatigue_signals"
    ],
    "progress_comment": [
        "goal",
        "workout_count",   # int
        "avg_reps",        # float
        "avg_weight",      # float kg
    ],
    "workout_logging": [
        "conditions",
        "logged_exercises",  # list of (exercise_name, sets, reps, weight_kg) from EXERCISE_POOL
    ],
    "metric_logging": [
        "metric",   # "weight" | "sleep" | "calories"
        "value",    # numeric value
    ],
    "coaching_qa": [
        "goal",
        "conditions",
    ],
}

# ---------------------------------------------------------------------------
# Example prompt texts per action type
# Used in tests and manual inspection. Actual prompts include embedded
# synthetic context — see generate_synthetic_queries.py for full assembly.
# ---------------------------------------------------------------------------

PLAN_CREATION_PROMPTS: list[str] = [
    "Create a 7-day hypertrophy plan. Include sets, reps, weight, RIR, and rest.",
    "Design a 3-day full body beginner strength program.",
    "Build a 5-day upper/lower split for fat loss with moderate activity.",
    "Create a home workout plan for general fitness with minimal equipment.",
    "Design a 4-day push/pull/legs split for intermediate lifters.",
]

PLAN_MODIFICATION_PROMPTS: list[str] = [
    "My knees have been sore. Replace squats with a lower-impact quad exercise.",
    "I've been hitting my RIR targets consistently for 3 weeks. Increase intensity.",
    "Add a second back day to my current plan.",
    "Reduce the volume — I've been feeling fatigued after each session.",
    "Swap the barbell movements for dumbbell equivalents.",
]

SAFETY_ADJUSTMENT_PROMPTS: list[str] = [
    "Which exercises in my current plan are contraindicated given my conditions?",
    "Suggest safer alternatives for the high-spinal-load movements in my plan.",
    "I have hypertension — flag any exercises I should avoid or modify.",
    "Given my lower back condition, which movements need substitution?",
    "Review my plan for exercises that conflict with my medical history.",
]

PROGRESS_ADAPTATION_PROMPTS: list[str] = [
    "I've hit a plateau for 3 weeks. Propose a progression strategy for the next 2 weeks.",
    "My RIR has been 0 consistently — I may be overreaching. Suggest a deload.",
    "I've been improving steadily. How should I progress over the next 2 weeks?",
    "I'm showing fatigue signals. Adjust my plan for the next week.",
    "Plateau on my main lifts for a month. What periodization change do you recommend?",
]

PROGRESS_COMMENT_PROMPTS: list[str] = [
    "How am I doing toward my goal? Give me an honest summary.",
    "I've completed several sessions this month — what should I focus on next?",
    "Am I on track based on my recent workout performance?",
    "Summarize my progress and tell me what to prioritize.",
    "How is my training going? What areas need attention?",
]

WORKOUT_LOGGING_PROMPTS: list[str] = [
    "I just finished my workout. Please log this session.",
    "Done for today. Log this and tell me how it aligns with my plan.",
    "Completed my training. Log it and flag anything unusual.",
    "Just wrapped up. Log my session and let me know if the loads were appropriate.",
    "Finished my workout. Log it and note if I'm progressing as expected.",
]

METRIC_LOGGING_PROMPTS: list[str] = [
    "Log my weight for today.",
    "Log my sleep from last night.",
    "Log my calorie intake for today.",
    "Record my body weight measurement.",
    "Log how many hours I slept.",
]

COACHING_QA_PROMPTS: list[str] = [
    "What's the best rep range for hypertrophy?",
    "Should I do cardio on rest days?",
    "How much protein do I need per day?",
    "Is it okay to work out on an empty stomach?",
    "How long should I rest between sets for strength?",
    "What are the signs of overtraining?",
    "Should I stretch before or after my workout?",
    "How do I break through a strength plateau?",
    "Is creatine safe to take daily?",
    "How important is sleep for muscle recovery?",
    "What is progressive overload and how do I apply it?",
    "Is it better to train 3 days or 5 days per week?",
    "How do I know if I'm eating enough to build muscle?",
    "What does RIR mean and how do I use it?",
    "Should I do compound or isolation exercises first?",
]

# Combined lookup — used by generate_synthetic_queries.py
PROMPT_EXAMPLES: dict[str, list[str]] = {
    "plan_creation": PLAN_CREATION_PROMPTS,
    "plan_modification": PLAN_MODIFICATION_PROMPTS,
    "safety_adjustment": SAFETY_ADJUSTMENT_PROMPTS,
    "progress_adaptation": PROGRESS_ADAPTATION_PROMPTS,
    "progress_comment": PROGRESS_COMMENT_PROMPTS,
    "workout_logging": WORKOUT_LOGGING_PROMPTS,
    "metric_logging": METRIC_LOGGING_PROMPTS,
    "coaching_qa": COACHING_QA_PROMPTS,
}
