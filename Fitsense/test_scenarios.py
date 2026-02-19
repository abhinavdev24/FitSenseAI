"""
Test scenarios for comparing parent models
Each scenario tests different rule combinations
"""

test_scenarios = [
    {
        "name": "High Fatigue + Low Adherence",
        "description": "User is exhausted and missing workouts",
        "user_profile": {
            "goal": "Muscle Gain",
            "experience": "Beginner",
            "days_per_week": 4,
            "equipment": ["Dumbbells"]
        },
        "weekly_summary": {
            "planned_sessions": 4,
            "completed_sessions": 2,
            "avg_rpe": 9,
            "avg_fatigue": 9,
            "sleep_avg": 5,
            "target_reps_completed": False,
            "form_quality": "fair"
        },
        "performance_trends": {
            "chest_volume": "down",
            "leg_volume": "down",
            "reps_decreased": True
        },
        "expected_actions": [
            "deload",
            "reduce volume",
            "recovery",
            "sleep"
        ]
    },
    {
        "name": "Pain Safety",
        "description": "User reports pain during exercises",
        "user_profile": {
            "goal": "Strength",
            "experience": "Intermediate",
            "days_per_week": 3,
            "equipment": ["Barbell", "Dumbbells"]
        },
        "weekly_summary": {
            "planned_sessions": 3,
            "completed_sessions": 3,
            "avg_rpe": 7,
            "avg_fatigue": 6,
            "pain_reported": True,
            "pain_locations": ["knee", "lower back"]
        },
        "performance_trends": {
            "chest_volume": "stable",
            "leg_volume": "stable"
        },
        "expected_actions": [
            "remove",
            "stop",
            "alternative",
            "doctor",
            "medical"
        ]
    },
    {
        "name": "Underloaded - Too Easy",
        "description": "Workouts are too easy, time to progress",
        "user_profile": {
            "goal": "Muscle Gain",
            "experience": "Intermediate",
            "days_per_week": 4,
            "equipment": ["Full Gym"]
        },
        "weekly_summary": {
            "planned_sessions": 4,
            "completed_sessions": 4,
            "avg_rpe": 5,
            "avg_fatigue": 4,
            "target_reps_completed": True,
            "form_quality": "good"
        },
        "performance_trends": {
            "chest_volume": "stable",
            "leg_volume": "stable"
        },
        "expected_actions": [
            "increase",
            "add weight",
            "progress",
            "5%",
            "reps"
        ]
    },
    {
        "name": "Performance Drop",
        "description": "User's strength is declining",
        "user_profile": {
            "goal": "Strength",
            "experience": "Advanced",
            "days_per_week": 5,
            "equipment": ["Barbell", "Dumbbells"]
        },
        "weekly_summary": {
            "planned_sessions": 5,
            "completed_sessions": 5,
            "avg_rpe": 9.5,
            "avg_fatigue": 8,
            "target_reps_completed": False
        },
        "performance_trends": {
            "chest_volume": "down",
            "leg_volume": "down",
            "reps_decreased": True
        },
        "expected_actions": [
            "reduce",
            "decrease weight",
            "recovery",
            "overreaching"
        ]
    },
    {
        "name": "Equipment Change",
        "description": "User's gym closed, only has dumbbells now",
        "user_profile": {
            "goal": "Muscle Gain",
            "experience": "Intermediate",
            "days_per_week": 4,
            "equipment": ["Dumbbells"]  # Previously had barbell
        },
        "weekly_summary": {
            "planned_sessions": 4,
            "completed_sessions": 4,
            "avg_rpe": 7,
            "avg_fatigue": 6,
            "equipment_changed": True
        },
        "performance_trends": {
            "chest_volume": "stable",
            "leg_volume": "stable"
        },
        "expected_actions": [
            "replace",
            "alternative",
            "dumbbell",
            "maintain"
        ]
    }
]