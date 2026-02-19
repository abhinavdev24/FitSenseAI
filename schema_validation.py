def validate_output(output):

    required_top_keys = [
        "weekly_plan",
        "adaptation_explanation",
        "safety_notes",
        "motivation_message"
    ]

    # Check top-level keys
    for key in required_top_keys:
        if key not in output:
            print(f"Missing key: {key}")
            return False

    # weekly_plan must be a list
    if not isinstance(output["weekly_plan"], list):
        print("weekly_plan is not a list")
        return False

    # Validate each day entry
    for day in output["weekly_plan"]:
        required_day_keys = ["day", "focus", "intensity", "duration_minutes"]
        for key in required_day_keys:
            if key not in day:
                print(f"Missing day key: {key}")
                return False

    return True
