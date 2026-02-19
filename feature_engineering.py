from datetime import datetime


def calculate_age(date_of_birth):
    dob = datetime.strptime(date_of_birth, "%Y-%m-%d")
    today = datetime.today()
    return today.year - dob.year - (
        (today.month, today.day) < (dob.month, dob.day)
    )


def extract_relevant_features(user):

    # ----- Sleep Logs -----
    sleep_logs = user.get("time_series", {}).get("sleep_logs", [])
    if sleep_logs:
        avg_sleep = sum(d.get("sleep_duration_hours", 0) for d in sleep_logs) / len(sleep_logs)
        avg_sleep = round(avg_sleep, 2)
    else:
        avg_sleep = None  # No data available

    # ----- Progress -----
    progress = user.get("time_series", {}).get("progress", [])
    if progress:
        latest_weight = progress[-1].get("weight_kg")
    else:
        latest_weight = None

    # ----- Core User Info -----
    user_info = user.get("user", {})

    return {
        "age": calculate_age(user_info["date_of_birth"]) if "date_of_birth" in user_info else None,
        "gender": user_info.get("gender"),
        "height_cm": user_info.get("height_cm"),
        "fitness_level": user_info.get("fitness_level"),
        "primary_goal": user_info.get("goals", {}).get("primary"),
        "medical_conditions": user_info.get("medical_conditions", []),
        "injuries": user_info.get("injuries", {}),
        "equipment_owned": user.get("equipment_owned", []),
        "avg_sleep_hours": avg_sleep,
        "latest_weight": latest_weight
    }
