"""
FitSense AI Rule Engine
Evaluates user workout data and returns boolean rule triggers
"""

def evaluate_rules(user_profile, weekly_summary, performance_trends):
    """
    Evaluates 12 fitness coaching rules
    
    Args:
        user_profile: dict with goal, experience, equipment
        weekly_summary: dict with sessions, RPE, fatigue, sleep
        performance_trends: dict with volume/strength changes
    
    Returns:
        dict of boolean triggers for each rule
    """
    
    triggers = {
        # Progression Rules
        "underloaded_exercise": False,
        "consistent_easy_weeks": False,
        
        # Maintenance Rules
        "appropriate_difficulty": False,
        
        # Fatigue/Deload Rules
        "high_fatigue": False,
        "performance_drop": False,
        "auto_deload_week": False,
        
        # Adherence Rules
        "low_adherence": False,
        "skipped_muscle_group": False,
        
        # Equipment Rules
        "equipment_changed": False,
        
        # Safety Rules
        "pain_reported": False,
        "extreme_overload": False,
    }
    
    # Calculate adherence
    completed = weekly_summary.get('completed_sessions', 0)
    planned = weekly_summary.get('planned_sessions', 1)
    adherence = completed / max(planned, 1)
    
    # Rule A1: Underloaded Exercise
    if (weekly_summary.get('target_reps_completed', False) and
        weekly_summary.get('avg_rpe', 10) <= 6 and
        weekly_summary.get('form_quality') == 'good'):
        triggers['underloaded_exercise'] = True
    
    # Rule A2: Consistent Easy Weeks
    if (weekly_summary.get('easy_weeks_count', 0) >= 2 and
        weekly_summary.get('avg_rpe', 10) <= 6):
        triggers['consistent_easy_weeks'] = True
    
    # Rule B1: Appropriate Difficulty
    avg_rpe = weekly_summary.get('avg_rpe', 0)
    if (7 <= avg_rpe <= 8 and
        weekly_summary.get('target_reps_completed', False)):
        triggers['appropriate_difficulty'] = True
    
    # Rule C1: High Fatigue
    if (weekly_summary.get('avg_fatigue', 0) >= 8 and
        adherence < 0.75):
        triggers['high_fatigue'] = True
    
    # Rule C2: Performance Drop
    if (performance_trends.get('reps_decreased', False) and
        weekly_summary.get('avg_rpe', 0) >= 9):
        triggers['performance_drop'] = True
    
    # Rule C3: Auto Deload Week
    if weekly_summary.get('progressive_weeks_count', 0) >= 4:
        triggers['auto_deload_week'] = True
    
    # Rule D1: Low Adherence
    if adherence < 0.75:
        triggers['low_adherence'] = True
    
    # Rule D2: Skipped Muscle Group
    skipped = weekly_summary.get('skipped_muscle_groups', [])
    if len(skipped) > 0:
        triggers['skipped_muscle_group'] = True
        triggers['skipped_muscles'] = skipped
    
    # Rule E1: Equipment Changed
    if weekly_summary.get('equipment_changed', False):
        triggers['equipment_changed'] = True
    
    # Rule F1: Pain Reported
    if weekly_summary.get('pain_reported', False):
        triggers['pain_reported'] = True
        triggers['pain_locations'] = weekly_summary.get('pain_locations', [])
    
    # Rule F2: Extreme Overload
    if weekly_summary.get('load_increase_percent', 0) > 10:
        triggers['extreme_overload'] = True
    
    return triggers


def format_model_input(user_profile, weekly_summary, performance_trends):
    """
    Format data into structured text for model input
    """
    triggers = evaluate_rules(user_profile, weekly_summary, performance_trends)
    
    equipment_text = ", ".join(user_profile.get('equipment', []))
    
    input_text = f"""USER PROFILE
------------
Goal: {user_profile.get('goal', 'N/A')}
Experience Level: {user_profile.get('experience', 'N/A')}
Days Per Week: {user_profile.get('days_per_week', 'N/A')}
Equipment: {equipment_text}

WEEKLY SUMMARY
--------------
Planned Sessions: {weekly_summary.get('planned_sessions', 'N/A')}
Completed Sessions: {weekly_summary.get('completed_sessions', 'N/A')}
Average RPE: {weekly_summary.get('avg_rpe', 'N/A')}
Average Fatigue Score: {weekly_summary.get('avg_fatigue', 'N/A')}/10
Sleep Average: {weekly_summary.get('sleep_avg', 'N/A')} hours

PERFORMANCE TRENDS
------------------
Chest Volume Trend: {performance_trends.get('chest_volume', 'N/A')}
Leg Volume Trend: {performance_trends.get('leg_volume', 'N/A')}

RULE TRIGGERS
-------------
"""
    
    # Add active triggers
    for rule, active in triggers.items():
        if active and not rule.endswith('_locations') and not rule.endswith('_muscles'):
            input_text += f"{rule.replace('_', ' ').title()}: TRUE\n"
    
    return input_text, triggers


# Test the rule engine
if __name__ == "__main__":
    print("Testing Rule Engine...")
    print("=" * 50)
    
    test_user = {
        'goal': 'Muscle Gain',
        'experience': 'Beginner',
        'days_per_week': 4,
        'equipment': ['Dumbbells']
    }
    
    test_summary = {
        'planned_sessions': 4,
        'completed_sessions': 2,
        'avg_rpe': 9,
        'avg_fatigue': 9,
        'sleep_avg': 5
    }
    
    test_trends = {
        'chest_volume': 'down',
        'leg_volume': 'stable',
        'reps_decreased': True
    }
    
    model_input, triggers = format_model_input(test_user, test_summary, test_trends)
    
    print("Active Triggers:")
    for rule, active in triggers.items():
        if active:
            print(f"  ✅ {rule}")
    
    print("\n" + "=" * 50)
    print("✅ Rule engine test passed!")