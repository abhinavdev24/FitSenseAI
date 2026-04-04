import json, re

REPORT_PATH = "Model-Pipeline/reports/student_eval_20260403Z.json"

def strip_think(t):
    if "</think>" in t:
        t = t.split("</think>", 1)[-1].strip()
    return t

def extract_json(t):
    t = strip_think(t)
    m = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", t, re.DOTALL)
    if m: return m.group(1)
    s, e = t.find("{"), t.rfind("}")
    return t[s:e+1] if s != -1 and e > s else t

def nk(o):
    if isinstance(o, dict): return {k.lower(): nk(v) for k, v in o.items()}
    if isinstance(o, list): return [nk(i) for i in o]
    return o

def has_exercises(d):
    """Check if a dict contains exercises in any recognized format."""
    if not isinstance(d, dict):
        return False
    # Pattern A: exercises as a list
    exs = (d.get("exercises") or d.get("exercise_list") or
           d.get("workout") or d.get("movements") or
           d.get("main_exercises") or [])
    if isinstance(exs, list) and len(exs) > 0:
        return True
    # Pattern B: exercises as named keys with sets/reps
    for val in d.values():
        if isinstance(val, dict) and ("sets" in val or "reps" in val):
            return True
    return False

def check(t):
    try:
        obj = nk(json.loads(extract_json(t)))
        if "workout_plan" in obj and isinstance(obj["workout_plan"], dict):
            obj = obj["workout_plan"]
        elif "training_plan" in obj and isinstance(obj["training_plan"], dict):
            obj = obj["training_plan"]
        elif "plan" in obj and isinstance(obj["plan"], dict):
            obj = obj["plan"]

        # Pattern A: "days" is a list
        days = (obj.get("days") or obj.get("workout_days") or
                obj.get("training_days") or obj.get("schedule") or [])
        if isinstance(days, list) and len(days) > 0:
            for d in days:
                if has_exercises(d):
                    return True

        # Pattern B: "days" is a dict with day names as keys
        if isinstance(days, dict) and len(days) > 0:
            for day_name, day_val in days.items():
                if isinstance(day_val, dict) and has_exercises(day_val):
                    return True

        return False
    except:
        return False

with open(REPORT_PATH) as f:
    report = json.load(f)

plan_recs = [r for r in report["per_record"] if r["prompt_type"] in {"plan_creation", "plan_updation"}]
sv = [check(r["prediction"]) for r in plan_recs]

print(f"Plan records checked : {len(plan_recs)}")
print(f"Schema valid         : {sum(sv)}/{len(plan_recs)} = {sum(sv)/len(plan_recs):.2%}")

creation_recs = [r for r in plan_recs if r["prompt_type"] == "plan_creation"]
updation_recs = [r for r in plan_recs if r["prompt_type"] == "plan_updation"]
creation_valid = sum(check(r["prediction"]) for r in creation_recs)
updation_valid = sum(check(r["prediction"]) for r in updation_recs)

print(f"\n--- By prompt type ---")
if creation_recs: print(f"  plan_creation : {creation_valid}/{len(creation_recs)} = {creation_valid/len(creation_recs):.2%}")
if updation_recs: print(f"  plan_updation : {updation_valid}/{len(updation_recs)} = {updation_valid/len(updation_recs):.2%}")
