import json, re

def extract_json(t):
    m = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", t, re.DOTALL)
    if m: return m.group(1)
    s, e = t.find("{"), t.rfind("}")
    return t[s:e+1] if s != -1 and e > s else t

def nk(o):
    if isinstance(o, dict): return {k.lower(): nk(v) for k, v in o.items()}
    if isinstance(o, list): return [nk(i) for i in o]
    return o

def check(t):
    try:
        obj = nk(json.loads(extract_json(t)))

        # Unwrap common nested structures
        # e.g. {"workout_plan": {...}} or {"plan": {...}, "days": [...]}
        if "workout_plan" in obj and isinstance(obj["workout_plan"], dict):
            obj = obj["workout_plan"]
        elif "plan" in obj and isinstance(obj["plan"], dict):
            obj = obj["plan"]

        # Find days list under any common key
        days = (obj.get("days") or obj.get("workout_days") or
                obj.get("training_days") or obj.get("schedule") or [])

        if not isinstance(days, list) or len(days) == 0:
            return False

        # At least one day must have an exercises list
        for d in days:
            if not isinstance(d, dict):
                continue
            exs = (d.get("exercises") or d.get("exercise_list") or
                   d.get("workout") or d.get("movements") or [])
            if isinstance(exs, list) and len(exs) > 0:
                return True
        return False
    except:
        return False

with open("Model-Pipeline/reports/eval_report_20260308T234052Z.json") as f:
    report = json.load(f)

plan_recs = [r for r in report["per_record"] if r["prompt_type"] in {"plan_creation", "plan_updation"}]
sv = [check(r["prediction"]) for r in plan_recs]
print(f"Plan records checked : {len(plan_recs)}")
print(f"Schema valid         : {sum(sv)}/{len(plan_recs)} = {sum(sv)/len(plan_recs):.2%}")

# Show top-level keys from first 5 predictions to diagnose structure
print("\n--- Top-level keys in first 5 predictions ---")
for r in plan_recs[:5]:
    try:
        obj = nk(json.loads(extract_json(r["prediction"])))
        print(f"  prompt_type={r['prompt_type']} | keys={list(obj.keys())}")
    except Exception as e:
        print(f"  FAILED to parse: {e}")