import json, re

REPORT_PATH = "/content/project_folder/FitSenseAI_Final/Model-Pipeline/reports/student_eval_20260331Z.json"

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

def check(t):
    try:
        obj = nk(json.loads(extract_json(t)))
        if "workout_plan" in obj and isinstance(obj["workout_plan"], dict):
            obj = obj["workout_plan"]
        elif "training_plan" in obj and isinstance(obj["training_plan"], dict):
            obj = obj["training_plan"]
        elif "plan" in obj and isinstance(obj["plan"], dict):
            obj = obj["plan"]
        days = (obj.get("days") or obj.get("workout_days") or
                obj.get("training_days") or obj.get("schedule") or [])
        if not isinstance(days, list) or len(days) == 0:
            return False
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

def diagnose(t):
    raw = t
    had_think = "</think>" in raw
    t_clean = strip_think(raw)
    truncated = t_clean.rstrip().endswith((",", "{", "[", '"', ":"))
    try:
        obj = nk(json.loads(extract_json(t_clean)))
        top_keys = list(obj.keys())
        has_phases = (
            "phases" in obj
            or any("phases" in v for v in obj.values() if isinstance(v, dict))
            or any("workouts" in v for v in obj.values() if isinstance(v, dict))
        )
    except Exception as e:
        top_keys = [f"PARSE_ERROR: {e}"]
        has_phases = False
    return {
        "had_think_block": had_think,
        "truncated":       truncated,
        "has_phases":      has_phases,
        "top_keys":        top_keys,
    }

with open(REPORT_PATH) as f:
    report = json.load(f)

plan_recs = [r for r in report["per_record"] if r["prompt_type"] in {"plan_creation", "plan_updation"}]
sv = [check(r["prediction"]) for r in plan_recs]

print(f"Plan records checked : {len(plan_recs)}")
print(f"Schema valid         : {sum(sv)}/{len(plan_recs)} = {sum(sv)/len(plan_recs):.2%}")

think_count = truncate_count = phase_count = 0
print("\n--- Diagnostics for first 10 predictions ---")
for r in plan_recs[:10]:
    d = diagnose(r["prediction"])
    think_count    += int(d["had_think_block"])
    truncate_count += int(d["truncated"])
    phase_count    += int(d["has_phases"])
    print(f"  prompt_type={r['prompt_type']} | think={d['had_think_block']} | truncated={d['truncated']} | phases={d['has_phases']} | keys={d['top_keys']}")

print("\n--- Full dataset diagnostics ---")
all_diag = [diagnose(r["prediction"]) for r in plan_recs]
print(f"  Had think block  : {sum(d['had_think_block'] for d in all_diag)}/{len(plan_recs)}")
print(f"  Truncated        : {sum(d['truncated'] for d in all_diag)}/{len(plan_recs)}")
print(f"  Phase-nested     : {sum(d['has_phases'] for d in all_diag)}/{len(plan_recs)}")
