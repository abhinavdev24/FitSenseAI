import json, re

<<<<<<< HEAD
def extract_json(t):
=======
def strip_think(t):
    """Remove Qwen3 <think>...</think> block before any JSON extraction."""
    if "</think>" in t:
        t = t.split("</think>", 1)[-1].strip()
    return t

def extract_json(t):
    t = strip_think(t)
>>>>>>> fa3788e7322f6c3b4708c33047fa9cce653a9ffb
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
<<<<<<< HEAD
        # e.g. {"workout_plan": {...}} or {"plan": {...}, "days": [...]}
=======
>>>>>>> fa3788e7322f6c3b4708c33047fa9cce653a9ffb
        if "workout_plan" in obj and isinstance(obj["workout_plan"], dict):
            obj = obj["workout_plan"]
        elif "plan" in obj and isinstance(obj["plan"], dict):
            obj = obj["plan"]

        # Find days list under any common key
        days = (obj.get("days") or obj.get("workout_days") or
                obj.get("training_days") or obj.get("schedule") or [])

        if not isinstance(days, list) or len(days) == 0:
            return False

<<<<<<< HEAD
        # At least one day must have an exercises list
=======
>>>>>>> fa3788e7322f6c3b4708c33047fa9cce653a9ffb
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

<<<<<<< HEAD
=======
def diagnose(t):
    """Extended diagnostics: detect think leakage, phase nesting, truncation."""
    raw = t
    had_think = "</think>" in raw
    t_clean = strip_think(raw)

    truncated = t_clean.rstrip().endswith((",", "{", "[", '"', ":"))

    try:
        obj = nk(json.loads(extract_json(t_clean)))
        top_keys = list(obj.keys())

        # Check for phase-nested structure
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


>>>>>>> fa3788e7322f6c3b4708c33047fa9cce653a9ffb
with open("Model-Pipeline/reports/eval_report_20260308T234052Z.json") as f:
    report = json.load(f)

plan_recs = [r for r in report["per_record"] if r["prompt_type"] in {"plan_creation", "plan_updation"}]
sv = [check(r["prediction"]) for r in plan_recs]
<<<<<<< HEAD
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
=======

print(f"Plan records checked : {len(plan_recs)}")
print(f"Schema valid         : {sum(sv)}/{len(plan_recs)} = {sum(sv)/len(plan_recs):.2%}")

# Aggregate diagnostics
think_count    = 0
truncate_count = 0
phase_count    = 0

print("\n--- Diagnostics for first 10 predictions ---")
for r in plan_recs[:10]:
    d = diagnose(r["prediction"])
    think_count    += int(d["had_think_block"])
    truncate_count += int(d["truncated"])
    phase_count    += int(d["has_phases"])
    print(
        f"  prompt_type={r['prompt_type']} | "
        f"think={d['had_think_block']} | "
        f"truncated={d['truncated']} | "
        f"phases={d['has_phases']} | "
        f"keys={d['top_keys']}"
    )

# Full-set aggregates
print("\n--- Full dataset diagnostics ---")
all_diag = [diagnose(r["prediction"]) for r in plan_recs]
print(f"  Had think block  : {sum(d['had_think_block'] for d in all_diag)}/{len(plan_recs)}")
print(f"  Truncated        : {sum(d['truncated'] for d in all_diag)}/{len(plan_recs)}")
print(f"  Phase-nested     : {sum(d['has_phases'] for d in all_diag)}/{len(plan_recs)}")
>>>>>>> fa3788e7322f6c3b4708c33047fa9cce653a9ffb
