"""
Parent Model Comparison Script - CURRENT FREE MODELS ONLY
Tests Groq models on FitSense AI scenarios
"""

import json
import time
import os
from dotenv import load_dotenv
from rule_engine import format_model_input
from test_scenarios import test_scenarios

# Load environment variables
load_dotenv()

print("\n" + "="*60)
print("CHECKING API CONFIGURATION")
print("="*60)

# ========================================
# CHECK API KEY
# ========================================
groq_key = os.getenv("GROQ_API_KEY")

print(f"\nGroq API Key: {'Found' if groq_key else 'Missing'}")
if groq_key:
    print(f"  Key starts with: {groq_key[:10]}...")

# ========================================
# IMPORT GROQ CLIENT
# ========================================
GROQ_AVAILABLE = False

try:
    from groq import Groq
    if groq_key:
        groq_client = Groq(api_key=groq_key)
        GROQ_AVAILABLE = True
        print("\nGroq client initialized successfully")
    else:
        print("\nGroq API key missing in .env file")
except Exception as e:
    print(f"\nGroq client failed: {e}")

# ========================================
# MODEL CONFIGURATIONS - CURRENT GROQ MODELS (FEB 2025)
# ========================================
MODELS = {
    # Current Working Models on Groq
    "llama-3.3-70b": {
        "provider": "groq",
        "model_id": "llama-3.3-70b-versatile",
        "available": GROQ_AVAILABLE,
        "description": "70B params - Main Llama model"
    },
    "llama-3.3-70b-specdec": {
        "provider": "groq",
        "model_id": "llama-3.3-70b-specdec",
        "available": GROQ_AVAILABLE,
        "description": "70B params - Speculative decoding (faster)"
    },
    "llama-3.1-8b": {
        "provider": "groq",
        "model_id": "llama-3.1-8b-instant",
        "available": GROQ_AVAILABLE,
        "description": "8B params - Fast, smaller model"
    },
    "gemma-2-9b": {
        "provider": "groq",
        "model_id": "gemma2-9b-it",
        "available": GROQ_AVAILABLE,
        "description": "9B params - Google's Gemma"
    },
    "llama-guard-3-8b": {
        "provider": "groq",
        "model_id": "llama-guard-3-8b",
        "available": GROQ_AVAILABLE,
        "description": "8B params - Safety-focused"
    }
}

# ========================================
# CREATE PROMPT
# ========================================
def create_prompt(scenario):
    """Create prompt from scenario"""
    
    model_input, triggers = format_model_input(
        scenario['user_profile'],
        scenario['weekly_summary'],
        scenario['performance_trends']
    )
    
    prompt = f"""{model_input}

TASK
----
You are FitSense AI, an expert fitness coach. Generate next week's workout plan.

Provide:
1. Workout structure for next week (specific days, exercises, sets, reps)
2. List of specific adjustments made (with rule references)
3. Coaching explanation for the user

Format as JSON:
{{
  "plan": "detailed workout structure",
  "adjustments": ["adjustment 1 with rule", "adjustment 2 with rule"],
  "explanation": "coaching explanation"
}}
"""
    
    return prompt

# ========================================
# CALL MODELS
# ========================================
def call_model(model_name, prompt):
    """Call Groq models"""
    
    config = MODELS[model_name]
    
    if not config['available']:
        return {"error": "Groq not configured"}
    
    try:
        response = groq_client.chat.completions.create(
            model=config['model_id'],
            messages=[
                {"role": "system", "content": "You are FitSense AI, a fitness coach. Output valid JSON."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            response_format={"type": "json_object"},
            max_tokens=2000
        )
        return response.choices[0].message.content
    
    except Exception as e:
        return {"error": str(e)}

# ========================================
# EVALUATE RESPONSE
# ========================================
def evaluate_response(scenario, response):
    """Score response quality (0-10)"""
    
    if isinstance(response, dict) and "error" in response:
        return {
            "total": 0, 
            "rule_adherence": 0,
            "structure": 0,
            "explanation": 0,
            "safety": 0,
            "error": response["error"]
        }
    
    score = {
        "rule_adherence": 0,
        "structure": 0,
        "explanation": 0,
        "safety": 0,
        "total": 0
    }
    
    response_lower = response.lower()
    
    # Rule adherence
    expected = scenario.get('expected_actions', [])
    if expected:
        mentioned = sum(1 for action in expected if action in response_lower)
        score['rule_adherence'] = (mentioned / len(expected)) * 10
    
    # Structure
    has_plan = 'plan' in response_lower or 'workout' in response_lower or 'day' in response_lower
    has_adjustments = 'adjust' in response_lower or 'change' in response_lower or 'reduce' in response_lower
    has_explanation = 'because' in response_lower or 'explanation' in response_lower or 'reason' in response_lower
    score['structure'] = (has_plan + has_adjustments + has_explanation) * 3.33
    
    # Explanation
    rule_words = ['rule', 'fatigue', 'adherence', 'recovery', 'progression', 'overload', 'deload']
    mentions = sum(1 for word in rule_words if word in response_lower)
    score['explanation'] = min(mentions * 2.5, 10)
    
    # Safety
    if scenario['weekly_summary'].get('pain_reported'):
        safety_words = ['doctor', 'medical', 'stop', 'avoid', 'alternative', 'professional', 'consult']
        has_safety = any(word in response_lower for word in safety_words)
        score['safety'] = 10 if has_safety else 0
    else:
        score['safety'] = 10
    
    score['total'] = sum([
        score['rule_adherence'],
        score['structure'],
        score['explanation'],
        score['safety']
    ]) / 4
    
    return score

# ========================================
# RUN COMPARISON
# ========================================
def compare_models(models_to_test=None):
    """Test models on all scenarios"""
    
    if models_to_test is None:
        models_to_test = [name for name, config in MODELS.items() if config['available']]
    
    if not models_to_test:
        return {}
    
    print("\n" + "="*60)
    print("FITSENSE AI - PARENT MODEL COMPARISON")
    print("="*60)
    print(f"\nTesting {len(models_to_test)} FREE models on {len(test_scenarios)} scenarios")
    print(f"Models: {', '.join(models_to_test)}\n")
    
    results = {}
    
    for model_name in models_to_test:
        if model_name not in MODELS or not MODELS[model_name]['available']:
            continue
        
        print(f"\n{'='*60}")
        print(f"Testing: {model_name}")
        print(f"Description: {MODELS[model_name]['description']}")
        print(f"{'='*60}")
        
        results[model_name] = {"scenarios": [], "avg_score": 0}
        
        for i, scenario in enumerate(test_scenarios, 1):
            print(f"\n  [{i}/{len(test_scenarios)}] {scenario['name']}")
            
            prompt = create_prompt(scenario)
            start_time = time.time()
            response = call_model(model_name, prompt)
            elapsed = time.time() - start_time
            score = evaluate_response(scenario, response)
            
            results[model_name]["scenarios"].append({
                "name": scenario['name'],
                "score": score,
                "response": response,
                "time": elapsed
            })
            
            if "error" in score:
                print(f"      Error: {score['error'][:100]}...")
            else:
                print(f"      Score: {score['total']:.1f}/10")
                print(f"      Time: {elapsed:.1f}s")
            
            time.sleep(1.5)
        
        # Calculate average
        valid_scores = [s['score']['total'] for s in results[model_name]["scenarios"] if 'error' not in s['score']]
        if valid_scores:
            avg = sum(valid_scores) / len(valid_scores)
            results[model_name]["avg_score"] = avg
            print(f"\n  Average Score: {avg:.1f}/10")
        else:
            results[model_name]["avg_score"] = 0
    
    return results

# ========================================
# GENERATE REPORT
# ========================================
def generate_report(results):
    """Create markdown report"""
    
    if not results:
        return "# No results\n\nNo models tested successfully."
    
    report = "# FitSense AI - Parent Model Comparison (FREE Models)\n\n"
    report += f"**Date:** {time.strftime('%Y-%m-%d %H:%M')}\n"
    report += f"**Scenarios:** {len(test_scenarios)}\n"
    report += f"**Cost:** $0.00 (All FREE)\n\n"
    
    report += "## Summary\n\n"
    report += "| Rank | Model | Score |\n"
    report += "|------|-------|-------|\n"
    
    sorted_models = sorted(results.items(), key=lambda x: x[1]['avg_score'], reverse=True)
    
    for rank, (model_name, data) in enumerate(sorted_models, 1):
        report += f"| {rank} | {model_name} | {data['avg_score']:.1f}/10 |\n"
    
    report += "\n## Details\n\n"
    
    for model_name, data in sorted_models:
        report += f"### {model_name}\n\n"
        report += f"**Score:** {data['avg_score']:.1f}/10\n\n"
        
        for sr in data['scenarios']:
            score = sr['score']
            report += f"**{sr['name']}**\n"
            if 'error' not in score:
                report += f"- Total: {score['total']:.1f}/10\n"
                report += f"- Rules: {score['rule_adherence']:.1f}/10\n"
                report += f"- Structure: {score['structure']:.1f}/10\n"
                report += f"- Explanation: {score['explanation']:.1f}/10\n"
                report += f"- Safety: {score['safety']:.1f}/10\n\n"
    
    if sorted_models and sorted_models[0][1]['avg_score'] > 0:
        winner = sorted_models[0]
        report += f"\n## Recommendation\n\n**Use {winner[0]}** (Score: {winner[1]['avg_score']:.1f}/10)\n\n"
        if winner[1]['avg_score'] >= 8.0:
            report += "Strong performance. Recommended for 50K examples.\n"
        elif winner[1]['avg_score'] >= 7.0:
            report += "Acceptable. Consider testing paid models for better quality.\n"
        else:
            report += "Low score. Recommend paid alternatives (DeepSeek V3: $0.54).\n"
    
    return report

# ========================================
# MAIN
# ========================================
if __name__ == "__main__":
    
    if not GROQ_AVAILABLE:
        print("\nERROR: Groq not available")
        print("1. Check .env has: GROQ_API_KEY=gsk_...")
        print("2. Run: pip install groq python-dotenv")
        exit(1)
    
    print("\n" + "="*60)
    print("FITSENSE AI - FREE MODEL TESTING")
    print("="*60)
    
    # ONLY CURRENT WORKING MODELS
    models_to_test = [
        "llama-3.3-70b",          # WORKS - 70B
        "llama-3.3-70b-specdec",  # WORKS - 70B faster variant
        "gemma-2-9b",             # WORKS - Google 9B
    ]
    
    models_to_test = [m for m in models_to_test if m in MODELS and MODELS[m]['available']]
    
    if not models_to_test:
        print("\nNo models available.")
        exit(1)
    
    print(f"\nModels: {', '.join(models_to_test)}")
    print(f"Scenarios: {len(test_scenarios)}")
    print(f"Cost: $0.00")
    print(f"Time: ~{len(models_to_test) * len(test_scenarios) * 2} sec")
    
    input("\nPress Enter to start...")
    
    start_total = time.time()
    results = compare_models(models_to_test)
    total_time = time.time() - start_total
    
    if not results:
        print("\nNo results.")
        exit(1)
    
    report = generate_report(results)
    
    with open("model_comparison_report.md", "w", encoding="utf-8") as f:
        f.write(report)
    
    print("\n" + "="*60)
    print("Complete!")
    print(f"Time: {total_time/60:.1f} min")
    print(f"Report: model_comparison_report.md")
    print("="*60)
    
    print("\nRANKINGS:\n")
    sorted_results = sorted(results.items(), key=lambda x: x[1]['avg_score'], reverse=True)
    
    for rank, (model, data) in enumerate(sorted_results, 1):
        print(f"{rank}. {model}: {data['avg_score']:.1f}/10 (FREE)")
    
    if sorted_results and sorted_results[0][1]['avg_score'] > 0:
        winner = sorted_results[0][0]
        print(f"\nRecommendation: {winner}")
    
    print("\n" + "="*60)