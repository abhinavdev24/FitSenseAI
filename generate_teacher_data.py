import os
import json
import time
from groq import Groq
from dotenv import load_dotenv
from prompts import SYSTEM_PROMPT
from feature_engineering import extract_relevant_features
from schema_validation import validate_output


# =============================
# Setup
# =============================

load_dotenv()
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

MODEL_NAME = "llama-3.3-70b-versatile"


# =============================
# LLM Generation
# =============================

def generate_plan(user_data):

    response = client.chat.completions.create(
        model=MODEL_NAME,
        temperature=0.2,
        max_tokens=1000,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": json.dumps(user_data)}
        ],
    )

    print("Token Usage:", response.usage)

    return response.choices[0].message.content


# =============================
# Safe JSON Parse
# =============================

def safe_parse(output):

    if output is None:
        return None

    output = output.strip()

    if output.startswith("```"):
        output = output.replace("```json", "").replace("```", "").strip()

    try:
        return json.loads(output)
    except json.JSONDecodeError:
        return None


# =============================
# Main Pipeline
# =============================

def main():

    # ----- Load Users Safely -----
    with open("data/users_data.json", "r") as f:
        data = json.load(f)

    if isinstance(data, dict):
        users = data.get("users_data", [])
    elif isinstance(data, list):
        users = data
    else:
        raise ValueError("Unexpected JSON structure in users_data.json")

    if not users:
        raise ValueError("No users found in dataset.")

    print(f"Loaded {len(users)} users.")

    training_data = []

    # ----- Generate for Each User -----
    for idx, user in enumerate(users):
        print(f"\nGenerating for user {idx+1}...")

        success = False
        features = extract_relevant_features(user)

        for attempt in range(3):
            raw_output = generate_plan(features)
            parsed_output = safe_parse(raw_output)

            if parsed_output:
                if validate_output(parsed_output):
                    training_data.append({
                        "input": features,
                        "output": parsed_output
                    })
                    success = True
                    break
                else:
                    print(f"⚠ Attempt {attempt+1}: Parsed but failed validation. Retrying...")
            else:
                print(f"⚠ Attempt {attempt+1}: JSON parsing failed. Retrying...")

            time.sleep(1)

        if not success:
            print(f"❌ Failed for user {idx+1} after 3 attempts. Skipping.")

    # ----- Save Synthetic Dataset -----
    with open("data/synthetic_training_data.json", "w") as f:
        json.dump(training_data, f, indent=2)

    print("\n✅ Synthetic dataset generated successfully.")
    print(f"Valid samples generated: {len(training_data)} / {len(users)}")


if __name__ == "__main__":
    main()