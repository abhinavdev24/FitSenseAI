# FitSense AI - Parent Model Testing

## Goal
Test multiple parent LLM models to determine which generates the best quality training data for FitSense AI.

---

## 🎯 You Can Test ANY Models You Want!

This script is **provider-agnostic** and supports testing ANY LLM model from ANY provider.

**Just:**
1. Get the API key for that provider
2. Add the model to the `MODELS` dictionary in `test_parent_models.py`
3. Run the test script

**Supported Providers:**
- ✅ Groq (FREE - Llama, Mixtral models)
- ✅ OpenRouter (200+ models with single API key)
- ✅ Anthropic (Claude models)
- ✅ OpenAI (GPT models)
- ✅ Google (Gemini models)
- ✅ Together AI
- ✅ Cohere
- ✅ Any provider with OpenAI-compatible API

---

## Setup Instructions (15 minutes)

### Step 1: Install Python Dependencies
```bash
pip install -r requirements.txt
```

### Step 2: Get API Keys

**FREE Options (Start Here):**
- **Groq**: https://console.groq.com/keys
  - Sign up → Create API key → Copy it
  - Models: Llama 3.1/3.3 70B, Mixtral 8x7B
  - Cost: FREE

**Paid Options (For Comparison):**
- **OpenRouter**: https://openrouter.ai/keys
  - Access to 200+ models with one API key
  - Models: Qwen, DeepSeek, GPT-4, Claude, Gemini, etc.
  - Cost: $5 free credit, then pay-as-you-go
  
- **Anthropic**: https://console.anthropic.com
  - Models: Claude 3.5 Sonnet, Claude 3 Opus
  - Cost: $5 free credit
  
- **OpenAI**: https://platform.openai.com/api-keys
  - Models: GPT-4o, GPT-4 Turbo
  - Cost: Pay-as-you-go

### Step 3: Configure API Keys

1. Copy `.env.example` to `.env`:
```bash
cp .env.example .env
```

2. Edit `.env` and add your keys:
```env
GROQ_API_KEY=your_groq_key_here
OPENROUTER_API_KEY=your_openrouter_key_here
ANTHROPIC_API_KEY=your_anthropic_key_here
```

### Step 4: Test Setup
```bash
python rule_engine.py
```

You should see:
```
Testing Rule Engine...
✅ Rules evaluated: {'high_fatigue': True, 'low_adherence': True}
```

### Step 5: Run Model Comparison
```bash
python test_parent_models.py
```

This will:
- Test 2-5 parent models on 5 scenarios
- Generate scores for each model
- Save results to `model_comparison_report.md`
- Take ~5-10 minutes per model

---

## 🔧 Adding Custom Models

Want to test a specific model not in the default list? Easy!

### Example: Adding a New Model

1. Open `test_parent_models.py`

2. Add your model to the `MODELS` dictionary:
```python
MODELS = {
    # ... existing models ...
    
    "your-model-name": {
        "provider": "openrouter",  # or "groq", "anthropic", etc.
        "model_id": "provider/model-identifier",
        "available": OPENROUTER_AVAILABLE  # or relevant client
    }
}
```

3. Add to test list at bottom of file:
```python
models_to_test = [
    "llama-3.1-70b",
    "your-model-name",  # ← Add here
]
```

4. Run the test!

### Example Models You Can Add:

**Via OpenRouter (One API Key = 200+ Models):**
```python
"deepseek-v3": {
    "provider": "openrouter",
    "model_id": "deepseek/deepseek-chat",  # $0.27/1M tokens
    "available": OPENROUTER_AVAILABLE
}

"gpt-4o": {
    "provider": "openrouter",
    "model_id": "openai/gpt-4o",  # $2.50/1M tokens
    "available": OPENROUTER_AVAILABLE
}

"gemini-pro": {
    "provider": "openrouter",
    "model_id": "google/gemini-pro-1.5",  # $1.25/1M tokens
    "available": OPENROUTER_AVAILABLE
}

"mixtral-8x22b": {
    "provider": "openrouter",
    "model_id": "mistralai/mixtral-8x22b-instruct",  # $0.65/1M tokens
    "available": OPENROUTER_AVAILABLE
}
```

**Via Groq (FREE):**
```python
"mixtral-8x7b": {
    "provider": "groq",
    "model_id": "mixtral-8x7b-32768",
    "available": GROQ_AVAILABLE
}
```

**Full list of OpenRouter models:** https://openrouter.ai/docs#models

---

## What to Test

You're evaluating which model is best at:
1. **Following triggered rules correctly** (e.g., reduces volume when high fatigue detected)
2. **Generating natural coaching explanations** (sounds like a real fitness coach)
3. **Creating structured workout plans** (clear format with days, exercises, sets, reps)
4. **Prioritizing safety** (mentions doctor/medical when pain is reported)

---

## Recommended Testing Strategy

### Quick Test (FREE, 15 min)
```python
models_to_test = [
    "llama-3.1-70b",    # Groq - FREE
    "llama-3.3-70b",    # Groq - FREE
]
```

### Balanced Test ($0.05, 25 min)
```python
models_to_test = [
    "llama-3.1-70b",    # FREE
    "llama-3.3-70b",    # FREE
    "deepseek-v3",      # $0.014 for 5 scenarios
    "qwen-2.5-72b",     # $0.025 for 5 scenarios
]
```

### Comprehensive Test ($0.20, 40 min)
```python
models_to_test = [
    "llama-3.1-70b",        # FREE
    "qwen-2.5-72b",         # $0.025
    "claude-3.5-sonnet",    # $0.15
]
```

---

## Division of Work

**Person 1:**
- Test: Llama 3.1 70B (Groq - FREE)
- Test: Qwen 2.5 72B (OpenRouter - ~$0.03)

**Person 2:**
- Test: Llama 3.3 70B (Groq - FREE)
- Test: DeepSeek V3 (OpenRouter - ~$0.01)
- Test: Claude 3.5 Sonnet (optional, ~$0.15)

---

## Expected Output

After running, you'll have:

1. **Console Output:**
```
===========================================================
Testing: llama-3.1-70b
===========================================================

  [1/5] High Fatigue + Low Adherence
      Score: 8.5/10
      Time: 2.3s

  [2/5] Pain Safety
      Score: 9.2/10
      Time: 1.8s
  
  ...
  
  📊 Average Score: 8.7/10
```

2. **model_comparison_report.md:**
- Detailed rankings
- Score breakdown per scenario
- Recommendation on which model to use

---

## Share Results

Send back:
1. `model_comparison_report.md`
2. Screenshot of console output showing final rankings
3. Your recommendation: Which model scored best?

---

## Timeline

- **Setup**: 15 minutes
- **Testing**: 5-10 minutes per model
- **Total**: 30-60 minutes depending on number of models

---

## Troubleshooting

### "ModuleNotFoundError: No module named 'groq'"
```bash
pip install -r requirements.txt
```

### "API key not found"
Make sure `.env` file exists and contains your API keys:
```bash
cat .env  # Check if keys are there
```

### "Rate limit exceeded"
- Groq: Wait 60 seconds between runs
- Add `time.sleep(2)` between API calls in the script

### Model returns errors
- Check if model ID is correct
- Verify API key has access to that model
- Some models require specific formatting

---

## Cost Estimates

**For 5 test scenarios (~10K tokens total per model):**

| Model | Provider | Cost per Test |
|-------|----------|---------------|
| Llama 3.1 70B | Groq | FREE |
| Llama 3.3 70B | Groq | FREE |
| Mixtral 8x7B | Groq | FREE |
| DeepSeek V3 | OpenRouter | $0.0027 |
| Qwen 2.5 72B | OpenRouter | $0.005 |
| Gemini 1.5 Pro | OpenRouter | $0.0125 |
| GPT-4o | OpenRouter | $0.025 |
| Claude 3.5 Sonnet | OpenRouter | $0.03 |

**For 50K training examples (~2M tokens):**

| Model | Cost for 50K |
|-------|--------------|
| Llama 3.3 70B (Groq) | FREE |
| DeepSeek V3 | $13.50 |
| Qwen 2.5 72B | $25.00 |
| Gemini 1.5 Pro | $62.50 |
| GPT-4o | $125.00 |
| Claude 3.5 Sonnet | $150.00 |

---

## File Descriptions

- **README.md** - This file, setup instructions
- **requirements.txt** - Python dependencies
- **rule_engine.py** - Core logic that evaluates 12 fitness rules
- **test_scenarios.py** - 5 test cases covering different rule combinations
- **test_parent_models.py** - Main script that tests models and generates report
- **.env.example** - Template for API keys
- **.env** - Your actual API keys (create this, don't commit to git!)

---

## Questions?

Contact: [Your contact info]

GitHub Issues: [Your repo link]

---

## Next Steps After Testing

Once you've identified the best model:

1. ✅ Generate 50K training examples using that model
2. ✅ Fine-tune Llama 3.1 8B (child model) on those examples
3. ✅ Deploy fine-tuned model for production use

**Current Phase:** Week 1 - Model Selection
**Next Phase:** Week 2 - Data Generation (50K examples)
```

---

# **DIRECTORY STRUCTURE**

Here's the complete directory structure showing where everything goes:
```
fitsense-ml/
│
├── README.md                          # ← Setup instructions (UPDATED)
├── requirements.txt                   # ← Python dependencies
├── .env.example                       # ← API key template
├── .env                              # ← Your actual API keys (gitignored)
├── .gitignore                        # ← Git ignore file
│
├── rule_engine.py                    # ← Rule evaluation logic
├── test_scenarios.py                 # ← Test cases for models
├── test_parent_models.py            # ← Main testing script
│
├── model_comparison_report.md        # ← Generated after running tests (OUTPUT)
│
├── data/                             # ← Data folder
│   ├── users_data.json              # ← Your existing user data 
│   └── json_output.ipynb            # ← Your existing notebook
│
├── generated_data/                   # ← Generated training data 
│   ├── training_data.jsonl          # ← 50K training examples
│   └── fitsense_dataset/            # ← Processed for fine-tuning
│
├── models/                           # ← Fine-tuned models 
│   └── fitsense-llama-8b-final/     # ← Your trained model
│
└── scripts/                          # ← Additional utility scripts
    ├── scenario_generator.py        # ← Generate diverse scenarios
    ├── generate_training_data.py    # ← Generate 50K examples
    └── prepare_finetuning_data.py   # ← Prepare data for training
```

---

## **WHERE TO PLACE YOUR EXISTING FILES**

### **users_data.json**
```
fitsense-ml/data/users_data.json
```

This is reference data showing user structure. You'll use this format when generating training scenarios.

### **json_output.ipynb**
```
fitsense-ml/data/json_output.ipynb