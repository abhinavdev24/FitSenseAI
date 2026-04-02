import os
import torch
from unsloth import FastLanguageModel
from trl import SFTTrainer, SFTConfig
from datasets import load_dataset

# 1. Clear GPU memory
torch.cuda.empty_cache()

# 2. Configuration
MODEL_NAME     = "unsloth/Qwen3-4B-bnb-4bit"
MAX_SEQ_LENGTH = 2048
DATASET_PATH   = "Model-Pipeline/data/formatted/20260308T234052Z/train_formatted.jsonl"
OUTPUT_DIR     = "Model-Pipeline/adapters/qwen3-4b-v2"

# 3. Load Model & Tokenizer
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=MODEL_NAME,
    max_seq_length=MAX_SEQ_LENGTH,
    load_in_4bit=True,
)

# 4. Add LoRA Adapters
model = FastLanguageModel.get_peft_model(
    model,
    r=16,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    lora_alpha=32,
    lora_dropout=0,
    bias="none",
)

model.gradient_checkpointing_enable()

# 5. Prepare Dataset
def formatting_prompts_func(examples):
    # AUTO-DETECT KEY: Handles 'formatted_text' or variations
    column_name = "formatted_text" if "formatted_text" in examples else list(examples.keys())[0]
    
    texts = []
    for full_text in examples[column_name]:
        if "<|im_start|>assistant\n" in full_text and "<think>" not in full_text:
            parts = full_text.split("<|im_start|>assistant\n")
            header = parts[0]
            body = parts[1]
            text = (
                header.replace("<|im_end|>", " /no_think<|im_end|>") +
                "<|im_start|>assistant\n<think>\n</think>\n" +
                body + tokenizer.eos_token
            )
        else:
            text = full_text + tokenizer.eos_token
        texts.append(text)
    return {"text": texts}

# Load and Map
dataset = load_dataset("json", data_files=DATASET_PATH, split="train")
dataset = dataset.map(formatting_prompts_func, batched=True)

# 6. Trainer Setup
trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=dataset,
    args=SFTConfig(
        dataset_text_field="text",
        max_seq_length=MAX_SEQ_LENGTH,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=8,
        warmup_steps=10,
        max_steps=150,
        learning_rate=1e-4,
        fp16=True,
        logging_steps=1,
        optim="adamw_8bit",
        weight_decay=0.01,
        lr_scheduler_type="cosine",
        seed=3407,
        output_dir=OUTPUT_DIR,
        report_to="none",
        gradient_checkpointing=True,
    ),
)

# 7. Execute Training
trainer.train()

# 8. Save
model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)
print(f"✅ Training Complete! Model saved to {OUTPUT_DIR}")