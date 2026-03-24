import os
import torch
from unsloth import FastLanguageModel
from trl import SFTTrainer
from transformers import TrainingArguments
from datasets import load_dataset

# 1. Clear GPU memory before starting
torch.cuda.empty_cache()

# 2. Configuration
model_name = "unsloth/Qwen2.5-7B-Instruct-bnb-4bit"
max_seq_length = 2048 
dataset_path = "Model-Pipeline/data/formatted/20260308T234052Z/train_formatted.jsonl"
output_dir = "Model-Pipeline/adapters/qwen-fitsense"

# 3. Load Model & Tokenizer
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = model_name,
    max_seq_length = max_seq_length,
    load_in_4bit = True,
)

# 4. Add LoRA Adapters
model = FastLanguageModel.get_peft_model(
    model,
    r = 16, 
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj"],
    lora_alpha = 16,
    lora_dropout = 0,
    bias = "none",
)

# CRITICAL: Enable gradient checkpointing to save ~30-50% VRAM
model.gradient_checkpointing_enable()

# 5. Prepare Dataset
def formatting_prompts_func(examples):
    instructions = examples["user_message"]
    outputs      = examples["assistant"]
    texts = []
    for instruction, output in zip(instructions, outputs):
        text = f"### Instruction:\n{instruction}\n\n### Response:\n{output}" + tokenizer.eos_token
        texts.append(text)
    return { "text" : texts, }

dataset = load_dataset("json", data_files=dataset_path, split="train")
dataset = dataset.map(formatting_prompts_func, batched = True,)

# 6. Trainer Setup (Optimized for T4 VRAM)
trainer = SFTTrainer(
    model = model,
    tokenizer = tokenizer,
    train_dataset = dataset,
    dataset_text_field = "text",
    max_seq_length = max_seq_length,
    args = TrainingArguments(
        per_device_train_batch_size = 1,      # Reduced from 2 to 1 to prevent OOM
        gradient_accumulation_steps = 8,      # Increased from 4 to 8 to keep total batch size at 8
        warmup_steps = 5,
        max_steps = 60, 
        learning_rate = 2e-4,
        fp16 = True,                          # Force fp16 for T4 compatibility
        logging_steps = 1,
        optim = "adamw_8bit",
        weight_decay = 0.01,
        lr_scheduler_type = "linear",
        seed = 3407,
        output_dir = output_dir,
        report_to = "none",                   # Set to "wandb" if you want live tracking
        gradient_checkpointing = True,        # Extra safety for memory
    ),
)

# 7. Execute Training
trainer.train()

# 8. Save the Adapter
model.save_pretrained(output_dir)
tokenizer.save_pretrained(output_dir)
print(f"✅ Training Complete! Model saved to {output_dir}")