import os
import torch
from unsloth import FastLanguageModel
from trl import SFTTrainer
from transformers import TrainingArguments
from datasets import load_dataset

# 1. Clear GPU memory
torch.cuda.empty_cache()

# 2. Configuration - Using the model you found!
model_name = "unsloth/Qwen3-8B-bnb-4bit" 
max_seq_length = 2048 
dataset_path = "Model-Pipeline/data/formatted/20260308T234052Z/train_formatted.jsonl"
output_dir = "Model-Pipeline/adapters/qwen3-8b-fitsense"

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
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    lora_alpha = 32,
    lora_dropout = 0,
    bias = "none",
)

model.gradient_checkpointing_enable()

# 5. Prepare Dataset (ChatML Format for Qwen3)
def formatting_prompts_func(examples):
    instructions = examples["user_message"]
    outputs      = examples["assistant"]
    texts = []
    for instruction, output in zip(instructions, outputs):
        text = f"<|im_start|>user\n{instruction}<|im_end|>\n<|im_start|>assistant\n{output}<|im_end|>" + tokenizer.eos_token
        texts.append(text)
    return { "text" : texts, }

dataset = load_dataset("json", data_files=dataset_path, split="train")
dataset = dataset.map(formatting_prompts_func, batched = True,)

# 6. Trainer Setup
trainer = SFTTrainer(
    model = model,
    tokenizer = tokenizer,
    train_dataset = dataset,
    dataset_text_field = "text",
    max_seq_length = max_seq_length,
    args = TrainingArguments(
        per_device_train_batch_size = 1,
        gradient_accumulation_steps = 8,
        warmup_steps = 10,
        max_steps = 60, 
        learning_rate = 1e-4, 
        fp16 = True,
        logging_steps = 1,
        optim = "adamw_8bit",
        weight_decay = 0.01,
        lr_scheduler_type = "cosine", 
        seed = 3407,
        output_dir = output_dir,
        report_to = "none",
        gradient_checkpointing = True,
    ),
)

# 7. Execute Training
trainer.train()

# 8. Save
model.save_pretrained(output_dir)
tokenizer.save_pretrained(output_dir)
print(f"✅ Training Complete! Qwen3-8B model saved to {output_dir}")