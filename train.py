import torch
import pandas as pd
import os
from datasets import load_dataset, Dataset
from sklearn.model_selection import train_test_split

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    BitsAndBytesConfig
)

from peft import LoraConfig, get_peft_model, PeftModel, prepare_model_for_kbit_training
from trl import SFTTrainer

# ============================================================
# 1. CONFIG
# ============================================================
# Changed to DeepSeek-R1-Distill-Llama-8B (The Llama-3.1 based version)
MODEL_NAME = "deepseek-ai/DeepSeek-R1-Distill-Llama-8B" 
# DATASET_NAME = "RafaM97/marketing_social_media"
DATASET_NAME = "./dataset"
OUTPUT_DIR = "./marketing_agent_deepseek_v1"
MERGED_DIR = "./marketing_agent_deepseek_v1_merged"

MAX_SEQ_LENGTH = 2048
TEST_SIZE = 0.05

# ============================================================
# 2. CLEAN DATASET (ML Term: Data Clearance)
# ============================================================
print("\n=== 1 & 2. Loading and Cleaning Dataset ===")

raw_dataset = load_dataset(DATASET_NAME, split="train")
df = raw_dataset.to_pandas()

df.dropna(subset=["question", "answer"], inplace=True)
df.drop_duplicates(subset=["question", "answer"], inplace=True)
df.reset_index(drop=True, inplace=True)

print(f"Total samples after cleaning: {len(df)}")

train_df, test_df = train_test_split(df, test_size=TEST_SIZE, random_state=42)
train_dataset = Dataset.from_pandas(train_df)
test_dataset = Dataset.from_pandas(test_df)

# ============================================================
# 3. LOAD TOKENIZER & MODEL (QLoRA)
# ============================================================
print("\n=== 3. Loading DeepSeek Model with 4-bit QLoRA ===")

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4"
)

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
# Fix for the padding warning:
tokenizer.padding_side = 'right' 
tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    quantization_config=bnb_config,
    trust_remote_code=True,
    low_cpu_mem_usage=True
)

model = prepare_model_for_kbit_training(model)

# ============================================================
# 4. APPLY LoRA
# ============================================================
print("\n=== 4. Applying LoRA ===")

lora_cfg = LoraConfig(
    r=16,
    lora_alpha=16,
    lora_dropout=0.05,
    bias="none",
    # DeepSeek-R1-Distill-Llama models use standard Llama module names
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    task_type="CAUSAL_LM"
)

model = get_peft_model(model, lora_cfg)
model.print_trainable_parameters()

# ============================================================
# 5. FORMAT DATASET FOR TRAINING
# ============================================================
def format_samples(batch):
    texts = []
    eos = tokenizer.eos_token

    for q, a, meta in zip(batch["question"], batch["answer"], batch["metadata"]):
        
        # Extract metadata to guide the 'thinking' process
        concept = meta.get("concept", "General Marketing")
        context = meta.get("business_context", "Strategy")
        
        # Create a reasoning block that forces the model to identify the marketing framework first
        reasoning = (
            f"The user is asking about {context}. "
            f"I should apply the {concept} framework to provide a structured and professional response. "
            "I will break down the steps and provide a clear expected outcome."
        )
        
        # DeepSeek-style Assistant content
        assistant_content = f"<think>{reasoning}</think>{a}"
        
        messages = [
            {"role": "user", "content": q},
            {"role": "assistant", "content": assistant_content} 
        ]

        formatted = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False
        ) + eos

        texts.append(formatted)

    return {"text": texts}

# Apply mapping (Make sure your dataset is loaded as 'train_dataset')
train_dataset = train_dataset.map(format_samples, batched=True)

train_dataset = train_dataset.map(format_samples, batched=True, remove_columns=train_dataset.column_names)
test_dataset = test_dataset.map(format_samples, batched=True, remove_columns=test_dataset.column_names)

# ============================================================
# 6. TRAINING (ML Term: Training Loop)
# ============================================================
print("\n=== 6. Training Setup ===")

from trl import SFTConfig, SFTTrainer

# Define SFT-specific and standard training arguments in SFTConfig
sft_config = SFTConfig(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4, 
    num_train_epochs=3,
    learning_rate=2e-4,
    logging_steps=5,
    save_steps=200,
    eval_steps=200,
    eval_strategy="steps", 
    save_strategy="steps",
    bf16=False, 
    fp16=True, 
    weight_decay=0.01,
    optim="adamw_8bit",
    report_to="none", 
    remove_unused_columns=False,
    
    # --- UPDATED PARAMETER NAMES ---
    dataset_text_field="text",
    max_length=MAX_SEQ_LENGTH  # 🐛 FIX: Changed 'max_seq_length' to 'max_length'
)

trainer = SFTTrainer(
    model=model,
    processing_class=tokenizer, # (Already fixed in previous step)
    args=sft_config,
    train_dataset=train_dataset,
    eval_dataset=test_dataset
)

# trainer.train() 

# Ensure the adapter is saved for Section 7
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
model.save_pretrained(OUTPUT_DIR)

print("Training setup complete. Adapter weights saved to:", OUTPUT_DIR)

# ============================================================
# 7. MERGE & SAVE
# ============================================================
print("\n=== 7. Merging & Saving final model ===")

# 1. Clear memory to make room for the 16-bit model
del model
if torch.cuda.is_available():
    torch.cuda.empty_cache()

# 2. Load a FRESH, UNQUANTIZED version of the base model
# We use torch_dtype=torch.float16 (or bfloat16) and NO quantization_config
base_model_reload = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
    trust_remote_code=True,
    device_map="cpu", # Load to CPU first to avoid OOM, then merge moves to GPU if needed
)

# 3. Load the adapter onto the fresh 16-bit model
# Point this to the OUTPUT_DIR where your adapter_config.json is
adapter_model = PeftModel.from_pretrained(
    base_model_reload,
    OUTPUT_DIR,
)

# 4. Merge the weights
print("Merging weights... this may take a minute.")
merged_model = adapter_model.merge_and_unload()

# 5. Save the final result
merged_model.save_pretrained(MERGED_DIR)
tokenizer.save_pretrained(MERGED_DIR)

print(f"\n✅ Success! Merged model saved to: {MERGED_DIR}")