import os
import torch
import time
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM, 
    BitsAndBytesConfig, 
    TextStreamer
)

# Optimization environment variables
os.environ["TOKENIZERS_PARALLELISM"] = "false"

MODEL_PATH = "./marketing_agent_deepseek_v1_merged"

# 1. Load Tokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, fix_mistral_regex=True)

# 2. Optimized 4-bit Config
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True
)

# 3. Load Model
print(f"Loading model onto GPU (RTX 3070)...")
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True,
    attn_implementation="sdpa", 
)

# 4. Query
user_query = "How can I increase the retention rate for my new fitness app? Provide a structured marketing strategy."
messages = [{"role": "user", "content": user_query}]
prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

# Streaming setup
streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=False)

print("\n--- Generating Response ---\n")

start_time = time.time()
with torch.no_grad():
    outputs = model.generate(
        **inputs,
        streamer=streamer,
        max_new_tokens=512,
        temperature=0.6,
        top_p=0.95,
        use_cache=True, # Critical for speed
        pad_token_id=tokenizer.eos_token_id,
    )
end_time = time.time()

# 5. Performance Stats
total_tokens = len(outputs[0]) - len(inputs[0])
duration = end_time - start_time
tps = total_tokens / duration

print(f"\n\n--- Performance Stats ---")
print(f"Tokens Generated: {total_tokens}")
print(f"Time Taken: {duration:.2f}s")
print(f"Speed: {tps:.2f} tokens/sec")