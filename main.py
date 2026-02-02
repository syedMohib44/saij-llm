import os
import torch
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, TextIteratorStreamer
from threading import Thread

# Initialize FastAPI app
app = FastAPI(title="Marketing Agent API")

# Path to your merged model from Section 7 of train.py
MODEL_PATH = "./marketing_agent_deepseek_v1_merged"

# Load Tokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, fix_mistral_regex=True)

# Optimized 4-bit config for MojoHost (saves VRAM for concurrent requests)
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True
)

# Load Model onto GPU
print("Loading model for production inference...")
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True,
    attn_implementation="sdpa" # Production-stable for Llama 3.1
)

@app.post("/generate")
async def generate(request: Request):
    data = await request.json()
    user_query = data.get("prompt", "")
    
    # Structure the message for the model
    messages = [{"role": "user", "content": user_query}]
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    # Setup Streamer
    streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=False)
    
    generation_kwargs = dict(
        **inputs,
        streamer=streamer,
        max_new_tokens=1024,
        temperature=0.6,
        top_p=0.95,
        do_sample=True,
        use_cache=True,
        pad_token_id=tokenizer.eos_token_id
    )

    # Execute generation in a separate thread
    thread = Thread(target=model.generate, kwargs=generation_kwargs)
    thread.start()

    # Async generator to stream tokens to client
    async def stream_tokens():
        for token in streamer:
            yield token

    return StreamingResponse(stream_tokens(), media_type="text/plain")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)