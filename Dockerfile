# Use CUDA-enabled PyTorch base image
FROM pytorch/pytorch:2.5.1-cuda12.1-cudnn9-runtime

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y git && rm -rf /var/lib/apt/lists/*

# Install Python requirements
RUN pip install --no-cache-dir \
    transformers \
    fastapi \
    uvicorn \
    accelerate \
    bitsandbytes \
    scikit-learn \
    datasets \
    trl \
    peft

# Copy your merged model folder and API script
# Ensure the model folder name matches MODEL_PATH in main.py
# COPY ./marketing_agent_deepseek_v1_merged ./marketing_agent_deepseek_v1_merged
COPY main.py .

# Expose port 8000
EXPOSE 8000

# Start the API
CMD ["python", "main.py"]