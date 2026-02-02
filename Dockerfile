FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime

WORKDIR /app

# Install git and build essentials for potential kernel compilation
RUN apt-get update && apt-get install -y git build-essential && rm -rf /var/lib/apt/lists/*

# Copy and install requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Force bitsandbytes to use the correct CUDA library if it fails to auto-detect
ENV BNB_CUDA_VERSION=124
ENV LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH

COPY . .

EXPOSE 8000
CMD ["python", "main.py"]
