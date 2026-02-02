# Use a development image to allow on-the-fly kernel compilation for sm_120
FROM pytorch/pytorch:2.6.0-cuda12.4-cudnn9-devel-ubuntu22.04

WORKDIR /app

# Install system dependencies including compilers
RUN apt-get update && apt-get install -y git build-essential && rm -rf /var/lib/apt/lists/*

# Install requirements from the nightly index
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Force bitsandbytes to use the CUDA 12.4 backend for Blackwell
ENV BNB_CUDA_VERSION=124
ENV LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH

COPY . .

EXPOSE 8000
CMD ["python", "main.py"]