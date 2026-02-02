# Use NVIDIA's optimized Blackwell container (requires NGC account/login if private)
# This image already contains CUDA 12.8 and Blackwell-compatible PyTorch
FROM nvcr.io/nvidia/pytorch:25.01-py3

WORKDIR /app

# System dependencies
RUN apt-get update && apt-get install -y git && rm -rf /var/lib/apt/lists/*

# Copy and install your Python requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Ensure bitsandbytes is the latest for Blackwell support
RUN pip install --upgrade bitsandbytes>=0.45.0

COPY . .

EXPOSE 8000
CMD ["python", "main.py"]