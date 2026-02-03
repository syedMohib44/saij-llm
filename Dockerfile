FROM nvcr.io/nvidia/pytorch:25.01-py3

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y git && rm -rf /var/lib/apt/lists/*

# Copy and install requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Ensure bitsandbytes is the Blackwell-specific version
RUN pip install --upgrade bitsandbytes>=0.45.0

COPY . .

EXPOSE 8000
CMD ["python", "main.py"]