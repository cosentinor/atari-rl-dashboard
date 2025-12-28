# Atari RL Training - Docker Image
# For headless training on Thunder Compute or any NVIDIA GPU cloud

# Use NVIDIA CUDA base image with Python
FROM nvidia/cuda:12.1-cudnn8-runtime-ubuntu22.04

# Prevent interactive prompts during package installation
ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3.11 \
    python3-pip \
    python3.11-venv \
    git \
    wget \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    && rm -rf /var/lib/apt/lists/*

# Set Python 3.11 as default
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.11 1 \
    && update-alternatives --install /usr/bin/pip pip /usr/bin/pip3 1

# Set working directory
WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# Install PyTorch with CUDA support
RUN pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cu121

# Copy application code
COPY *.py ./
COPY config.py ./

# Create directories for models and data
RUN mkdir -p saved_models data

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV CUDA_VISIBLE_DEVICES=0

# Default command: show help
CMD ["python", "train.py", "--help"]

# Example usage:
# docker build -t atari-rl .
# docker run --gpus all atari-rl python train.py --game Pong --episodes 3000
# docker run --gpus all -v $(pwd)/saved_models:/app/saved_models atari-rl python train.py --game Pong --episodes 3000

