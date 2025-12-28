#!/bin/bash
# ============================================================
# Thunder Compute Setup Script for Atari RL Training
# ============================================================
# Run this script after connecting to your Thunder Compute instance
# Usage: ./setup_thunder_compute.sh
# ============================================================

set -e  # Exit on error

echo "============================================================"
echo "⚡ Atari RL Training - Thunder Compute Setup"
echo "============================================================"

# Check if we're on the Thunder Compute instance
echo ""
echo "📍 System Info:"
echo "   Hostname: $(hostname)"
echo "   User: $(whoami)"
echo "   Home: $HOME"

# Check GPU
echo ""
echo "🎮 GPU Status:"
if command -v nvidia-smi &> /dev/null; then
    nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
    GPU_AVAILABLE=true
else
    echo "   ⚠️  nvidia-smi not found. GPU may not be available."
    GPU_AVAILABLE=false
fi

# Install system dependencies
echo ""
echo "📦 Installing system dependencies..."
sudo apt-get update -qq
sudo apt-get install -y -qq python3-pip python3-venv git wget libgl1-mesa-glx libglib2.0-0 2>/dev/null || true

# Clone the repository
echo ""
echo "📥 Cloning repository..."
cd ~
if [ -d "atari-rl-dashboard" ]; then
    echo "   Directory exists, pulling latest changes..."
    cd atari-rl-dashboard
    git pull
else
    git clone https://github.com/cosentinor/atari-rl-dashboard.git
    cd atari-rl-dashboard
fi

# Create virtual environment
echo ""
echo "🐍 Setting up Python virtual environment..."
python3 -m venv .venv
source .venv/bin/activate

# Upgrade pip
pip install --upgrade pip -q

# Install requirements
echo ""
echo "📦 Installing Python dependencies..."
pip install -r requirements.txt -q

# Install PyTorch with CUDA
echo ""
echo "🔥 Installing PyTorch with CUDA support..."
if [ "$GPU_AVAILABLE" = true ]; then
    # Detect CUDA version
    CUDA_VERSION=$(nvidia-smi | grep -oP 'CUDA Version: \K[0-9]+\.[0-9]+' || echo "12.1")
    echo "   Detected CUDA version: $CUDA_VERSION"
    
    # Install appropriate PyTorch
    if [[ "$CUDA_VERSION" == 12.* ]]; then
        pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121 -q
    elif [[ "$CUDA_VERSION" == 11.* ]]; then
        pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118 -q
    else
        pip install torch torchvision -q
    fi
else
    pip install torch torchvision -q
fi

# Verify installation
echo ""
echo "✅ Verifying installation..."
python3 -c "
import torch
print(f'   PyTorch version: {torch.__version__}')
print(f'   CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'   CUDA devices: {torch.cuda.device_count()}')
    for i in range(torch.cuda.device_count()):
        print(f'   Device {i}: {torch.cuda.get_device_name(i)}')
"

# Show available games
echo ""
echo "🎮 Available games:"
python3 train.py --list-games

# Final instructions
echo ""
echo "============================================================"
echo "✅ SETUP COMPLETE!"
echo "============================================================"
echo ""
echo "To start training, run:"
echo ""
echo "  # Activate environment (if not already active)"
echo "  cd ~/atari-rl-dashboard && source .venv/bin/activate"
echo ""
echo "  # Train Pong (POC - fastest to see results)"
echo "  python train.py --game Pong --episodes 3000"
echo ""
echo "  # Train Breakout (medium difficulty)"
echo "  python train.py --game Breakout --episodes 5000"
echo ""
echo "  # Train all games"
echo "  python train.py --game all"
echo ""
echo "💡 Tips:"
echo "  - Use 'tmux' or 'screen' to keep training running after disconnect"
echo "  - Models auto-save every 90 seconds and every 100 episodes"
echo "  - Monitor GPU: watch -n 1 nvidia-smi"
echo ""
echo "============================================================"

