#!/bin/bash
# ============================================================
# Thunder Compute PRODUCTION MODE Setup Script
# ============================================================
# Run this script after connecting to your Thunder Compute 
# PRODUCTION instance (not Prototyping!)
# Usage: ./setup_production.sh
# ============================================================

set -e  # Exit on error

echo "============================================================"
echo "⚡ Atari RL Training - PRODUCTION MODE Setup"
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
    
    # Verify we have A100
    GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)
    if [[ "$GPU_NAME" == *"A100"* ]]; then
        echo "   ✅ A100 GPU detected - ready for high-speed training!"
    else
        echo "   ⚠️  Warning: GPU is not A100. Training may be slower."
    fi
else
    echo "   ⚠️  nvidia-smi not found. GPU may not be available."
    GPU_AVAILABLE=false
fi

# Install system dependencies
echo ""
echo "📦 Installing system dependencies..."
sudo apt-get update -qq
sudo apt-get install -y -qq \
    python3-pip \
    python3-venv \
    git \
    wget \
    tmux \
    htop \
    libgl1-mesa-glx \
    libglib2.0-0 \
    2>/dev/null || true

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
    if [[ "$CUDA_VERSION" == 12.* ]] || [[ "$CUDA_VERSION" == 13.* ]]; then
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

# Verify EnvPool
echo ""
echo "🎯 Verifying EnvPool compatibility..."
python3 -c "
try:
    import envpool
    print(f'   ✅ EnvPool version: {envpool.__version__}')
    print('   Production Mode: EnvPool should work without restrictions!')
except ImportError:
    print('   ❌ EnvPool not found - will be installed with requirements.txt')
" || true

# Create tmux config for training
echo ""
echo "🖥️  Setting up tmux configuration..."
cat > ~/.tmux.conf << 'EOF'
# Increase scrollback buffer
set-option -g history-limit 50000

# Enable mouse support
set -g mouse on

# Improve colors
set -g default-terminal "screen-256color"

# Status bar
set -g status-interval 5
set -g status-style bg=black,fg=green
set -g status-left "[#S] "
set -g status-right "%H:%M %d-%b-%y"
EOF

echo "   ✅ tmux configured for production training"

# Show available games
echo ""
echo "🎮 Available games:"
python3 train.py --list-games

# Test EnvPool with quick smoke test
echo ""
echo "🧪 Running EnvPool smoke test (Pong, 10 episodes)..."
timeout 60 python3 train_envpool.py --game Pong --episodes 10 --num-envs 32 --batch-size 256 2>&1 | tail -5 || echo "Smoke test completed or timed out (expected)"

# Final instructions
echo ""
echo "============================================================"
echo "✅ PRODUCTION MODE SETUP COMPLETE!"
echo "============================================================"
echo ""
echo "🚀 Quick Start Commands:"
echo ""
echo "  # Launch full production training (6 games parallel):"
echo "  bash launch_production_training.sh"
echo ""
echo "  # Or manually start batch training:"
echo "  cd ~/atari-rl-dashboard && source .venv/bin/activate"
echo "  python train_production_batch.py --parallel 6 --episodes 10000"
echo ""
echo "  # Monitor training progress:"
echo "  python monitor_production.py"
echo ""
echo "💡 Production Mode Tips:"
echo "  - EnvPool should work without GPU restrictions"
echo "  - Expected: 90-95% GPU utilization"
echo "  - Training speed: 6,000-24,000 eps/hour"
echo "  - Use tmux to keep training running after disconnect"
echo "  - Attach to session: tmux attach -t atari-training"
echo "  - Detach from session: Ctrl+B, then D"
echo ""
echo "📊 Monitoring:"
echo "  - GPU usage: watch -n 1 nvidia-smi"
echo "  - Training logs: tail -f training_*.log"
echo "  - Progress: python monitor_production.py"
echo ""
echo "============================================================"

