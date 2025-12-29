#!/bin/bash
# ============================================================
# Production Training Launcher
# ============================================================
# One-command launcher for parallel EnvPool training
# Usage: bash launch_production_training.sh
# ============================================================

set -e

echo "============================================================"
echo "🚀 PRODUCTION TRAINING LAUNCHER"
echo "============================================================"

# Check if we're in the right directory
if [ ! -f "train_production_batch.py" ]; then
    echo "❌ Error: Must run from atari-rl-dashboard directory"
    echo "   cd ~/atari-rl-dashboard && bash launch_production_training.sh"
    exit 1
fi

# Check if virtual environment exists
if [ ! -d ".venv" ]; then
    echo "❌ Error: Virtual environment not found"
    echo "   Run setup_production.sh first"
    exit 1
fi

# Activate virtual environment
echo "🐍 Activating virtual environment..."
source .venv/bin/activate

# Check for GPU
echo "🎮 Checking GPU..."
if command -v nvidia-smi &> /dev/null; then
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
    GPU_AVAILABLE=true
else
    echo "⚠️  Warning: nvidia-smi not found"
    GPU_AVAILABLE=false
fi

# Check for existing tmux session
TMUX_SESSION="atari-training"
if tmux has-session -t $TMUX_SESSION 2>/dev/null; then
    echo ""
    echo "⚠️  Tmux session '$TMUX_SESSION' already exists!"
    echo ""
    read -p "Options: [a]ttach, [k]ill and restart, [c]ancel? " -n 1 -r
    echo
    
    if [[ $REPLY =~ ^[Aa]$ ]]; then
        echo "Attaching to existing session..."
        tmux attach -t $TMUX_SESSION
        exit 0
    elif [[ $REPLY =~ ^[Kk]$ ]]; then
        echo "Killing existing session..."
        tmux kill-session -t $TMUX_SESSION
    else
        echo "Cancelled."
        exit 0
    fi
fi

# Configuration
echo ""
echo "📋 Training Configuration:"
echo "   Parallel games: 6"
echo "   Environments per game: 256"
echo "   Batch size: 1024"
echo "   Total games: 10"
echo ""
echo "⏱️  Expected Duration:"
echo "   Fast games (Breakout, Pong, Freeway): 0.4-1.5 hours each"
echo "   Medium games: 2-3 hours each"
echo "   Long games (MsPacman): 4-5 hours"
echo "   Total: 2-3 days for all 10 games"
echo ""
echo "💾 Models will be saved to: saved_models/"
echo "📝 Logs will be written to: training_*.log"
echo ""

# Confirm start
read -p "Start production training? [Y/n] " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]] && [[ ! -z $REPLY ]]; then
    echo "Cancelled."
    exit 0
fi

echo ""
echo "🚀 Starting training in tmux session '$TMUX_SESSION'..."
echo ""

# Create tmux session and start training
tmux new-session -d -s $TMUX_SESSION -n "batch-training"

# Send command to tmux session
tmux send-keys -t $TMUX_SESSION:batch-training "cd ~/atari-rl-dashboard" C-m
tmux send-keys -t $TMUX_SESSION:batch-training "source .venv/bin/activate" C-m
tmux send-keys -t $TMUX_SESSION:batch-training "python train_production_batch.py --parallel 6 --num-envs 256 --batch-size 1024" C-m

# Create monitoring window
tmux new-window -t $TMUX_SESSION -n "monitor"
tmux send-keys -t $TMUX_SESSION:monitor "cd ~/atari-rl-dashboard" C-m
tmux send-keys -t $TMUX_SESSION:monitor "source .venv/bin/activate" C-m
tmux send-keys -t $TMUX_SESSION:monitor "sleep 10 && python monitor_production.py --watch --interval 60" C-m

# Create GPU monitoring window
tmux new-window -t $TMUX_SESSION -n "gpu"
tmux send-keys -t $TMUX_SESSION:gpu "watch -n 2 nvidia-smi" C-m

# Create logs window
tmux new-window -t $TMUX_SESSION -n "logs"
tmux send-keys -t $TMUX_SESSION:logs "cd ~/atari-rl-dashboard" C-m

# Select first window
tmux select-window -t $TMUX_SESSION:batch-training

echo "✅ Training started in tmux session!"
echo ""
echo "============================================================"
echo "📖 TMUX QUICK REFERENCE"
echo "============================================================"
echo ""
echo "Attach to session:"
echo "  tmux attach -t $TMUX_SESSION"
echo ""
echo "Detach from session (inside tmux):"
echo "  Press: Ctrl+B, then D"
echo ""
echo "Switch between windows (inside tmux):"
echo "  Ctrl+B, then 0-3 (window number)"
echo "  Ctrl+B, then N (next window)"
echo "  Ctrl+B, then P (previous window)"
echo ""
echo "Windows created:"
echo "  0: batch-training - Main orchestrator"
echo "  1: monitor        - Real-time progress"
echo "  2: gpu            - GPU utilization"
echo "  3: logs           - Training logs"
echo ""
echo "Kill session (if needed):"
echo "  tmux kill-session -t $TMUX_SESSION"
echo ""
echo "============================================================"
echo ""
echo "🎯 Next Steps:"
echo ""
echo "1. Attach to session:"
echo "   tmux attach -t $TMUX_SESSION"
echo ""
echo "2. Monitor progress remotely (from local machine):"
echo "   python monitor_production.py --host tnr-prod --watch"
echo ""
echo "3. Check specific log:"
echo "   tail -f training_mspacman.log"
echo ""
echo "============================================================"
echo ""
echo "💡 The training will continue even if you disconnect!"
echo "   Just reconnect and attach to the tmux session."
echo ""

