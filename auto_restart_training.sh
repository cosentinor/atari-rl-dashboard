#!/bin/bash
# Intelligent Auto-Restart Training Script
# Resumes training in appropriate mode based on stability state

set -e

echo "=============================================="
echo "INTELLIGENT AUTO-RESTART SCRIPT"
echo "Started at: $(date)"
echo "=============================================="

# Paths
STATE_MANAGER="/home/ubuntu/state_manager.sh"
LIGHT_SCRIPT="/home/ubuntu/atari-rl-dashboard/light_training.sh"
MEDIUM_SCRIPT="/home/ubuntu/atari-rl-dashboard/medium_training.sh"

cd /home/ubuntu/atari-rl-dashboard
source .venv/bin/activate

# Wait for GPU to be available
echo "Waiting for GPU to initialize..."
sleep 30

# Check if GPU is available
if ! nvidia-smi &> /dev/null; then
    echo "ERROR: GPU not available!"
    exit 1
fi

echo "GPU detected!"
echo ""

# Initialize state if needed
if [ ! -f "/home/ubuntu/scaling_state.json" ]; then
    echo "Initializing state file..."
    bash "$STATE_MANAGER" init
fi

# Record this reboot
echo "Recording reboot event..."
bash "$STATE_MANAGER" record_reboot

# Get current mode from state
CURRENT_MODE=$(bash "$STATE_MANAGER" get_mode)
REBOOTS=$(bash "$STATE_MANAGER" get_reboots)

echo "Current Mode: $CURRENT_MODE"
echo "Reboots (last hour): $REBOOTS"
echo ""

# Check if we need to fallback due to too many reboots
if [ $REBOOTS -gt 3 ] && [ "$CURRENT_MODE" = "medium" ]; then
    echo "⚠️  WARNING: $REBOOTS reboots in last hour (>3 threshold)"
    echo "Forcing fallback to LIGHT mode for stability"
    bash "$STATE_MANAGER" set_mode light
    CURRENT_MODE="light"
fi

# Launch training in appropriate mode
echo "Launching training in $CURRENT_MODE mode..."
echo ""

if [ "$CURRENT_MODE" = "medium" ]; then
    bash "$MEDIUM_SCRIPT"
else
    bash "$LIGHT_SCRIPT"
fi

sleep 10

echo ""
echo "=============================================="
echo "Training resumed successfully!"
echo "=============================================="
echo ""
echo "Running processes:"
ps aux | grep python | grep train | grep -v grep | wc -l
echo ""
echo "Mode: $CURRENT_MODE"
echo "Reboots (last hour): $REBOOTS"
echo ""
echo "Monitor with:"
echo "  tail -f ~/atari-rl-dashboard/training_*.log"
echo "  cat /home/ubuntu/scaling_state.json"
echo ""
echo "Check status:"
echo "  ps aux | grep train | grep -v grep"
echo "  nvidia-smi"
echo "=============================================="

# Log to file for debugging
echo "Auto-restart completed at $(date) in $CURRENT_MODE mode" >> /home/ubuntu/atari-rl-dashboard/auto_restart.log
