#!/bin/bash
# Light training mode - minimal load to test stability

echo "========================================"
echo "STARTING LIGHT TRAINING MODE"
echo "Testing with minimal load to check stability"
echo "========================================"

cd /home/ubuntu/atari-rl-dashboard
source .venv/bin/activate

echo "Starting 2 games with reduced parallel environments..."
echo ""

# Start just 2 games with much smaller parallel settings
echo "1. MsPacman (optimized, but light load)"
nohup python train_envpool.py \
    --game MsPacman \
    --episodes 30000 \
    --num-envs 32 \
    --batch-size 512 \
    > training_mspacman.log 2>&1 &
echo "   Started MsPacman (32 envs, batch 512)"

sleep 5

echo "2. Pong (single environment - minimal load)"
nohup python train.py \
    --game Pong \
    --episodes 3000 \
    > pong_train.log 2>&1 &
echo "   Started Pong (1 env, batch 32)"

sleep 3

echo ""
echo "========================================"
echo "Light training started!"
echo "========================================"
echo ""
echo "Running processes:"
ps aux | grep python | grep train | grep -v grep | wc -l
echo ""
echo "GPU Status:"
nvidia-smi --query-gpu=utilization.gpu,memory.used,temperature.gpu --format=csv,noheader
echo ""
echo "This light load uses:"
echo "  - Only 2 games (vs 8)"
echo "  - 32 parallel envs (vs 256)"
echo "  - Smaller batch size (512 vs 2048)"
echo "  - ~50-60 processes total (vs 365)"
echo ""
echo "Monitor stability for 30 minutes"
echo "If stable, we can gradually increase"
echo "========================================"
