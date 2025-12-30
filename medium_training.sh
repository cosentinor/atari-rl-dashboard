#!/bin/bash
# Medium Training Mode - Balanced load for stability and speed

echo "========================================"
echo "STARTING MEDIUM TRAINING MODE"
echo "Balanced configuration for stability"
echo "========================================"

cd /home/ubuntu/atari-rl-dashboard
source .venv/bin/activate

echo "Configuration:"
echo "  Games: 4"
echo "  Parallel Envs: 64 per game"
echo "  Batch Size: 1024"
echo "  Expected Processes: ~100-120"
echo "  Expected GPU Memory: 10-15 GB"
echo ""

echo "Starting 4 games with moderate settings..."
echo ""

# Start optimized batch training with medium settings
nohup python train_production_batch.py \
    --parallel 4 \
    --batch-size 1024 \
    --num-envs 64 \
    --games MsPacman Asteroids BeamRider Seaquest \
    > medium_training.log 2>&1 &

sleep 5

echo ""
echo "========================================"
echo "Medium training started!"
echo "========================================"
echo ""
echo "Running processes:"
ps aux | grep python | grep train | grep -v grep | wc -l
echo ""
echo "GPU Status:"
nvidia-smi --query-gpu=utilization.gpu,memory.used,temperature.gpu --format=csv,noheader
echo ""
echo "Games training:"
echo "  1. MsPacman (30,000 episodes)"
echo "  2. Asteroids (25,000 episodes)"
echo "  3. BeamRider (20,000 episodes)"
echo "  4. Seaquest (25,000 episodes)"
echo ""
echo "Monitor with:"
echo "  tail -f medium_training.log"
echo "  tail -f training_mspacman.log"
echo ""
echo "This will train 4 games in parallel with balanced settings."
echo "Should complete in ~5-7 days if stable."
echo "========================================"
