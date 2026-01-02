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
PYTHONUNBUFFERED=1 nohup python -u train_production_batch.py \
    --parallel 4 \
    --batch-size 1024 \
    --num-envs 64 \
    --resume \
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
echo "  - Auto-selected from remaining games"
echo "  - Resumes from latest checkpoints"
echo ""
echo "Monitor with:"
echo "  tail -f medium_training.log"
echo "  tail -f training_*.log"
echo ""
echo "This will train up to 4 games in parallel with balanced settings."
echo "========================================"
