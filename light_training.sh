#!/bin/bash
# Light training mode - minimal load to test stability

echo "========================================"
echo "STARTING LIGHT TRAINING MODE"
echo "Testing with minimal load to check stability"
echo "========================================"

cd /home/ubuntu/atari-rl-dashboard
source .venv/bin/activate

echo "Starting low-load batch with reduced parallelism..."
echo ""

# Low-load batch runner (auto-resume from checkpoints)
PYTHONUNBUFFERED=1 nohup python -u train_production_batch.py \
    --parallel 2 \
    --num-envs 32 \
    --batch-size 512 \
    --resume \
    > light_training.log 2>&1 &

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
echo "  - 2 parallel games"
echo "  - 32 parallel envs per game"
echo "  - Batch size 512"
echo "  - Auto-resume from latest checkpoints"
echo ""
echo "Monitor with:"
echo "  tail -f light_training.log"
echo "  tail -f training_*.log"
echo ""
echo "Monitor stability for 30 minutes"
echo "If stable, we can gradually increase"
echo "========================================"
