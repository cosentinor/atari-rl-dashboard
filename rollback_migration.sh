#!/bin/bash
# Rollback Script for Migration
# Run this on tnr-0 if migration fails

set -e

echo "=========================================="
echo "MIGRATION ROLLBACK SCRIPT"
echo "=========================================="
echo ""

cd ~/atari-rl-dashboard
source .venv/bin/activate

echo "Stopping any optimized training processes..."
pkill -f "train_production_batch" || true
pkill -f "train_envpool" || true
sleep 5

echo ""
echo "Restarting with original settings..."
echo "These will resume from latest checkpoints automatically."
echo ""

# Restart the 4 migrated games with original settings
echo "Starting MsPacman..."
nohup python train.py --game MsPacman --episodes 30000 > mspacman_train.log 2>&1 &
sleep 2

echo "Starting BeamRider..."
nohup python train.py --game BeamRider --episodes 20000 > beamrider_train.log 2>&1 &
sleep 2

echo "Starting Asteroids..."
nohup python train.py --game Asteroids --episodes 25000 > asteroids_train.log 2>&1 &
sleep 2

echo "Starting Enduro..."
nohup python train.py --game Enduro --episodes 15000 > enduro_train.log 2>&1 &
sleep 2

echo ""
echo "Rollback complete! Processes started:"
ps aux | grep "train.py" | grep -v grep

echo ""
echo "Monitor with:"
echo "  tail -f mspacman_train.log"
echo "  tail -f beamrider_train.log"
echo "  tail -f asteroids_train.log"
echo "  tail -f enduro_train.log"
echo ""
echo "=========================================="
