#!/bin/bash
# Gracefully stop all training processes

echo "========================================"
echo "STOPPING ALL TRAINING"
echo "========================================"

cd /home/ubuntu/atari-rl-dashboard

echo "Sending SIGTERM for graceful shutdown..."
pkill -TERM -f 'python.*train'

echo "Waiting 30 seconds for checkpoints to save..."
sleep 30

echo ""
echo "Checking for remaining processes..."
REMAINING=$(ps aux | grep python | grep train | grep -v grep | wc -l)

if [ $REMAINING -gt 0 ]; then
    echo "⚠️  $REMAINING processes still running, force killing..."
    pkill -KILL -f 'python.*train'
    sleep 3
fi

echo ""
echo "========================================"
echo "Training stopped"
echo "========================================"
echo ""
echo "Process count: $(ps aux | grep python | grep train | grep -v grep | wc -l)"
echo ""
echo "Latest checkpoints saved:"
ls -lht ~/atari-rl-dashboard/saved_models/*/checkpoint*.pt | head -8
echo ""
echo "All progress has been saved to checkpoints."
echo "========================================"
