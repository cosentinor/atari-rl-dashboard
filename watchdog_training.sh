#!/bin/bash
# Watchdog script to monitor and restart crashed training processes
# Run this with: watch -n 300 bash watchdog_training.sh (every 5 minutes)

cd /home/ubuntu/atari-rl-dashboard
source .venv/bin/activate

LOG_FILE="/home/ubuntu/atari-rl-dashboard/watchdog.log"

echo "=== Watchdog Check: $(date) ===" >> $LOG_FILE

# Count running training processes
PROCESS_COUNT=$(ps aux | grep python | grep train | grep -v grep | wc -l)

echo "Running processes: $PROCESS_COUNT" >> $LOG_FILE

# If no processes are running, restart everything
if [ $PROCESS_COUNT -eq 0 ]; then
    echo "WARNING: No training processes found! Restarting..." >> $LOG_FILE
    bash /home/ubuntu/atari-rl-dashboard/auto_restart_training.sh >> $LOG_FILE 2>&1
    echo "Training restarted by watchdog" >> $LOG_FILE
else
    echo "Training is running normally" >> $LOG_FILE
fi

# Check GPU utilization
GPU_UTIL=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits)
echo "GPU utilization: ${GPU_UTIL}%" >> $LOG_FILE

# Alert if GPU is idle but processes exist
if [ $PROCESS_COUNT -gt 0 ] && [ $GPU_UTIL -lt 10 ]; then
    echo "WARNING: GPU idle but processes running - possible hang!" >> $LOG_FILE
fi

echo "" >> $LOG_FILE
