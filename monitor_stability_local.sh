#!/bin/bash
# Monitor script to track reboots and training stability (runs ON the server)

LOGFILE="/home/ubuntu/stability_monitor.log"
REBOOT_LOG="/home/ubuntu/reboot_alerts.log"

echo "======================================" >> $LOGFILE
echo "Check Time: $(date)" >> $LOGFILE

# Get uptime
UPTIME=$(uptime -p)
echo "Uptime: $UPTIME" >> $LOGFILE

# Count processes
PROCESSES=$(ps aux | grep python | grep train | grep -v grep | wc -l)
echo "Training Processes: $PROCESSES" >> $LOGFILE

# GPU status
GPU_STATUS=$(nvidia-smi --query-gpu=utilization.gpu,memory.used,temperature.gpu --format=csv,noheader,nounits 2>/dev/null || echo "N/A")
echo "GPU Status: $GPU_STATUS" >> $LOGFILE

# Check if uptime is less than 10 minutes (recent reboot)
UPTIME_MIN=$(awk '{print int($1/60)}' /proc/uptime)
if [ $UPTIME_MIN -lt 10 ]; then
    echo "⚠️  WARNING: Recent reboot detected! Uptime: $UPTIME_MIN minutes" >> $LOGFILE
    echo "⚠️  REBOOT ALERT at $(date) - Uptime: $UPTIME_MIN min" >> $REBOOT_LOG
fi

# Check if processes are low (crash)
if [ $PROCESSES -lt 100 ]; then
    echo "⚠️  WARNING: Low process count! Only $PROCESSES running" >> $LOGFILE
fi

echo "" >> $LOGFILE

# Display summary to stdout
echo "======================================"
echo "Stability Check: $(date)"
echo "Uptime: $UPTIME"
echo "Processes: $PROCESSES"
echo "GPU: $GPU_STATUS"
if [ $UPTIME_MIN -lt 10 ]; then
    echo "⚠️  REBOOT DETECTED: Uptime only $UPTIME_MIN minutes!"
fi
echo "======================================"
