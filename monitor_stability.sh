#!/bin/bash
# Monitor script to track reboots and training stability

LOGFILE="stability_monitor.log"

echo "======================================" >> $LOGFILE
echo "Check Time: $(date)" >> $LOGFILE
echo "Server: tnr-0" >> $LOGFILE

# Get uptime
UPTIME=$(ssh tnr-0 "uptime -p")
echo "Uptime: $UPTIME" >> $LOGFILE

# Count processes
PROCESSES=$(ssh tnr-0 "ps aux | grep python | grep train | grep -v grep | wc -l")
echo "Training Processes: $PROCESSES" >> $LOGFILE

# GPU status
GPU_STATUS=$(ssh tnr-0 "nvidia-smi --query-gpu=utilization.gpu,memory.used,temperature.gpu --format=csv,noheader,nounits")
echo "GPU Status: $GPU_STATUS" >> $LOGFILE

# Check if uptime is less than 10 minutes (recent reboot)
UPTIME_MIN=$(ssh tnr-0 "awk '{print int(\$1/60)}' /proc/uptime")
if [ $UPTIME_MIN -lt 10 ]; then
    echo "⚠️  WARNING: Recent reboot detected! Uptime: $UPTIME_MIN minutes" >> $LOGFILE
    echo "⚠️  REBOOT ALERT at $(date)" >> reboot_alerts.log
fi

# Check if processes are low (crash)
if [ $PROCESSES -lt 100 ]; then
    echo "⚠️  WARNING: Low process count! Only $PROCESSES running" >> $LOGFILE
fi

echo "" >> $LOGFILE

# Display summary
echo "======================================"
echo "Stability Check: $(date)"
echo "Uptime: $UPTIME"
echo "Processes: $PROCESSES"
echo "GPU: $GPU_STATUS"
echo "======================================"
echo ""
echo "Full log: $LOGFILE"
echo "Reboots log: reboot_alerts.log"
