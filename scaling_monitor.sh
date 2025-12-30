#!/bin/bash
# Scaling Monitor - Periodic checks for auto-scaling decisions
# Runs every 5 minutes to check if scaling actions needed

# Paths
STATE_MANAGER="/home/ubuntu/state_manager.sh"
INTELLIGENT_SCALER="/home/ubuntu/intelligent_scaler.sh"
LOG_FILE="/home/ubuntu/scaling_monitor.log"

# Logging
log_msg() {
    echo "$(date '+%Y-%m-%d %H:%M:%S') | $1" | tee -a "$LOG_FILE"
}

log_msg "=========================================="
log_msg "SCALING MONITOR CHECK"

# Get current state
if [ -f "/home/ubuntu/scaling_state.json" ]; then
    CURRENT_MODE=$(bash "$STATE_MANAGER" get_mode 2>/dev/null || echo "unknown")
    MODE_UPTIME=$(bash "$STATE_MANAGER" get_uptime 2>/dev/null || echo "0")
    REBOOTS=$(bash "$STATE_MANAGER" get_reboots 2>/dev/null || echo "0")
    MODE_UPTIME_MIN=$((MODE_UPTIME / 60))
else
    log_msg "State file not found, initializing..."
    bash "$STATE_MANAGER" init
    CURRENT_MODE="light"
    MODE_UPTIME_MIN=0
    REBOOTS=0
fi

# Get system stats
SERVER_UPTIME=$(awk '{print int($1/60)}' /proc/uptime)
PROCESS_COUNT=$(ps aux | grep python | grep train | grep -v grep | wc -l)
GPU_INFO=$(nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader,nounits 2>/dev/null || echo "N/A, N/A")

log_msg "Mode: $CURRENT_MODE | Mode Uptime: ${MODE_UPTIME_MIN}m | Reboots/hr: $REBOOTS"
log_msg "Server Uptime: ${SERVER_UPTIME}m | Processes: $PROCESS_COUNT | GPU: $GPU_INFO"

# Check for reboot (server uptime < 5 minutes)
if [ $SERVER_UPTIME -lt 5 ]; then
    log_msg "⚠️  REBOOT DETECTED (server uptime: ${SERVER_UPTIME}m)"
    bash "$STATE_MANAGER" record_reboot
    REBOOTS=$(bash "$STATE_MANAGER" get_reboots)
    log_msg "Updated reboot count: $REBOOTS in last hour"
fi

# Check for crashed training (no processes)
if [ $PROCESS_COUNT -lt 5 ]; then
    log_msg "⚠️  WARNING: Training processes low ($PROCESS_COUNT)"
    log_msg "Possible crash - intelligent scaler will handle restart"
fi

# Check for scale-up opportunity
if [ "$CURRENT_MODE" = "light" ] && [ $MODE_UPTIME_MIN -ge 60 ]; then
    log_msg "✅ Scale-up opportunity: Light mode stable for ${MODE_UPTIME_MIN} minutes"
    log_msg "Triggering intelligent scaler..."
    bash "$INTELLIGENT_SCALER"
fi

# Check for fallback need
if [ "$CURRENT_MODE" = "medium" ] && [ $REBOOTS -gt 3 ]; then
    log_msg "⚠️  Fallback needed: $REBOOTS reboots in last hour (>3 threshold)"
    log_msg "Triggering intelligent scaler for fallback..."
    bash "$INTELLIGENT_SCALER"
fi

log_msg "Check complete"
log_msg "=========================================="
