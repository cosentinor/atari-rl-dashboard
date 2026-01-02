#!/bin/bash
# Intelligent Scaler - Main controller for auto-scaling training load
# Automatically scales between light and medium modes based on stability

set -e

# Configuration
LIGHT_STABLE_MINUTES=60      # Minutes of stability before scaling up
MAX_REBOOTS_PER_HOUR=3       # Reboot threshold for fallback
COOLDOWN_MINUTES=60          # Wait time before retry after fallback

# Paths
STATE_MANAGER="/home/ubuntu/state_manager.sh"
LIGHT_SCRIPT="/home/ubuntu/atari-rl-dashboard/light_training.sh"
MEDIUM_SCRIPT="/home/ubuntu/atari-rl-dashboard/medium_training.sh"
STOP_SCRIPT="/home/ubuntu/atari-rl-dashboard/stop_training.sh"
LOG_FILE="/home/ubuntu/scaling_decisions.log"

# Logging function
log_decision() {
    echo "$(date '+%Y-%m-%d %H:%M:%S') | $1" | tee -a "$LOG_FILE"
}

# Check if we're in the middle of a mode change
is_transitioning() {
    # If uptime < 5 minutes, we're probably transitioning
    local uptime_seconds=$(awk '{print int($1)}' /proc/uptime)
    if [ $uptime_seconds -lt 300 ]; then
        return 0  # True
    fi
    return 1  # False
}

# Main decision logic
make_decision() {
    log_decision "=============================================="
    log_decision "INTELLIGENT SCALER CHECK"
    
    # Initialize state if needed
    bash "$STATE_MANAGER" init
    
    # Get current state
    local current_mode=$(bash "$STATE_MANAGER" get_mode)
    local mode_uptime=$(bash "$STATE_MANAGER" get_uptime)
    local reboots_last_hour=$(bash "$STATE_MANAGER" get_reboots)
    local mode_uptime_min=$((mode_uptime / 60))
    
    log_decision "Current Mode: $current_mode"
    log_decision "Mode Uptime: ${mode_uptime_min} minutes"
    log_decision "Reboots (last hour): $reboots_last_hour"
    
    # Check if we're transitioning (just rebooted)
    if is_transitioning; then
        log_decision "Decision: WAIT (system just rebooted, stabilizing)"
        log_decision "=============================================="
        return 0
    fi
    
    # Decision tree
    case "$current_mode" in
        light)
            # Check if we should scale up to medium
            if [ $mode_uptime_min -ge $LIGHT_STABLE_MINUTES ]; then
                if [ $reboots_last_hour -le 1 ]; then
                    log_decision "Decision: SCALE UP to MEDIUM"
                    log_decision "Reason: Stable for ${mode_uptime_min} min with ${reboots_last_hour} reboots"
                    scale_to_medium
                else
                    log_decision "Decision: STAY LIGHT"
                    log_decision "Reason: Recent reboots detected ($reboots_last_hour in last hour)"
                fi
            else
                log_decision "Decision: STAY LIGHT"
                log_decision "Reason: Need $((LIGHT_STABLE_MINUTES - mode_uptime_min)) more minutes before scaling"
            fi
            ;;
            
        medium)
            # Check if we should fallback to light
            if [ $reboots_last_hour -gt $MAX_REBOOTS_PER_HOUR ]; then
                log_decision "Decision: FALLBACK to LIGHT"
                log_decision "Reason: Too many reboots ($reboots_last_hour > $MAX_REBOOTS_PER_HOUR threshold)"
                scale_to_light
            else
                log_decision "Decision: STAY MEDIUM"
                log_decision "Reason: Stable ($reboots_last_hour reboots, under $MAX_REBOOTS_PER_HOUR threshold)"
            fi
            ;;
            
        *)
            log_decision "Decision: ERROR - Unknown mode: $current_mode"
            log_decision "Defaulting to LIGHT mode"
            scale_to_light
            ;;
    esac
    
    log_decision "=============================================="
}

# Scale to medium mode
scale_to_medium() {
    log_decision "Executing scale-up to MEDIUM..."
    
    # Stop current training
    log_decision "  1. Stopping current training..."
    bash "$STOP_SCRIPT"
    
    sleep 5
    
    # Update state
    log_decision "  2. Updating state to MEDIUM..."
    bash "$STATE_MANAGER" set_mode medium
    
    # Start medium training
    log_decision "  3. Starting medium training..."
    bash "$MEDIUM_SCRIPT"
    
    sleep 10
    
    # Verify
    local process_count=$(ps aux | grep python | grep train | grep -v grep | wc -l)
    log_decision "  4. Verification: $process_count processes running"
    
    if [ $process_count -gt 50 ]; then
        log_decision "✅ Scale-up SUCCESSFUL"
    else
        log_decision "⚠️  Warning: Process count lower than expected"
    fi
}

# Scale to light mode
scale_to_light() {
    log_decision "Executing fallback to LIGHT..."
    
    # Stop current training
    log_decision "  1. Stopping current training..."
    bash "$STOP_SCRIPT"
    
    sleep 5
    
    # Update state
    log_decision "  2. Updating state to LIGHT..."
    bash "$STATE_MANAGER" set_mode light
    
    # Clear reboot counter (fresh start)
    log_decision "  3. Resetting reboot counter..."
    bash "$STATE_MANAGER" cleanup
    
    # Start light training
    log_decision "  4. Starting light training..."
    bash "$LIGHT_SCRIPT"
    
    sleep 10
    
    # Verify
    local process_count=$(ps aux | grep python | grep train | grep -v grep | wc -l)
    log_decision "  5. Verification: $process_count processes running"
    
    if [ $process_count -gt 10 ]; then
        log_decision "✅ Fallback SUCCESSFUL"
    else
        log_decision "⚠️  Warning: Process count lower than expected"
    fi
}

# Force scale up (for testing)
if [ "${1:-}" = "--force-scale-up" ]; then
    log_decision "MANUAL: Forcing scale-up to medium mode"
    scale_to_medium
    exit 0
fi

# Force scale down (for testing)
if [ "${1:-}" = "--force-scale-down" ]; then
    log_decision "MANUAL: Forcing scale-down to light mode"
    scale_to_light
    exit 0
fi

# Run decision logic
make_decision
