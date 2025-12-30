#!/bin/bash
# State Management for Intelligent Auto-Scaling
# Manages scaling_state.json with atomic operations

STATE_FILE="/home/ubuntu/scaling_state.json"
LOCK_FILE="/home/ubuntu/scaling_state.lock"

# Acquire lock for atomic operations
acquire_lock() {
    local timeout=30
    local elapsed=0
    while [ -f "$LOCK_FILE" ] && [ $elapsed -lt $timeout ]; do
        sleep 1
        elapsed=$((elapsed + 1))
    done
    
    if [ $elapsed -ge $timeout ]; then
        echo "ERROR: Could not acquire lock after ${timeout}s"
        return 1
    fi
    
    touch "$LOCK_FILE"
    return 0
}

# Release lock
release_lock() {
    rm -f "$LOCK_FILE"
}

# Initialize state file if it doesn't exist
initialize_state() {
    if [ ! -f "$STATE_FILE" ]; then
        cat > "$STATE_FILE" << EOF
{
  "current_mode": "light",
  "mode_start_time": $(date +%s),
  "reboots_in_current_hour": 0,
  "reboot_times": [],
  "last_scale_attempt": $(date +%s),
  "scale_attempts": 0,
  "total_reboots": 0
}
EOF
        echo "State file initialized at $STATE_FILE"
    fi
}

# Get current mode
get_mode() {
    if [ ! -f "$STATE_FILE" ]; then
        echo "light"
        return
    fi
    python3 -c "import json; print(json.load(open('$STATE_FILE'))['current_mode'])" 2>/dev/null || echo "light"
}

# Get mode uptime in seconds
get_mode_uptime() {
    if [ ! -f "$STATE_FILE" ]; then
        echo "0"
        return
    fi
    
    local start_time=$(python3 -c "import json; print(json.load(open('$STATE_FILE'))['mode_start_time'])" 2>/dev/null || echo "0")
    local current_time=$(date +%s)
    echo $((current_time - start_time))
}

# Set mode
set_mode() {
    local new_mode=$1
    
    acquire_lock || return 1
    
    initialize_state
    
    python3 << EOF
import json
from datetime import datetime

with open('$STATE_FILE', 'r') as f:
    state = json.load(f)

state['current_mode'] = '$new_mode'
state['mode_start_time'] = $(date +%s)

with open('$STATE_FILE', 'w') as f:
    json.dump(state, f, indent=2)

print(f"Mode changed to: $new_mode at {datetime.now()}")
EOF
    
    release_lock
    
    echo "$(date): Mode set to $new_mode" >> /home/ubuntu/scaling_decisions.log
}

# Record reboot
record_reboot() {
    acquire_lock || return 1
    
    initialize_state
    
    python3 << EOF
import json
from datetime import datetime

with open('$STATE_FILE', 'r') as f:
    state = json.load(f)

current_time = $(date +%s)
state['reboot_times'].append(current_time)
state['total_reboots'] = state.get('total_reboots', 0) + 1

# Keep only reboots from last hour (3600 seconds)
one_hour_ago = current_time - 3600
state['reboot_times'] = [t for t in state['reboot_times'] if t > one_hour_ago]
state['reboots_in_current_hour'] = len(state['reboot_times'])

with open('$STATE_FILE', 'w') as f:
    json.dump(state, f, indent=2)

print(f"Reboot recorded. Total in last hour: {state['reboots_in_current_hour']}")
EOF
    
    release_lock
    
    echo "$(date): Reboot recorded" >> /home/ubuntu/scaling_decisions.log
}

# Get reboots in last hour
get_reboots_last_hour() {
    if [ ! -f "$STATE_FILE" ]; then
        echo "0"
        return
    fi
    
    python3 << EOF
import json

try:
    with open('$STATE_FILE', 'r') as f:
        state = json.load(f)
    
    current_time = $(date +%s)
    one_hour_ago = current_time - 3600
    recent_reboots = [t for t in state.get('reboot_times', []) if t > one_hour_ago]
    print(len(recent_reboots))
except:
    print(0)
EOF
}

# Clear old reboots (cleanup)
cleanup_old_reboots() {
    acquire_lock || return 1
    
    if [ ! -f "$STATE_FILE" ]; then
        release_lock
        return
    fi
    
    python3 << EOF
import json

with open('$STATE_FILE', 'r') as f:
    state = json.load(f)

current_time = $(date +%s)
one_hour_ago = current_time - 3600
state['reboot_times'] = [t for t in state.get('reboot_times', []) if t > one_hour_ago]
state['reboots_in_current_hour'] = len(state['reboot_times'])

with open('$STATE_FILE', 'w') as f:
    json.dump(state, f, indent=2)
EOF
    
    release_lock
}

# Get full state as JSON
get_state_json() {
    if [ ! -f "$STATE_FILE" ]; then
        initialize_state
    fi
    cat "$STATE_FILE" | python3 -m json.tool
}

# Main CLI interface
case "${1:-}" in
    init)
        initialize_state
        ;;
    get_mode)
        get_mode
        ;;
    set_mode)
        set_mode "$2"
        ;;
    get_uptime)
        get_mode_uptime
        ;;
    record_reboot)
        record_reboot
        ;;
    get_reboots)
        get_reboots_last_hour
        ;;
    cleanup)
        cleanup_old_reboots
        ;;
    show)
        get_state_json
        ;;
    *)
        echo "Usage: $0 {init|get_mode|set_mode|get_uptime|record_reboot|get_reboots|cleanup|show}"
        echo ""
        echo "Commands:"
        echo "  init              - Initialize state file"
        echo "  get_mode          - Get current mode (light|medium)"
        echo "  set_mode <mode>   - Set mode (light|medium)"
        echo "  get_uptime        - Get seconds in current mode"
        echo "  record_reboot     - Record a reboot event"
        echo "  get_reboots       - Get reboot count in last hour"
        echo "  cleanup           - Remove old reboot records"
        echo "  show              - Display full state"
        exit 1
        ;;
esac
