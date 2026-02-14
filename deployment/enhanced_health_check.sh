#!/bin/bash
# Enhanced Health Check Script with Email Notifications
# Runs every 5 minutes via cron

SERVICE="atari"
URL="http://localhost:5001"
ADMIN_EMAIL="riccardo@riccardocosentino.com"
HOSTNAME=$(hostname)
LOG_FILE="/var/log/atari/health.log"
ALERT_COOLDOWN_DIR="/var/log/atari/alert_cooldowns"
ALERT_COOLDOWN_HOURS=1  # Only send one email per hour for same issue

# Create cooldown directory if it doesn't exist
mkdir -p "$ALERT_COOLDOWN_DIR"

# Function to check if we should send alert (rate limiting)
should_send_alert() {
    local alert_type="$1"
    local cooldown_file="$ALERT_COOLDOWN_DIR/${alert_type}.last_sent"
    
    if [ ! -f "$cooldown_file" ]; then
        return 0  # No previous alert, send it
    fi
    
    local last_sent=$(cat "$cooldown_file")
    local current_time=$(date +%s)
    local cooldown_seconds=$((ALERT_COOLDOWN_HOURS * 3600))
    local time_diff=$((current_time - last_sent))
    
    if [ $time_diff -ge $cooldown_seconds ]; then
        return 0  # Cooldown expired, send it
    else
        return 1  # Still in cooldown, don't send
    fi
}

# Function to record that alert was sent
record_alert_sent() {
    local alert_type="$1"
    local cooldown_file="$ALERT_COOLDOWN_DIR/${alert_type}.last_sent"
    echo $(date +%s) > "$cooldown_file"
}

# Function to send email (with rate limiting)
send_alert() {
    local alert_type="$1"
    local subject="$2"
    local message="$3"
    
    if should_send_alert "$alert_type"; then
        echo "$message" | mail -s "[Atari App] $subject" "$ADMIN_EMAIL"
        record_alert_sent "$alert_type"
        log_msg "📧 Alert sent: $alert_type"
    else
        log_msg "⏸️ Alert suppressed (cooldown): $alert_type"
    fi
}

# Function to log with timestamp
log_msg() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" >> "$LOG_FILE"
}

# Check if service is running
if ! systemctl is-active --quiet $SERVICE; then
    log_msg "❌ Service $SERVICE is down! Attempting restart..."
    sudo systemctl restart $SERVICE
    sleep 3
    
    if systemctl is-active --quiet $SERVICE; then
        log_msg "✅ Service restarted successfully (no email - see daily summary)"
    else
        log_msg "❌ CRITICAL: Service restart failed!"
        send_alert "service_restart_failed" "🚨 CRITICAL: Atari Service Restart Failed" \
"The Atari RL Dashboard service is down and automatic restart FAILED.

Time: $(date)
Server: $HOSTNAME
Status: Service is NOT running

IMMEDIATE ACTION REQUIRED - Please investigate manually:
ssh $HOSTNAME 'sudo systemctl status atari'
ssh $HOSTNAME 'sudo journalctl -u atari -n 50'"
    fi
    exit 0
fi

# Check HTTP endpoint
HTTP_CODE=$(curl -s -o /dev/null -w "%{http_code}" --max-time 10 "$URL")
if [ "$HTTP_CODE" != "200" ]; then
    log_msg "❌ HTTP check failed (code: $HTTP_CODE). Restarting service..."
    sudo systemctl restart $SERVICE
    sleep 3
    
    # Verify restart
    NEW_HTTP_CODE=$(curl -s -o /dev/null -w "%{http_code}" --max-time 10 "$URL")
    if [ "$NEW_HTTP_CODE" = "200" ]; then
        log_msg "✅ Service restarted, HTTP now responding (no email - see daily summary)"
    else
        log_msg "❌ CRITICAL: HTTP still failing after restart (code: $NEW_HTTP_CODE)"
        send_alert "http_failed" "🚨 CRITICAL: Atari HTTP Endpoint Failure" \
"The Atari RL Dashboard HTTP endpoint is not responding even after restart.

Time: $(date)
Server: $HOSTNAME
HTTP Code: $NEW_HTTP_CODE (Expected: 200)

IMMEDIATE ACTION REQUIRED - Check logs:
ssh $HOSTNAME 'tail -100 /var/log/atari/error.log'"
    fi
else
    log_msg "✅ Health check passed (HTTP $HTTP_CODE)"
fi

# Check disk space (alert if >90%)
DISK_USAGE=$(df / | tail -1 | awk '{print $5}' | sed 's/%//')
if [ "$DISK_USAGE" -gt 90 ]; then
    log_msg "⚠️ Disk usage critical: ${DISK_USAGE}%"
    send_alert "disk_critical" "⚠️ Disk Space Critical on Atari Server" \
"Disk space is critically low on the Atari RL Dashboard server.

Time: $(date)
Server: $HOSTNAME
Disk Usage: ${DISK_USAGE}%

ACTION NEEDED:
1. SSH into server and investigate large files
2. Run cleanup script manually: ssh $HOSTNAME 'bash /home/riccardo/atari-rl-dashboard/deployment/enhanced_cleanup.sh'
3. Consider clearing old model checkpoints or logs

Note: You will only receive this alert once per hour to avoid spam."
fi

# Check memory usage (alert if >90%)
MEM_USAGE=$(free | grep Mem | awk '{printf "%.0f", ($3/$2) * 100}')
if [ "$MEM_USAGE" -gt 90 ]; then
    log_msg "⚠️ Memory usage high: ${MEM_USAGE}%"
    send_alert "memory_high" "⚠️ High Memory Usage on Atari Server" \
"Memory usage is high on the Atari RL Dashboard server.

Time: $(date)
Server: $HOSTNAME
Memory Usage: ${MEM_USAGE}%

This may indicate a memory leak or resource-intensive training.
Monitor the situation. Service will auto-restart if it crashes.

Note: You will only receive this alert once per hour to avoid spam."
fi
