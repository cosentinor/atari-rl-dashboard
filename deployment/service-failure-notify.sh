#!/bin/bash
# Service Failure Notification Script
# Called by systemd when atari.service fails

SERVICE="$1"
# Send only to alerts@ (works reliably)
# Set up forwarding in Office365 to riccardo@ and/or me.com
ADMIN_EMAIL="alerts@riccardocosentino.com"
HOSTNAME=$(hostname)
ALERT_COOLDOWN_DIR="/var/log/atari/alert_cooldowns"
ALERT_COOLDOWN_MINUTES=30  # Only send one email per 30 minutes for service failures

# Create cooldown directory if it doesn't exist
mkdir -p "$ALERT_COOLDOWN_DIR"

# Check if we should send alert (rate limiting)
ALERT_TYPE="service_failure"
COOLDOWN_FILE="$ALERT_COOLDOWN_DIR/${ALERT_TYPE}.last_sent"

should_send=true
if [ -f "$COOLDOWN_FILE" ]; then
    last_sent=$(cat "$COOLDOWN_FILE")
    current_time=$(date +%s)
    cooldown_seconds=$((ALERT_COOLDOWN_MINUTES * 60))
    time_diff=$((current_time - last_sent))
    
    if [ $time_diff -lt $cooldown_seconds ]; then
        should_send=false
    fi
fi

if [ "$should_send" = true ]; then
    # Get service status details
    STATUS=$(systemctl status $SERVICE --no-pager -l)
    JOURNAL=$(journalctl -u $SERVICE -n 50 --no-pager)

    MESSAGE="CRITICAL: The Atari RL Dashboard service has FAILED!

Time: $(date)
Server: $HOSTNAME
Service: $SERVICE

=== SERVICE STATUS ===
$STATUS

=== RECENT LOGS (Last 50 lines) ===
$JOURNAL

=== IMMEDIATE ACTIONS ===
1. SSH into server: ssh $HOSTNAME
2. Check logs: sudo journalctl -u $SERVICE -n 100
3. Check error log: tail -100 /var/log/atari/error.log
4. Restart manually: sudo systemctl restart $SERVICE

The health check script will attempt automatic recovery every 5 minutes.

Note: You will only receive this alert once per 30 minutes to avoid spam.
"

    echo "$MESSAGE" | mail -s "CRITICAL: Atari Service Failed on $HOSTNAME" "$ADMIN_EMAIL"
    echo $(date +%s) > "$COOLDOWN_FILE"
fi
