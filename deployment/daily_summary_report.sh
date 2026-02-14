#!/bin/bash
# Daily Summary Report - ONE email per day with maintenance + traffic summary
# Runs daily at 9 AM UTC via cron
# Critical issues (service down, disk full, etc.) are sent immediately by health_check.sh

APP_DIR="/home/riccardo/atari-rl-dashboard"
ADMIN_EMAIL="alerts@riccardocosentino.com"
HOSTNAME=$(hostname)
SERVICE="atari"
LOG_FILE="/var/log/atari/summary.log"

# Function to log
log_msg() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" >> "$LOG_FILE"
}

log_msg "Generating daily summary report..."

# --- Section 1: Maintenance Summary (last 24h) ---
MAINT_SUMMARY="=== MAINTENANCE (Last 24h) ===

Last Cleanup:    $(tail -5 /var/log/atari/cleanup.log 2>/dev/null | head -1)
Last Backup:     $(tail -5 /var/log/atari/backup.log 2>/dev/null | head -1)
Service Status:  $(systemctl is-active $SERVICE 2>/dev/null)
Service Uptime:  $(systemctl show $SERVICE --property=ActiveEnterTimestamp --value 2>/dev/null)
Restarts (24h):  $(journalctl -u $SERVICE --since "24 hours ago" 2>/dev/null | grep -c "Started\|Stopped" || echo "0")
"

# --- Section 2: Resource Usage ---
RESOURCE_SUMMARY="
=== RESOURCE USAGE ===
Disk:   $(df -h / | tail -1 | awk '{print $5 " used, " $4 " free"}')
Memory: $(free -h | grep Mem | awk '{print $3 " / " $2}')
Load:   $(uptime | awk -F'load average:' '{print $2}')
"

# --- Section 3: Traffic & Metrics (from Python report) ---
METRICS_REPORT=""
if [ -f "$APP_DIR/daily_metrics_report.py" ] && [ -f "$APP_DIR/data/rl_training.db" ]; then
    METRICS_REPORT=$(cd "$APP_DIR" && ./.venv/bin/python daily_metrics_report.py --no-email 2>/dev/null) || \
        METRICS_REPORT="(Metrics report unavailable - check database)"
else
    METRICS_REPORT="(Metrics report unavailable)"
fi

# --- Section 4: Recent Issues (last 24h) ---
ISSUES=$(grep -E "❌|CRITICAL" /var/log/atari/health.log 2>/dev/null | tail -20)
if [ -n "$ISSUES" ]; then
    ISSUES_SUMMARY="
=== RECENT ISSUES (Last 24h) ===
$ISSUES
"
else
    ISSUES_SUMMARY="
=== RECENT ISSUES (Last 24h) ===
No critical issues detected ✅
"
fi

# --- Assemble full report ---
FULL_REPORT="
================================================================================
           ATARI RL DASHBOARD - DAILY SUMMARY REPORT
                    $(date '+%Y-%m-%d %H:%M UTC')
================================================================================

$MAINT_SUMMARY
$RESOURCE_SUMMARY
$ISSUES_SUMMARY

================================================================================
                        TRAFFIC & METRICS
================================================================================

$METRICS_REPORT

================================================================================
  Dashboard: https://atari.riccardocosentino.com
  Note: Critical issues (service down, disk full) are sent immediately.
================================================================================
"

# Send single daily email
echo "$FULL_REPORT" | mail -s "[Atari App] Daily Summary - $(date +%Y-%m-%d)" "$ADMIN_EMAIL"

log_msg "Daily summary sent to $ADMIN_EMAIL"
