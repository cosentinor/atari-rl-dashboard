#!/bin/bash
# Daily Status Report - Comprehensive system health summary
# Runs daily at 8 AM via cron

APP_DIR="/home/riccardo/atari-rl-dashboard"
ADMIN_EMAIL="riccardo@riccardocosentino.com"
HOSTNAME=$(hostname)
SERVICE="atari"

# Gather system information
UPTIME=$(uptime -p)
LOAD=$(uptime | awk -F'load average:' '{print $2}')
DISK_USAGE=$(df -h / | tail -1 | awk '{print $5 " used, " $4 " free"}')
MEMORY=$(free -h | grep Mem | awk '{print $3 " used / " $2 " total"}')
SERVICE_STATUS=$(systemctl is-active $SERVICE)
SERVICE_UPTIME=$(systemctl show $SERVICE --property=ActiveEnterTimestamp --value)

# Check recent restarts (last 24 hours)
RECENT_RESTARTS=$(journalctl -u $SERVICE --since "24 hours ago" | grep -c "Started\|Stopped" || echo "0")

# Get active training sessions
TRAINING_SESSIONS=$(ps aux | grep -E 'train.*\.py' | grep -v grep | wc -l)

# Database size
DB_SIZE=$(ls -lh $APP_DIR/data/rl_training.db 2>/dev/null | awk '{print $5}' || echo "N/A")

# Count saved models
MODEL_COUNT=$(find $APP_DIR/saved_models -name "best_model.pt" 2>/dev/null | wc -l)

# Recent health check issues (last 24 hours)
HEALTH_ISSUES=$(grep -E "❌|CRITICAL" /var/log/atari/health.log 2>/dev/null | tail -5)

# Get recent HTTP requests count (from app log)
HTTP_REQUESTS=$(grep "GET\|POST" /var/log/atari/app.log 2>/dev/null | wc -l)

# Generate report
REPORT="Daily Status Report - Atari RL Dashboard

=== SYSTEM HEALTH ===
Time: $(date)
Server: $HOSTNAME
System Uptime: $UPTIME
Load Average:$LOAD

=== SERVICE STATUS ===
Service Status: $SERVICE_STATUS
Service Started: $SERVICE_UPTIME
Restarts (24h): $RECENT_RESTARTS
HTTP Requests (total): $HTTP_REQUESTS

=== RESOURCE USAGE ===
Disk Usage: $DISK_USAGE
Memory Usage: $MEMORY
CPU Load: $(top -bn1 | grep "Cpu(s)" | sed "s/.*, *\([0-9.]*\)%* id.*/\1/" | awk '{print 100 - $1"%"}')

=== APPLICATION STATUS ===
Active Training Sessions: $TRAINING_SESSIONS
Saved Models: $MODEL_COUNT games
Database Size: $DB_SIZE

=== RECENT ACTIVITY ===
Last Backup: $(tail -1 /var/log/atari/backup.log 2>/dev/null | awk '{print $1, $2}')
Last Cleanup: $(tail -1 /var/log/atari/cleanup.log 2>/dev/null | awk '{print $1, $2}')
Last Health Check: $(tail -1 /var/log/atari/health.log 2>/dev/null | awk '{print $1, $2}')

=== RECENT ISSUES (24h) ===
$(if [ -n "$HEALTH_ISSUES" ]; then echo "$HEALTH_ISSUES"; else echo "No issues detected ✅"; fi)

=== QUICK LINKS ===
Dashboard: http://atari.riccardocosentino.com:5001
SSH Access: ssh $HOSTNAME

=== MAINTENANCE SCHEDULE ===
• Health Check: Every 5 minutes
• Cleanup: Daily at 4:00 AM UTC
• Backup: Daily at 5:00 AM UTC
• Status Report: Daily at 8:00 AM UTC

---
This is an automated report. Reply to this email if you need assistance.
"

# Routine status - no email (consolidated into daily_summary_report.sh)
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Daily status generated (included in daily summary)" >> /var/log/atari/health.log
