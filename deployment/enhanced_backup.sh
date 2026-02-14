#!/bin/bash
# Enhanced Backup Script with Email Notifications
# Runs daily at 5 AM via cron

APP_DIR="/home/riccardo/atari-rl-dashboard"
BACKUP_DIR="/home/riccardo/atari-models-backup"
ADMIN_EMAIL="riccardo@riccardocosentino.com"
HOSTNAME=$(hostname)
LOG_FILE="/var/log/atari/backup.log"

# Function to log with timestamp
log_msg() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a "$LOG_FILE"
}

# Function to send email
send_notification() {
    local subject="$1"
    local message="$2"
    echo "$message" | mail -s "[Atari App] $subject" "$ADMIN_EMAIL"
}

log_msg "💾 Starting model backup..."

mkdir -p "$BACKUP_DIR"

# Track backup statistics
BACKED_UP_MODELS=0
TOTAL_SIZE=0

# Copy best models
for game_dir in $APP_DIR/saved_models/*/; do
    if [ -d "$game_dir" ]; then
        game_name=$(basename "$game_dir")
        mkdir -p "$BACKUP_DIR/$game_name"
        
        if [ -f "$game_dir/best_model.pt" ]; then
            cp "$game_dir/best_model.pt" "$BACKUP_DIR/$game_name/"
            SIZE=$(stat -f%z "$game_dir/best_model.pt" 2>/dev/null || stat -c%s "$game_dir/best_model.pt" 2>/dev/null)
            TOTAL_SIZE=$((TOTAL_SIZE + SIZE))
            BACKED_UP_MODELS=$((BACKED_UP_MODELS + 1))
            log_msg "  ✓ Backed up $game_name ($(numfmt --to=iec $SIZE 2>/dev/null || echo "${SIZE} bytes"))"
        fi
    fi
done

# Copy registry
if [ -f "$APP_DIR/saved_models/model_registry.json" ]; then
    cp "$APP_DIR/saved_models/model_registry.json" "$BACKUP_DIR/" 2>/dev/null
    log_msg "  ✓ Backed up model registry"
fi

# Commit and push to git (if configured)
cd "$BACKUP_DIR"
if [ -d ".git" ]; then
    log_msg "Committing to git repository..."
    git add -A
    if git commit -m "Automated backup: $(date '+%Y-%m-%d %H:%M')"; then
        if git push origin main 2>&1; then
            log_msg "✅ Backup committed and pushed to git"
            BACKUP_STATUS="Success - Pushed to Git"
        else
            log_msg "⚠️ Backup committed but push failed"
            BACKUP_STATUS="Warning - Commit OK, Push Failed"
        fi
    else
        log_msg "ℹ️ No changes to commit"
        BACKUP_STATUS="Success - No Changes"
    fi
else
    log_msg "⚠️ $BACKUP_DIR is not a git repository - local backup only"
    BACKUP_STATUS="Local Only - Git Not Configured"
fi

# Calculate total size
TOTAL_SIZE_HR=$(numfmt --to=iec $TOTAL_SIZE 2>/dev/null || echo "$TOTAL_SIZE bytes")

# Send summary email
SUMMARY="Model backup completed.

Time: $(date)
Server: $HOSTNAME

Backup Summary:
  • Models backed up: $BACKED_UP_MODELS games
  • Total size: $TOTAL_SIZE_HR
  • Backup location: $BACKUP_DIR
  • Status: $BACKUP_STATUS

Model Details:$(cd "$BACKUP_DIR" && find . -name "best_model.pt" -exec ls -lh {} \; 2>/dev/null | awk '{print "  • " $9 " - " $5}')

Next backup: Tomorrow at 5:00 AM UTC
"

# Routine backup - no email (included in daily summary report)
log_msg "📧 Backup complete (summary in daily report)"
