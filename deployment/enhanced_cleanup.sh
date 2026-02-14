#!/bin/bash
# Enhanced Cleanup Script with Archiving and Email Notifications
# Runs every 6 hours via cron

APP_DIR="/home/riccardo/atari-rl-dashboard"
BACKUP_DIR="/home/riccardo/atari-models-backup"
ARCHIVE_DIR="$BACKUP_DIR/archived_checkpoints"
ARCHIVE_LOGS_DIR="$BACKUP_DIR/archived_logs"
ADMIN_EMAIL="riccardo@riccardocosentino.com"
HOSTNAME=$(hostname)
MAX_CHECKPOINTS=5  # Keep last N checkpoints per game (reduced from 10)
LOG_FILE="/var/log/atari/cleanup.log"

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

# Function to archive checkpoints to GitHub backup repo
archive_checkpoints() {
    local game_name="$1"
    local checkpoints_to_archive="$2"
    
    mkdir -p "$ARCHIVE_DIR/$game_name"
    
    while IFS= read -r checkpoint; do
        if [ -f "$checkpoint" ]; then
            checkpoint_name=$(basename "$checkpoint")
            # Archive with timestamp to avoid conflicts
            archive_name="${checkpoint_name%.pt}_$(date +%Y%m%d_%H%M%S).pt"
            cp "$checkpoint" "$ARCHIVE_DIR/$game_name/$archive_name"
            log_msg "  📦 Archived $checkpoint_name to backup repo"
        fi
    done <<< "$checkpoints_to_archive"
}

# Function to archive logs to GitHub backup repo
archive_logs() {
    local logs_to_archive="$1"
    
    mkdir -p "$ARCHIVE_LOGS_DIR"
    local archive_date=$(date +%Y%m%d)
    
    while IFS= read -r log_file; do
        if [ -f "$log_file" ]; then
            log_name=$(basename "$log_file")
            archive_name="${log_name}_${archive_date}.gz"
            # Compress and archive
            gzip -c "$log_file" > "$ARCHIVE_LOGS_DIR/$archive_name" 2>/dev/null || \
                cp "$log_file" "$ARCHIVE_LOGS_DIR/${log_name}_${archive_date}" 2>/dev/null
            log_msg "  📦 Archived $log_name to backup repo"
        fi
    done <<< "$logs_to_archive"
}

# Function to commit and push archives to GitHub
push_archives() {
    cd "$BACKUP_DIR"
    if [ -d ".git" ]; then
        # Check if there are any changes
        if git diff --quiet && git diff --cached --quiet && [ -z "$(git ls-files --others --exclude-standard)" ]; then
            log_msg "  ℹ️ No archive changes to commit"
            return 0
        fi
        
        git add -A
        if git commit -m "Archive: Cleanup $(date '+%Y-%m-%d %H:%M:%S')" 2>&1; then
            if git push origin main 2>&1; then
                log_msg "  ✅ Archives pushed to GitHub backup repo"
                return 0
            else
                log_msg "  ⚠️ Archive commit OK but push failed"
                return 1
            fi
        else
            log_msg "  ⚠️ Archive commit failed (may be no changes)"
            return 0
        fi
    else
        log_msg "  ⚠️ $BACKUP_DIR is not a git repository - archives stored locally only"
        return 1
    fi
}

log_msg "🧹 Starting cleanup with archiving..."

# Track what was cleaned
CLEANED_CHECKPOINTS=0
ARCHIVED_CHECKPOINTS=0
CLEANED_PYCACHE=0
CLEANED_LOGS=0
ARCHIVED_LOGS=0
CLEANED_PIP_CACHE=0
CLEANED_JOURNAL=0
INITIAL_DISK=$(df / | tail -1 | awk '{print $5}')

# Ensure backup directory exists
mkdir -p "$BACKUP_DIR"
mkdir -p "$ARCHIVE_DIR"
mkdir -p "$ARCHIVE_LOGS_DIR"

# Clean old checkpoints (keep newest N per game, archive before deleting)
log_msg "Cleaning old model checkpoints..."
for game_dir in $APP_DIR/saved_models/*/; do
    if [ -d "$game_dir" ]; then
        game_name=$(basename "$game_dir")
        # Count checkpoints (exclude best_model.pt)
        checkpoint_count=$(ls -1 "$game_dir"checkpoint_*.pt 2>/dev/null | wc -l)
        
        if [ "$checkpoint_count" -gt "$MAX_CHECKPOINTS" ]; then
            # Get checkpoints to archive (oldest ones)
            checkpoints_to_archive=$(ls -1t "$game_dir"checkpoint_*.pt | tail -n +$((MAX_CHECKPOINTS + 1)))
            archived_count=$(echo "$checkpoints_to_archive" | grep -c . || echo 0)
            
            if [ "$archived_count" -gt 0 ]; then
                # Archive before deleting
                archive_checkpoints "$game_name" "$checkpoints_to_archive"
                ARCHIVED_CHECKPOINTS=$((ARCHIVED_CHECKPOINTS + archived_count))
                
                # Delete after archiving
                echo "$checkpoints_to_archive" | xargs rm -f
                CLEANED_CHECKPOINTS=$((CLEANED_CHECKPOINTS + archived_count))
                log_msg "  ✓ Cleaned $archived_count old checkpoints from $game_name (archived before deletion)"
            fi
        fi
    fi
done

# Clean Python cache
log_msg "Cleaning Python cache..."
PYCACHE_DIRS=$(find $APP_DIR -type d -name "__pycache__" 2>/dev/null | wc -l)
find $APP_DIR -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null
CLEANED_PYCACHE=$PYCACHE_DIRS
log_msg "  ✓ Removed $CLEANED_PYCACHE __pycache__ directories"

# Archive and clean old compressed logs (keep 7 days locally, archive older ones)
log_msg "Archiving and cleaning old compressed logs..."
OLD_LOGS=$(find /var/log/atari -name "*.gz" -mtime +7 2>/dev/null)
if [ -n "$OLD_LOGS" ]; then
    archive_logs "$OLD_LOGS"
    ARCHIVED_LOGS=$(echo "$OLD_LOGS" | wc -l)
    echo "$OLD_LOGS" | xargs rm -f
    CLEANED_LOGS=$ARCHIVED_LOGS
    log_msg "  ✓ Archived and removed $CLEANED_LOGS old compressed log files"
fi

# Archive and clean old uncompressed logs (keep 14 days locally, archive older ones)
OLD_UNCOMPRESSED=$(find /var/log/atari -name "*.log.[0-9]" -mtime +14 2>/dev/null)
if [ -n "$OLD_UNCOMPRESSED" ]; then
    archive_logs "$OLD_UNCOMPRESSED"
    archived_count=$(echo "$OLD_UNCOMPRESSED" | wc -l)
    ARCHIVED_LOGS=$((ARCHIVED_LOGS + archived_count))
    echo "$OLD_UNCOMPRESSED" | xargs rm -f
    CLEANED_LOGS=$((CLEANED_LOGS + archived_count))
    log_msg "  ✓ Archived and removed $archived_count old uncompressed log files"
fi

# Clean pip cache (safe to remove, will re-download if needed)
log_msg "Cleaning pip cache..."
PIP_CACHE_SIZE_BEFORE=$(du -sb /home/riccardo/.cache/pip 2>/dev/null | awk '{print $1}' || echo 0)
if [ -d "/home/riccardo/.cache/pip" ] && [ "$PIP_CACHE_SIZE_BEFORE" -gt 0 ]; then
    # Use pip cache purge if available, otherwise remove directory
    if command -v pip3 &> /dev/null; then
        pip3 cache purge 2>&1 | tee -a "$LOG_FILE" || true
    fi
    # Fallback: remove old cache files (older than 30 days)
    find /home/riccardo/.cache/pip -type f -mtime +30 -delete 2>/dev/null
    PIP_CACHE_SIZE_AFTER=$(du -sb /home/riccardo/.cache/pip 2>/dev/null | awk '{print $1}' || echo 0)
    CLEANED_PIP_CACHE=$((PIP_CACHE_SIZE_BEFORE - PIP_CACHE_SIZE_AFTER))
    if [ "$CLEANED_PIP_CACHE" -gt 0 ]; then
        CLEANED_PIP_CACHE_MB=$((CLEANED_PIP_CACHE / 1024 / 1024))
        log_msg "  ✓ Cleaned pip cache: ${CLEANED_PIP_CACHE_MB}MB freed"
    else
        log_msg "  ℹ️ Pip cache already clean"
    fi
else
    log_msg "  ℹ️ Pip cache directory not found or empty"
fi

# Clean systemd journal logs (keep last 7 days, limit to 500MB)
log_msg "Cleaning systemd journal logs..."
JOURNAL_SIZE_BEFORE=$(du -sb /var/log/journal 2>/dev/null | awk '{print $1}' || echo 0)
if [ "$JOURNAL_SIZE_BEFORE" -gt 0 ]; then
    # Use journalctl to vacuum logs (keep last 7 days, limit to 500MB)
    sudo journalctl --vacuum-time=7d --vacuum-size=500M 2>&1 | tee -a "$LOG_FILE" || true
    JOURNAL_SIZE_AFTER=$(du -sb /var/log/journal 2>/dev/null | awk '{print $1}' || echo 0)
    CLEANED_JOURNAL=$((JOURNAL_SIZE_BEFORE - JOURNAL_SIZE_AFTER))
    if [ "$CLEANED_JOURNAL" -gt 0 ]; then
        CLEANED_JOURNAL_MB=$((CLEANED_JOURNAL / 1024 / 1024))
        log_msg "  ✓ Cleaned journal logs: ${CLEANED_JOURNAL_MB}MB freed (kept last 7 days, max 500MB)"
    else
        log_msg "  ℹ️ Journal logs already within limits"
    fi
else
    log_msg "  ℹ️ Journal directory not found or empty"
fi

# Push archives to GitHub backup repo
if [ $ARCHIVED_CHECKPOINTS -gt 0 ] || [ $ARCHIVED_LOGS -gt 0 ]; then
    log_msg "Pushing archives to GitHub backup repository..."
    push_archives
fi

# Report disk usage
FINAL_DISK=$(df / | tail -1 | awk '{print $5}')
DISK_FREE=$(df -h / | tail -1 | awk '{print $4}')
DISK_FREED=$((100 - ${FINAL_DISK%\%} - (100 - ${INITIAL_DISK%\%}))) || DISK_FREED=0

log_msg "✅ Cleanup completed"
log_msg "Disk usage: $INITIAL_DISK -> $FINAL_DISK (Free: $DISK_FREE)"

# Send summary email
SUMMARY="Cleanup with archiving completed successfully.

Time: $(date)
Server: $HOSTNAME

Cleaned Items:
  • Model checkpoints removed: $CLEANED_CHECKPOINTS files
  • Model checkpoints archived: $ARCHIVED_CHECKPOINTS files
  • Python cache directories: $CLEANED_PYCACHE
  • Old log files removed: $CLEANED_LOGS files
  • Old log files archived: $ARCHIVED_LOGS files
  • Pip cache cleaned: $([ $CLEANED_PIP_CACHE -gt 0 ] && echo "$((CLEANED_PIP_CACHE / 1024 / 1024))MB" || echo "0MB")
  • Journal logs cleaned: $([ $CLEANED_JOURNAL -gt 0 ] && echo "$((CLEANED_JOURNAL / 1024 / 1024))MB" || echo "0MB")

Disk Status:
  • Before: $INITIAL_DISK
  • After: $FINAL_DISK
  • Available: $DISK_FREE
  • Total space freed: $([ $((CLEANED_PIP_CACHE + CLEANED_JOURNAL)) -gt 0 ] && echo "$(( (CLEANED_PIP_CACHE + CLEANED_JOURNAL) / 1024 / 1024 ))MB" || echo "0MB")

Archives Location:
  • Checkpoints: $ARCHIVE_DIR
  • Logs: $ARCHIVE_LOGS_DIR
  • Backup Repo: $BACKUP_DIR

Memory Usage: $(free -h | grep Mem | awk '{print $3 "/" $2}')

Next cleanup: In 6 hours
"

# Routine cleanup - no email (included in daily summary report)
log_msg "📧 Cleanup complete (summary in daily report)"
