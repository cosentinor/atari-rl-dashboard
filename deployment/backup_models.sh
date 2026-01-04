#!/bin/bash
# Backup best models to a separate GitHub repo
# Note: This assumes you have a repo named 'atari-models-backup' in the same account
APP_DIR="/home/riccardo/atari-rl-dashboard"
BACKUP_DIR="/home/riccardo/atari-models-backup"

mkdir -p "$BACKUP_DIR"
if [ ! -d "$BACKUP_DIR/.git" ]; then
    git -C "$BACKUP_DIR" init >/dev/null 2>&1 || true
fi

cd "$APP_DIR"

# Copy best models
shopt -s nullglob
for game_dir in saved_models/*/; do
    game_name=$(basename "$game_dir")
    mkdir -p "$BACKUP_DIR/$game_name"
    
    if [ -f "$game_dir/best_model.pt" ]; then
        cp "$game_dir/best_model.pt" "$BACKUP_DIR/$game_name/"
    fi
done
shopt -u nullglob

# Copy registry
cp saved_models/model_registry.json "$BACKUP_DIR/" 2>/dev/null || true

# Commit and push
if git -C "$BACKUP_DIR" rev-parse --is-inside-work-tree >/dev/null 2>&1; then
    if git -C "$BACKUP_DIR" remote get-url origin >/dev/null 2>&1 || git -C "$BACKUP_DIR" remote get-url origin-ssh >/dev/null 2>&1; then
        if ! git -C "$BACKUP_DIR" config user.email >/dev/null; then
            git -C "$BACKUP_DIR" config user.email "riccardo@localhost"
        fi
        if ! git -C "$BACKUP_DIR" config user.name >/dev/null; then
            git -C "$BACKUP_DIR" config user.name "riccardo"
        fi
        git -C "$BACKUP_DIR" add -A
        git -C "$BACKUP_DIR" commit -m "Backup: $(date '+%Y-%m-%d %H:%M')" || true
        if git -C "$BACKUP_DIR" remote get-url origin-ssh >/dev/null 2>&1; then
            if git -C "$BACKUP_DIR" push origin-ssh main; then
                echo "$(date): Backup completed (origin-ssh)"
                exit 0
            else
                echo "$(date): Backup push failed - origin-ssh"
            fi
        fi
        if git -C "$BACKUP_DIR" remote get-url origin >/dev/null 2>&1; then
            if git -C "$BACKUP_DIR" push origin main; then
                echo "$(date): Backup completed (origin)"
            else
                echo "$(date): Backup failed - git push failed"
            fi
        else
            echo "$(date): Backup skipped - origin remote not configured"
        fi
    else
        echo "$(date): Backup skipped - origin remote not configured"
    fi
else
    echo "$(date): Backup failed - $BACKUP_DIR is not a git repository"
fi
