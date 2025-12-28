#!/bin/bash
# Backup best models to a separate GitHub repo
# Note: This assumes you have a repo named 'atari-models-backup' in the same account
cd /home/riccardo/atari-rl-dashboard

BACKUP_DIR="/home/riccardo/atari-models-backup"
mkdir -p "$BACKUP_DIR"

# Copy best models
for game_dir in saved_models/*/; do
    game_name=$(basename "$game_dir")
    mkdir -p "$BACKUP_DIR/$game_name"
    
    if [ -f "$game_dir/best_model.pt" ]; then
        cp "$game_dir/best_model.pt" "$BACKUP_DIR/$game_name/"
    fi
done

# Copy registry
cp saved_models/model_registry.json "$BACKUP_DIR/" 2>/dev/null

# Commit and push
cd $BACKUP_DIR
if [ -d ".git" ]; then
    git add -A
    git commit -m "Backup: $(date '+%Y-%m-%d %H:%M')" || true
    git push origin main
    echo "$(date): Backup completed"
else
    echo "$(date): Backup failed - $BACKUP_DIR is not a git repository"
fi

