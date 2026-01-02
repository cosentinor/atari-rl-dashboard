#!/bin/bash
# Cleanup script - keeps disk usage under control
APP_DIR="/home/ubuntu/atari-rl-dashboard"
MAX_CHECKPOINTS=10  # Keep last N checkpoints per game

# Clean old checkpoints (keep newest 10 per game)
for game_dir in $APP_DIR/saved_models/*/; do
    if [ -d "$game_dir" ]; then
        # Count checkpoints (exclude best_model.pt)
        checkpoint_count=$(ls -1 "$game_dir"checkpoint_*.pt 2>/dev/null | wc -l)
        
        if [ "$checkpoint_count" -gt "$MAX_CHECKPOINTS" ]; then
            # Delete oldest checkpoints
            ls -1t "$game_dir"checkpoint_*.pt | tail -n +$((MAX_CHECKPOINTS + 1)) | xargs rm -f
            echo "$(date): Cleaned old checkpoints in $game_dir"
        fi
    fi
done

# Clean Python cache
find $APP_DIR -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null

# Clean old logs (older than 30 days)
find /var/log/atari -name "*.log" -mtime +30 -delete 2>/dev/null

# Report disk usage
echo "$(date): Disk usage: $(df -h / | tail -1 | awk '{print $5}')"
