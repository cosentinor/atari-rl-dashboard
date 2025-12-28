#!/bin/bash
# Health check script - runs every 5 minutes via cron
SERVICE="atari"
URL="http://localhost:5001"
EMAIL="riccardo@example.com" # Updated to your user context

# Check if service is running
if ! systemctl is-active --quiet $SERVICE; then
    echo "Service $SERVICE is down! Restarting..."
    sudo systemctl restart $SERVICE
    echo "Atari app was down and restarted at $(date)" | mail -s "⚠️ Atari App Restarted" $EMAIL
fi

# Check if web endpoint responds
HTTP_CODE=$(curl -s -o /dev/null -w "%{http_code}" $URL)
if [ "$HTTP_CODE" != "200" ]; then
    echo "HTTP check failed (code: $HTTP_CODE). Restarting..."
    sudo systemctl restart $SERVICE
    echo "Atari app HTTP check failed at $(date). Restarted." | mail -s "⚠️ Atari App HTTP Error" $EMAIL
fi

