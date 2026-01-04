#!/bin/bash
# Health check script - runs every 5 minutes via cron
SERVICE="atari"
URL="http://localhost:5001"
NGINX_SERVICE="nginx"
NGINX_URL="http://localhost"
NGINX_HOST_HEADER="atari.riccardocosentino.com"
EMAIL="riccardo@riccardocosentino.com"

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

# Check if nginx is running
if ! systemctl is-active --quiet $NGINX_SERVICE; then
    echo "Service $NGINX_SERVICE is down! Restarting..."
    sudo systemctl restart $NGINX_SERVICE
    echo "Nginx was down and restarted at $(date)" | mail -s "⚠️ Nginx Restarted" $EMAIL
fi

# Check if nginx responds (allow redirects)
NGINX_HTTP_CODE=$(curl -s -o /dev/null -w "%{http_code}" -H "Host: $NGINX_HOST_HEADER" $NGINX_URL)
if [ "$NGINX_HTTP_CODE" -lt 200 ] || [ "$NGINX_HTTP_CODE" -ge 400 ]; then
    echo "Nginx HTTP check failed (code: $NGINX_HTTP_CODE). Restarting..."
    sudo systemctl restart $NGINX_SERVICE
    echo "Nginx HTTP check failed at $(date). Restarted." | mail -s "⚠️ Nginx HTTP Error" $EMAIL
fi
