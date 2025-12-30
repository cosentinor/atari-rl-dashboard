#!/bin/bash
# Setup script for auto-restart system
# Run this ONCE on the Thunder Compute instance

echo "=============================================="
echo "SETTING UP AUTO-RESTART SYSTEM"
echo "=============================================="

# Make scripts executable
chmod +x /home/ubuntu/atari-rl-dashboard/auto_restart_training.sh
chmod +x /home/ubuntu/atari-rl-dashboard/watchdog_training.sh

# Copy systemd service
echo "Installing systemd service..."
sudo cp /home/ubuntu/atari-rl-dashboard/atari-training.service /etc/systemd/system/

# Reload systemd
echo "Reloading systemd..."
sudo systemctl daemon-reload

# Enable service to start on boot
echo "Enabling auto-start on boot..."
sudo systemctl enable atari-training.service

# Setup cron job for watchdog (every 5 minutes)
echo "Setting up watchdog cron job..."
(crontab -l 2>/dev/null | grep -v watchdog_training; echo "*/5 * * * * /bin/bash /home/ubuntu/atari-rl-dashboard/watchdog_training.sh") | crontab -

echo ""
echo "=============================================="
echo "✅ AUTO-RESTART SYSTEM INSTALLED!"
echo "=============================================="
echo ""
echo "What was configured:"
echo "  1. Systemd service: Restarts training on server boot"
echo "  2. Watchdog cron: Checks every 5 minutes and restarts if crashed"
echo ""
echo "Commands:"
echo "  Check service status:  sudo systemctl status atari-training"
echo "  Start manually:        sudo systemctl start atari-training"
echo "  Stop:                  sudo systemctl stop atari-training"
echo "  View watchdog log:     tail -f /home/ubuntu/atari-rl-dashboard/watchdog.log"
echo "  View cron jobs:        crontab -l"
echo ""
echo "To test the auto-restart:"
echo "  1. Reboot the server: sudo reboot"
echo "  2. Wait 2-3 minutes after reboot"
echo "  3. Check: ps aux | grep train"
echo ""
echo "=============================================="
