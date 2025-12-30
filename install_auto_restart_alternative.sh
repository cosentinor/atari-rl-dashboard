#!/bin/bash
# Alternative auto-restart setup for systems without systemd/cron

echo "=============================================="
echo "ALTERNATIVE AUTO-RESTART SETUP"
echo "=============================================="

cd /home/ubuntu/atari-rl-dashboard

# Make scripts executable
chmod +x auto_restart_training.sh
chmod +x watchdog_training.sh

# Create rc.local style boot script
echo "Creating boot script..."
cat > /home/ubuntu/start_training_on_boot.sh << 'BOOTSCRIPT'
#!/bin/bash
# Wait for system to fully boot
sleep 60

# Run auto-restart script
/bin/bash /home/ubuntu/atari-rl-dashboard/auto_restart_training.sh >> /home/ubuntu/boot_training.log 2>&1
BOOTSCRIPT

chmod +x /home/ubuntu/start_training_on_boot.sh

# Add to .profile for automatic start (will run on login/boot)
if ! grep -q "start_training_on_boot.sh" /home/ubuntu/.profile 2>/dev/null; then
    echo "" >> /home/ubuntu/.profile
    echo "# Auto-start training on boot" >> /home/ubuntu/.profile
    echo "if [ -f /home/ubuntu/start_training_on_boot.sh ]; then" >> /home/ubuntu/.profile
    echo "    nohup /home/ubuntu/start_training_on_boot.sh &" >> /home/ubuntu/.profile
    echo "fi" >> /home/ubuntu/.profile
fi

# Create simple watchdog loop script
cat > /home/ubuntu/watchdog_loop.sh << 'WATCHDOGLOOP'
#!/bin/bash
# Continuous watchdog - runs forever

while true; do
    /bin/bash /home/ubuntu/atari-rl-dashboard/watchdog_training.sh
    sleep 300  # Check every 5 minutes
done
WATCHDOGLOOP

chmod +x /home/ubuntu/watchdog_loop.sh

# Start watchdog in background
nohup /home/ubuntu/watchdog_loop.sh >> /home/ubuntu/watchdog_loop.log 2>&1 &
WATCHDOG_PID=$!

echo ""
echo "=============================================="
echo "✅ ALTERNATIVE AUTO-RESTART INSTALLED!"
echo "=============================================="
echo ""
echo "What was configured:"
echo "  1. Boot script: Added to .profile (runs on system start)"
echo "  2. Watchdog loop: Running in background (PID: $WATCHDOG_PID)"
echo ""
echo "How it works:"
echo "  - When server reboots, .profile runs and starts training"
echo "  - Watchdog checks every 5 minutes and restarts if needed"
echo ""
echo "Monitor:"
echo "  Boot log:      tail -f /home/ubuntu/boot_training.log"
echo "  Watchdog log:  tail -f /home/ubuntu/watchdog_loop.log"
echo "  Training logs: tail -f ~/atari-rl-dashboard/training_*.log"
echo ""
echo "Manual control:"
echo "  Start:   bash ~/atari-rl-dashboard/auto_restart_training.sh"
echo "  Stop:    pkill -f 'python.*train'"
echo "  Check:   ps aux | grep train | grep -v grep"
echo ""
echo "=============================================="
