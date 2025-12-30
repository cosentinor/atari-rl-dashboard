#!/bin/bash
# Setup continuous monitoring for reboot tracking (runs ON the server)

echo "========================================"
echo "SETTING UP STABILITY MONITORING"
echo "========================================"

cd /home/ubuntu

# Make script executable
chmod +x monitor_stability_local.sh

# Kill old monitoring loop if exists
pkill -f stability_monitor_loop.sh 2>/dev/null

# Create monitoring loop script
cat > stability_monitor_loop.sh << 'MONITORLOOP'
#!/bin/bash
# Continuous stability monitoring - checks every 10 minutes

while true; do
    bash /home/ubuntu/monitor_stability_local.sh
    sleep 600  # Check every 10 minutes
done
MONITORLOOP

chmod +x stability_monitor_loop.sh

# Start monitoring loop in background
nohup bash /home/ubuntu/stability_monitor_loop.sh >> /home/ubuntu/monitor_loop.log 2>&1 &
MONITOR_PID=$!

# Also add to watchdog for persistence
echo "# Stability monitoring" >> /home/ubuntu/.profile
echo "if ! pgrep -f stability_monitor_loop.sh > /dev/null; then" >> /home/ubuntu/.profile
echo "    nohup bash /home/ubuntu/stability_monitor_loop.sh >> /home/ubuntu/monitor_loop.log 2>&1 &" >> /home/ubuntu/.profile
echo "fi" >> /home/ubuntu/.profile

echo ""
echo "========================================"
echo "✅ STABILITY MONITORING ACTIVE!"
echo "========================================"
echo ""
echo "Monitoring PID: $MONITOR_PID"
echo "Check frequency: Every 10 minutes"
echo "Will auto-restart on reboot: YES"
echo ""
echo "View logs:"
echo "  tail -f /home/ubuntu/stability_monitor.log"
echo "  tail -f /home/ubuntu/reboot_alerts.log"
echo ""
echo "Check status anytime:"
echo "  tail -20 /home/ubuntu/stability_monitor.log"
echo ""
echo "See all reboots:"
echo "  cat /home/ubuntu/reboot_alerts.log"
echo ""
echo "========================================"

# Initial check
echo ""
echo "Running initial stability check..."
bash /home/ubuntu/monitor_stability_local.sh
