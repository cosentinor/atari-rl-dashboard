#!/bin/bash
# Setup continuous monitoring for reboot tracking

echo "========================================"
echo "SETTING UP STABILITY MONITORING"
echo "========================================"

cd ~

# Make script executable
chmod +x monitor_stability.sh

# Create monitoring loop script
cat > stability_monitor_loop.sh << 'MONITORLOOP'
#!/bin/bash
# Continuous stability monitoring - checks every 10 minutes

while true; do
    bash ~/monitor_stability.sh
    sleep 600  # Check every 10 minutes
done
MONITORLOOP

chmod +x stability_monitor_loop.sh

# Start monitoring loop in background
nohup bash ~/stability_monitor_loop.sh >> monitor_loop.log 2>&1 &
MONITOR_PID=$!

echo ""
echo "========================================"
echo "✅ STABILITY MONITORING ACTIVE!"
echo "========================================"
echo ""
echo "Monitoring PID: $MONITOR_PID"
echo "Check frequency: Every 10 minutes"
echo ""
echo "Logs:"
echo "  Detailed log:    tail -f ~/stability_monitor.log"
echo "  Reboot alerts:   tail -f ~/reboot_alerts.log"
echo "  Monitor loop:    tail -f ~/monitor_loop.log"
echo ""
echo "To check status anytime:"
echo "  cat ~/stability_monitor.log | tail -20"
echo ""
echo "To see reboot history:"
echo "  cat ~/reboot_alerts.log"
echo ""
echo "========================================"

# Initial check
echo "Running initial stability check..."
bash ~/monitor_stability.sh
