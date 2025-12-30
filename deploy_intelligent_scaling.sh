#!/bin/bash
# Deploy Intelligent Auto-Scaling System to Thunder Compute

echo "========================================"
echo "DEPLOYING INTELLIGENT AUTO-SCALING SYSTEM"
echo "========================================"

# Upload all scripts
echo "Uploading scripts to tnr-0..."
scp state_manager.sh tnr-0:/home/ubuntu/
scp intelligent_scaler.sh tnr-0:/home/ubuntu/
scp scaling_monitor.sh tnr-0:/home/ubuntu/
scp auto_restart_training.sh tnr-0:~/atari-rl-dashboard/
scp light_training.sh tnr-0:~/atari-rl-dashboard/
scp medium_training.sh tnr-0:~/atari-rl-dashboard/
scp stop_training.sh tnr-0:~/atari-rl-dashboard/

echo ""
echo "Making scripts executable..."
ssh tnr-0 "chmod +x /home/ubuntu/state_manager.sh"
ssh tnr-0 "chmod +x /home/ubuntu/intelligent_scaler.sh"
ssh tnr-0 "chmod +x /home/ubuntu/scaling_monitor.sh"
ssh tnr-0 "chmod +x ~/atari-rl-dashboard/auto_restart_training.sh"
ssh tnr-0 "chmod +x ~/atari-rl-dashboard/light_training.sh"
ssh tnr-0 "chmod +x ~/atari-rl-dashboard/medium_training.sh"
ssh tnr-0 "chmod +x ~/atari-rl-dashboard/stop_training.sh"

echo ""
echo "Initializing state..."
ssh tnr-0 "bash /home/ubuntu/state_manager.sh init"
ssh tnr-0 "bash /home/ubuntu/state_manager.sh set_mode light"

echo ""
echo "Setting up monitoring loop..."
ssh tnr-0 "pkill -f scaling_monitor_loop 2>/dev/null || true"

ssh tnr-0 "cat > /home/ubuntu/scaling_monitor_loop.sh << 'MONLOOP'
#!/bin/bash
# Continuous scaling monitor

while true; do
    bash /home/ubuntu/scaling_monitor.sh
    sleep 300  # Check every 5 minutes
done
MONLOOP"

ssh tnr-0 "chmod +x /home/ubuntu/scaling_monitor_loop.sh"
ssh tnr-0 "nohup bash /home/ubuntu/scaling_monitor_loop.sh >> /home/ubuntu/scaling_loop.log 2>&1 &"

echo ""
echo "Updating .profile for persistence..."
ssh tnr-0 "grep -q 'scaling_monitor_loop' /home/ubuntu/.profile || echo '
# Auto-start scaling monitor
if ! pgrep -f scaling_monitor_loop.sh > /dev/null 2>&1; then
    nohup bash /home/ubuntu/scaling_monitor_loop.sh >> /home/ubuntu/scaling_loop.log 2>&1 &
fi' >> /home/ubuntu/.profile"

echo ""
echo "========================================"
echo "✅ DEPLOYMENT COMPLETE!"
echo "========================================"
echo ""
echo "Installed components:"
echo "  ✅ State Manager (state_manager.sh)"
echo "  ✅ Intelligent Scaler (intelligent_scaler.sh)"
echo "  ✅ Scaling Monitor (scaling_monitor.sh)"
echo "  ✅ Light Training Mode (light_training.sh)"
echo "  ✅ Medium Training Mode (medium_training.sh)"
echo "  ✅ Stop Script (stop_training.sh)"
echo "  ✅ Updated Auto-Restart (auto_restart_training.sh)"
echo ""
echo "System is now running in LIGHT mode."
echo "Will automatically scale to MEDIUM after 60 minutes stable."
echo "Will fallback to LIGHT if >3 reboots in 1 hour."
echo ""
echo "Monitor with:"
echo "  ssh tnr-0 'cat /home/ubuntu/scaling_state.json | python3 -m json.tool'"
echo "  ssh tnr-0 'tail -f /home/ubuntu/scaling_decisions.log'"
echo "  ssh tnr-0 'tail -f /home/ubuntu/scaling_monitor.log'"
echo ""
echo "========================================"
