#!/bin/bash
# Quick dashboard to check training status from your local machine

echo "========================================"
echo "🎮 ATARI TRAINING DASHBOARD"
echo "========================================"
echo ""

echo "📊 System Status:"
ssh tnr-0 "
echo '  Uptime: '\$(uptime -p)
echo '  GPU: '\$(nvidia-smi --query-gpu=utilization.gpu,memory.used,temperature.gpu --format=csv,noheader,nounits)
echo '  Training Processes: '\$(ps aux | grep python | grep train | grep -v grep | wc -l)
"

echo ""
echo "🔄 Recent Reboots:"
ssh tnr-0 "tail -5 /home/ubuntu/reboot_alerts.log 2>/dev/null || echo '  No reboots detected yet'"

echo ""
echo "📈 Training Progress (Last 3 updates each):"
echo ""
echo "  🟢 MsPacman (Optimized):"
ssh tnr-0 "tail -100 ~/atari-rl-dashboard/training_mspacman.log | grep 'Ep ' | tail -3"

echo ""
echo "  🔵 Pong (Backup):"
ssh tnr-0 "tail -100 ~/atari-rl-dashboard/pong_train.log | grep 'Ep ' | tail -3"

echo ""
echo "========================================"
echo "📝 Full Logs:"
echo "  Stability: ssh tnr-0 'tail -f /home/ubuntu/stability_monitor.log'"
echo "  Training:  ssh tnr-0 'tail -f ~/atari-rl-dashboard/training_*.log'"
echo "  Reboots:   ssh tnr-0 'cat /home/ubuntu/reboot_alerts.log'"
echo "========================================"
