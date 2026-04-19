# Atari RL Dashboard - Monitoring & Auto-Recovery Guide

## 🎯 Overview

Your Atari RL Dashboard now has comprehensive automated monitoring, auto-restart capabilities, and email notifications to keep you informed of system status and maintenance activities.

## 📧 Email Notifications

**Admin Email:** riccardo@riccardocosentino.com

You will receive emails for:
- ✅ Daily status reports (8:00 AM UTC)
- ⚠️ Service failures and automatic restarts
- 🚨 Critical issues requiring manual intervention
- 🧹 Daily cleanup summaries
- 💾 Daily backup summaries
- 📊 High resource usage warnings (disk >90%, memory >90%)

## 🔄 Auto-Restart Configuration

### Systemd Service Auto-Restart
The service is configured to automatically restart on any failure:

```
Restart=always
RestartSec=5
StartLimitBurst=5
StartLimitInterval=600
```

**What this means:**
- If the service crashes, it will restart after 5 seconds
- It will attempt up to 5 restarts within 10 minutes
- If it fails more than that, you'll receive a critical alert

### Resource Limits
To prevent runaway processes:
- **Memory Limit:** 3GB max
- **CPU Limit:** 200% (2 cores)

## 🛠️ Automated Maintenance Schedule

### Every 5 Minutes - Health Check
**Script:** `enhanced_health_check.sh`

Checks:
- ✓ Service is running
- ✓ HTTP endpoint responds with 200 OK
- ✓ Disk usage (<90%)
- ✓ Memory usage (<90%)

Actions:
- Auto-restarts service if down
- Sends alerts if problems detected
- Logs all checks to `/var/log/atari/health.log`

### Daily at 4:00 AM UTC - Cleanup
**Script:** `enhanced_cleanup.sh`

Tasks:
- Removes old model checkpoints (keeps latest 10 per game)
- Cleans Python `__pycache__` directories
- Removes old compressed logs (>7 days)
- Reports disk space freed

Email: Summary of cleanup activities

### Daily at 5:00 AM UTC - Backup
**Script:** `enhanced_backup.sh`

Tasks:
- Copies all `best_model.pt` files to backup directory
- Backs up model registry
- Commits to git repository (if configured)
- Reports backup status and size

Email: Summary of backup with model counts

### Daily at 8:00 AM UTC - Status Report
**Script:** `daily_status_report.sh`

Provides comprehensive daily summary:
- System uptime and load
- Service status and recent restarts
- Resource usage (disk, memory, CPU)
- Active training sessions
- Recent issues in last 24 hours
- Quick access links

Email: Full status report

## 📊 Monitoring Locations

### Log Files
All monitoring logs are in `/var/log/atari/`:

```bash
# View logs
ssh atari "tail -f /var/log/atari/health.log"    # Health checks
ssh atari "tail -f /var/log/atari/cleanup.log"   # Cleanup activities
ssh atari "tail -f /var/log/atari/backup.log"    # Backup status
ssh atari "tail -f /var/log/atari/status.log"    # Status reports
ssh atari "tail -f /var/log/atari/app.log"       # Application logs
ssh atari "tail -f /var/log/atari/error.log"     # Error logs
```

### Service Status
Check service status anytime:

```bash
ssh atari "sudo systemctl status atari"
```

### Recent Activity
View recent service events:

```bash
ssh atari "sudo journalctl -u atari -n 50"
```

## 🔔 Alert Severity Levels

### ✅ Info (Green) - No Action Needed
- Daily status reports
- Successful maintenance activities
- Health checks passing

### ⚠️ Warning (Yellow) - Monitoring Required
- Service auto-restarted successfully
- HTTP endpoint recovered after restart
- High disk usage (>90%)
- High memory usage (>90%)
- Backup push to git failed

### 🚨 Critical (Red) - Immediate Action Required
- Service restart failed
- HTTP endpoint not responding after restart
- Multiple restarts in short period
- Service failure notification

## 🔧 Manual Commands

### Check System Health
```bash
# Quick status check
ssh atari "bash /home/riccardo/atari-rl-dashboard/deployment/daily_status_report.sh"

# Detailed service status
ssh atari "sudo systemctl status atari --no-pager -l"

# Check resource usage
ssh atari "df -h / && free -h && uptime"
```

### Force Maintenance Activities
```bash
# Run cleanup manually
ssh atari "bash /home/riccardo/atari-rl-dashboard/deployment/enhanced_cleanup.sh"

# Run backup manually
ssh atari "bash /home/riccardo/atari-rl-dashboard/deployment/enhanced_backup.sh"

# Run health check manually
ssh atari "bash /home/riccardo/atari-rl-dashboard/deployment/enhanced_health_check.sh"
```

### Restart Service Manually
```bash
# Restart the service
ssh atari "sudo systemctl restart atari"

# Check if it started successfully
ssh atari "sudo systemctl status atari"

# View recent logs
ssh atari "tail -50 /var/log/atari/error.log"
```

### View Cron Jobs
```bash
# List all scheduled tasks
ssh atari "crontab -l"
```

## 🧪 Testing Email Notifications

Send a test email:

```bash
ssh atari 'echo "Test email from Atari RL Dashboard" | mail -s "Test Email" riccardo@riccardocosentino.com'
```

Test the health check (will send email if issues found):

```bash
ssh atari "bash /home/riccardo/atari-rl-dashboard/deployment/enhanced_health_check.sh"
```

## 📈 Performance Monitoring

### Current Resource Usage
- **Disk:** 80% used (29GB / 38GB) - Cleanup helps maintain space
- **Memory:** 788MB used / 3.7GB total
- **Service Memory:** ~281MB with 3GB limit

### Disk Space Management
The cleanup script runs daily to prevent disk from filling up:
- Removes old model checkpoints
- Cleans temporary Python files
- Archives old logs

If disk reaches >90%, you'll receive an immediate alert.

## 🔐 Security Features

The systemd service includes:
- **NoNewPrivileges:** Prevents privilege escalation
- **PrivateTmp:** Isolated temporary directory
- **Resource Limits:** Prevents runaway processes
- **Non-root User:** Runs as `riccardo` user

## 📝 Troubleshooting

### Not Receiving Emails?

1. Check postfix is running:
   ```bash
   ssh atari "sudo systemctl status postfix"
   ```

2. Check mail logs:
   ```bash
   ssh atari "sudo tail -50 /var/log/mail.log"
   ```

3. Test email manually:
   ```bash
   ssh atari 'echo "Test" | mail -s "Test" riccardo@riccardocosentino.com'
   ```

### Service Keeps Restarting?

1. Check error logs:
   ```bash
   ssh atari "tail -100 /var/log/atari/error.log"
   ```

2. Check for Python errors:
   ```bash
   ssh atari "sudo journalctl -u atari -n 100"
   ```

3. Check resource usage:
   ```bash
   ssh atari "free -h && df -h"
   ```

### Disk Full?

1. Run cleanup manually:
   ```bash
   ssh atari "bash /home/riccardo/atari-rl-dashboard/deployment/enhanced_cleanup.sh"
   ```

2. Check large files:
   ```bash
   ssh atari "du -sh /home/riccardo/atari-rl-dashboard/* | sort -hr | head -10"
   ```

3. Clear old logs manually:
   ```bash
   ssh atari "sudo find /var/log/atari -name '*.gz' -mtime +3 -delete"
   ```

## 🎛️ Configuration Files

### Main Service
`/etc/systemd/system/atari.service` - Main service configuration

### Failure Notifications
`/etc/systemd/system/atari-failure-notify@.service` - Email on failures

### Scripts
All monitoring scripts are in:
`/home/riccardo/atari-rl-dashboard/deployment/`

## 📞 Support

If you receive a critical alert:

1. **Check the email** - It includes the issue and suggested commands
2. **Review logs** - Use commands from the email
3. **Try manual restart** - Often resolves temporary issues
4. **Check resources** - Disk/memory might be full

## 🔄 Update Monitoring Scripts

If you need to update monitoring scripts:

```bash
# 1. Update scripts locally in deployment/ directory
# 2. Upload to VPS
scp deployment/enhanced_*.sh atari:/home/riccardo/atari-rl-dashboard/deployment/

# 3. Make executable
ssh atari "chmod +x /home/riccardo/atari-rl-dashboard/deployment/*.sh"

# 4. Test manually
ssh atari "bash /home/riccardo/atari-rl-dashboard/deployment/enhanced_health_check.sh"
```

## 📊 Expected Email Schedule

- **Every 5 minutes:** Health checks (only on issues)
- **Daily 4:00 AM UTC:** Cleanup summary
- **Daily 5:00 AM UTC:** Backup summary
- **Daily 8:00 AM UTC:** Full status report
- **On failures:** Immediate critical alerts

**Typical email volume:** 3-4 per day (status updates) + alerts as needed

---

## ✅ Summary

Your Atari RL Dashboard is now fully monitored with:

✓ Automatic service restart on failures
✓ Proactive health monitoring every 5 minutes
✓ Daily automated maintenance (cleanup + backup)
✓ Email notifications for all important events
✓ Resource usage monitoring
✓ Comprehensive daily status reports

**You'll know immediately if something goes wrong, and most issues will auto-recover!**
