# Monitoring Quick Reference Card

## 🚨 Emergency Commands

```bash
# Check if service is running
ssh atari "sudo systemctl status atari"

# View recent errors
ssh atari "tail -50 /var/log/atari/error.log"

# Restart service manually
ssh atari "sudo systemctl restart atari"

# Check recent alerts
ssh atari "tail -20 /var/log/atari/health.log"
```

## 📧 Email Alerts You'll Receive

| Time | Type | What It Means |
|------|------|---------------|
| **Immediate** | 🚨 CRITICAL | Service failed and restart didn't work - CHECK NOW |
| **Immediate** | ⚠️ WARNING | Service auto-restarted successfully - monitor it |
| **Immediate** | ⚠️ WARNING | Disk/memory >90% - may need cleanup |
| **4:00 AM UTC** | ✅ INFO | Daily cleanup completed |
| **5:00 AM UTC** | ✅ INFO | Daily backup completed |
| **8:00 AM UTC** | 📊 INFO | Daily status report |

## 🔄 What Auto-Restarts

| Scenario | Auto-Action | Your Action |
|----------|-------------|-------------|
| Service crashes | Restarts in 5 seconds | None (you'll get email) |
| HTTP not responding | Service restart | None (you'll get email) |
| Memory limit exceeded | Service restart | None initially, check if it keeps happening |
| Multiple failures (>5 in 10min) | Stops trying | **MANUAL FIX NEEDED** |

## 📊 Monitoring Schedule

```
┌─────────────────────────────────────────┐
│ EVERY 5 MINUTES                         │
│ • Check service is running              │
│ • Check HTTP responds                   │
│ • Check disk < 90%                      │
│ • Check memory < 90%                    │
│ • Auto-restart if issues                │
└─────────────────────────────────────────┘

┌─────────────────────────────────────────┐
│ DAILY @ 4:00 AM UTC                     │
│ • Remove old model checkpoints          │
│ • Clean Python cache                    │
│ • Remove old logs (>7 days)             │
│ • Email cleanup summary                 │
└─────────────────────────────────────────┘

┌─────────────────────────────────────────┐
│ DAILY @ 5:00 AM UTC                     │
│ • Backup all best models                │
│ • Commit to git (if configured)         │
│ • Email backup summary                  │
└─────────────────────────────────────────┘

┌─────────────────────────────────────────┐
│ DAILY @ 8:00 AM UTC                     │
│ • Full system status report             │
│ • Resource usage summary                │
│ • Recent issues report                  │
│ • Email comprehensive report            │
└─────────────────────────────────────────┘
```

## 🎯 Dashboard Access

- **URL:** http://atari.riccardocosentino.com:5001
- **IP:** http://46.224.26.78:5001
- **SSH:** `ssh atari`

## 📁 Important Paths

```
/home/riccardo/atari-rl-dashboard/          # Application
/var/log/atari/                             # All logs
/home/riccardo/atari-models-backup/         # Model backups
/etc/systemd/system/atari.service           # Service config
```

## 🔧 Common Tasks

### View Logs
```bash
ssh atari "tail -f /var/log/atari/health.log"   # Health checks
ssh atari "tail -f /var/log/atari/error.log"    # Errors
```

### Manual Maintenance
```bash
# Force cleanup now
ssh atari "bash /home/riccardo/atari-rl-dashboard/deployment/enhanced_cleanup.sh"

# Force backup now
ssh atari "bash /home/riccardo/atari-rl-dashboard/deployment/enhanced_backup.sh"

# Get status report now
ssh atari "bash /home/riccardo/atari-rl-dashboard/deployment/daily_status_report.sh"
```

### Check Resources
```bash
ssh atari "df -h / && free -h"              # Disk and memory
ssh atari "systemctl status atari"          # Service status
```

## ⚡ When You Get An Alert

### ⚠️ Warning Alert
1. Read the email - it explains what happened
2. Usually auto-recovered - no action needed
3. If you get multiple in short time, investigate

### 🚨 Critical Alert
1. **ACT IMMEDIATELY** - service is down
2. Follow commands in the email
3. Check logs: `ssh atari "tail -100 /var/log/atari/error.log"`
4. Try manual restart: `ssh atari "sudo systemctl restart atari"`
5. If still failing, check resources: `ssh atari "df -h && free -h"`

## 💡 Pro Tips

- **Check email in the morning** - Daily report shows what happened overnight
- **Disk at 80%** - Cleanup script keeps it under control
- **No alerts is good!** - System is self-healing
- **Multiple restarts** - Check for memory leaks or config issues
- **Update code** - Service auto-restarts, but test first!

## 🔄 Service Limits

```
Memory: 3GB max (currently using ~281MB)
CPU: 200% max (2 cores)
Restart: Up to 5 times in 10 minutes
```

---

**Remember:** Most issues auto-recover. Only act on CRITICAL alerts! ✅
