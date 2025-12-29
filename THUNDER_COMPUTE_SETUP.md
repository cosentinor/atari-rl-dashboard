# ⚡ Thunder Compute Quick Setup

**📌 Save this - you'll need it when setting up new training instances!**

---

## 🚀 Setup New Thunder Compute Instance (5 Minutes)

### **Step 1: Create Instance** (Thunder Compute Dashboard)
- Go to https://www.thundercompute.com/
- Create **Production Mode** instance (NOT Prototyping!)
- GPU: **A100 80GB**
- Note: IP, Port, SSH Key Path

### **Step 2: Configure SSH** (Local Machine)
```bash
bash add_production_instance.sh
```
- Enter IP address, port, and SSH key path
- Choose an alias (e.g., `tnr-prod`, `tnr-2`, etc.)
- Script will test connection

### **Step 3: Setup Instance** (Copy & Run Script)
```bash
# Copy setup script to instance
scp setup_production.sh <your-alias>:~/

# SSH in and run setup
ssh <your-alias>
bash setup_production.sh
```
Wait 5-10 minutes for installation.

### **Step 4: Launch Training** (On Instance)
```bash
cd ~/atari-rl-dashboard
bash launch_production_training.sh
```
Trains 6 games in parallel. Runs in tmux - safe to disconnect.

### **Step 5: Monitor** (Local Machine)
```bash
python monitor_production.py --host <your-alias> --watch
```

---

## 📊 Expected Results

- **GPU Utilization**: 50-80%
- **Training Speed**: 2,000-3,000 FPS per game
- **Time to Complete**: 2-3 days for all 10 games
- **Episodes/Hour**: 6,000-24,000 per game

---

## 🔧 Common Commands

### Check Training Status
```bash
ssh <your-alias> "ps aux | grep python"
ssh <your-alias> "nvidia-smi"
```

### View Logs
```bash
ssh <your-alias> "tail -f ~/atari-rl-dashboard/training_mspacman.log"
```

### Download Models
```bash
# Download specific game
scp -r <your-alias>:~/atari-rl-dashboard/saved_models/MsPacman ./

# Download all models
scp -r <your-alias>:~/atari-rl-dashboard/saved_models ./saved_models_production
```

### Attach to Training Session
```bash
ssh <your-alias>
tmux attach -t atari-training
# Press Ctrl+B then D to detach without stopping
```

---

## ⚠️ Troubleshooting

### Training Stopped?
Thunder Compute instances may restart unexpectedly. Just re-run:
```bash
ssh <your-alias>
cd ~/atari-rl-dashboard
bash launch_production_training.sh
```
Training will resume from last checkpoint.

### Can't Connect?
```bash
# Test connection
ssh <your-alias>

# If fails, check SSH config
cat ~/.ssh/config | grep -A 10 "<your-alias>"
```

### Out of Disk Space?
```bash
ssh <your-alias> "df -h"
ssh <your-alias> "cd ~/atari-rl-dashboard && bash deployment/cleanup.sh"
```

---

## 📝 Your Active Instances

Update this section with your current instances:

| Alias | Status | Purpose | Created |
|-------|--------|---------|---------|
| tnr-1 | ✅ Active | Training | 2025-12-29 |
| | | | |
| | | | |

---

## 💡 Pro Tips

1. **Name your instances meaningfully**: `tnr-prod-1`, `tnr-pong-test`, etc.
2. **Monitor costs**: Production Mode is powerful but expensive
3. **Download models regularly**: Instances can be terminated
4. **Use tmux**: Always run training in tmux for disconnect safety
5. **Check GPU usage**: If under 50%, something's wrong

---

## 🆘 Need Help?

1. Check logs: `ssh <your-alias> "tail -100 ~/atari-rl-dashboard/training_*.log"`
2. See full setup guide: [PRODUCTION_SETUP.md](PRODUCTION_SETUP.md)
3. Test deployment: `python test_deployment.py`

---

**🎯 Remember:** The whole setup is 5 commands:
1. `bash add_production_instance.sh` (local)
2. `scp setup_production.sh <alias>:~/` (local)
3. `ssh <alias>` (connect)
4. `bash setup_production.sh` (on instance)
5. `bash launch_production_training.sh` (on instance)

Then monitor with: `python monitor_production.py --host <alias> --watch`

