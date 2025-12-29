# Thunder Compute Production Mode Setup Guide

Complete guide for setting up and running high-speed parallel training on Thunder Compute Production Mode.

## 🎯 Overview

This setup enables training all 10 Atari games in **2-3 days** (vs 10-14 days on Prototyping Mode) by:
- Using Thunder Compute **Production Mode** (no GPU restrictions)
- Running **6 games simultaneously** with EnvPool
- Achieving **90-95% GPU utilization** on A100
- Training at **6,000-24,000 episodes/hour** per game

## 📊 Performance Comparison

| Metric | Prototyping Mode | Production Mode |
|--------|------------------|-----------------|
| Training Script | train.py | train_envpool.py |
| Speed | 300-700 eps/hr | 6,000-24,000 eps/hr |
| Parallel Games | 3 (unstable) | 6 (stable) |
| GPU Utilization | 6-82% | 90-95% |
| Total Duration | 10-14 days | 2-3 days |
| Cost | $ × 14 days | $$$ × 2.5 days |
| **Total Cost** | 💰💰💰💰 | 💰💰💰 |

**Expected savings: 20-30%** despite higher hourly rate!

## 🚀 Quick Start

### Step 1: Create Production Instance (Manual)

1. Go to [Thunder Compute Dashboard](https://www.thundercompute.com/)
2. Click "Create Instance"
3. **CRITICAL**: Select **"Production Mode"** (not Prototyping!)
4. Choose:
   - GPU: **A100 80GB**
   - Region: Same as current instance for consistency
5. Note down:
   - IP Address
   - SSH Port
   - SSH Key Path (usually `~/.thunder/keys/XXXX`)

### Step 2: Add Instance to SSH Config (Local Machine)

Run on your **local machine**:

```bash
bash add_production_instance.sh
```

Follow prompts to add your instance (use a memorable alias like `tnr-prod`, `tnr-1`, etc.).

**Note:** You can have multiple Thunder Compute instances configured. The scripts work with any alias you choose.

### Step 3: Setup Instance (Remote)

Copy setup script to instance (replace `<your-host>` with your chosen alias):

```bash
scp setup_production.sh <your-host>:~/
```

SSH into instance and run setup:

```bash
ssh <your-host>
bash setup_production.sh
```

This will:
- Install all dependencies
- Clone repository
- Setup Python environment with EnvPool
- Run smoke test to verify EnvPool compatibility
- Configure tmux

**Expected time: 5-10 minutes**

### Step 4: Launch Training (Remote)

From the instance:

```bash
cd ~/atari-rl-dashboard
bash launch_production_training.sh
```

This will:
- Start 6 games in parallel in tmux
- Create monitoring dashboards
- Setup GPU monitoring
- Provide instructions for detaching

### Step 5: Monitor Progress (Local or Remote)

**From your local machine:**

```bash
python monitor_production.py --host <your-host> --watch
```

Replace `<your-host>` with your SSH config alias (e.g., `tnr-prod`, `tnr-1`, etc.)

**From the instance:**

```bash
tmux attach -t atari-training
# Press Ctrl+B, then 1 to see monitor window
```

## 📁 New Files

| File | Purpose | Location |
|------|---------|----------|
| `setup_production.sh` | Instance setup script | Run on **instance** |
| `train_production_batch.py` | Parallel training orchestrator | Runs on **instance** |
| `monitor_production.py` | Progress monitoring | Run from **anywhere** |
| `launch_production_training.sh` | One-command launcher | Run on **instance** |
| `add_production_instance.sh` | SSH config helper | Run on **local machine** |

## 🎮 Training Details

### Games & Episodes

| Game | Episodes | Expected Time | Priority |
|------|----------|---------------|----------|
| MsPacman | 30,000 | 4.5 hours | Start first |
| Asteroids | 25,000 | 3.7 hours | Start first |
| Seaquest | 25,000 | 3.7 hours | Start first |
| BeamRider | 20,000 | 3.0 hours | Start first |
| SpaceInvaders | 15,000 | 2.2 hours | Start first |
| Enduro | 15,000 | 2.2 hours | Start first |
| Breakout | 10,000 | 0.4 hours | Queue (fast!) |
| Boxing | 10,000 | 1.5 hours | Queue |
| Freeway | 3,000 | 0.4 hours | Queue (fast!) |
| Pong | 3,000 | 0.4 hours | Queue (fast!) |

### Configuration

**Per-game settings:**
- Environments: 256 parallel
- Batch size: 1024
- Learning rate: 6.25e-5
- Frame stack: 4
- Autosave: Every 90 seconds + every 100 episodes

**System configuration:**
- 6 games running simultaneously
- 4 games in queue (auto-start when slot opens)
- Total GPU memory: ~15-20 GB (25% of A100)
- Target GPU utilization: 90-95%

## 🖥️ Tmux Layout

The launcher creates 4 tmux windows:

```
Window 0: batch-training  ← Main orchestrator
Window 1: monitor         ← Real-time progress
Window 2: gpu             ← GPU utilization (nvidia-smi)
Window 3: logs            ← Manual log viewing
```

### Tmux Commands

```bash
# Attach to session
tmux attach -t atari-training

# Detach from session (inside tmux)
Ctrl+B, then D

# Switch windows (inside tmux)
Ctrl+B, then 0-3  # Window number
Ctrl+B, then N    # Next window
Ctrl+B, then P    # Previous window

# Kill session (if needed)
tmux kill-session -t atari-training
```

## 📊 Monitoring

### Real-time Monitor

**Local machine:**
```bash
python monitor_production.py --host <your-host> --watch --interval 60
```

**Instance:**
```bash
python monitor_production.py --watch
```

Shows:
- GPU utilization and memory
- Active training processes
- Episode progress per game
- Training speed (eps/hr)
- ETA to completion
- Checkpoint counts

### Manual Checks

```bash
# GPU status
nvidia-smi

# Specific game log
tail -f training_mspacman.log

# All recent progress
tail -n 5 training_*.log

# Model checkpoints
ls -lh saved_models/MsPacman/
```

## 🔧 Troubleshooting

### EnvPool Crashes

**Symptoms:** Games crash with "GPU Error" message

**Diagnosis:** You're on Prototyping Mode, not Production Mode!

**Solution:**
1. Verify mode: Check Thunder Compute dashboard
2. If Prototyping: Must create new Production instance
3. Cannot switch existing instance modes

### Low GPU Utilization

**Symptoms:** GPU below 80%

**Possible causes:**
- Not enough parallel games (increase `--parallel`)
- Environments too few (increase `--num-envs`)
- CPU bottleneck (check `htop`)

**Solution:**
```bash
# Try 8 parallel with fewer envs per game
python train_production_batch.py --parallel 8 --num-envs 128
```

### Process Frozen/Zombie

**Symptoms:** Process exists but no progress

**Diagnosis:**
```bash
# Check if actually working
ps aux | grep train_envpool.py

# Check recent log activity
ls -lt training_*.log | head
```

**Solution:**
```bash
# Kill and restart
pkill -f train_envpool.py
bash launch_production_training.sh
```

### Out of Memory

**Symptoms:** CUDA out of memory errors

**Solution:** Reduce parallel games or batch size:
```bash
python train_production_batch.py --parallel 4 --batch-size 512
```

## 💾 Model Management

### Automatic Saves

Models auto-save:
- Every **90 seconds** (autosave)
- Every **100 episodes** (checkpoint)
- On **best reward** (best_model.pt)

Location: `~/atari-rl-dashboard/saved_models/`

### Download Models to Local

**Single game:**
```bash
scp -r <your-host>:~/atari-rl-dashboard/saved_models/MsPacman ./
```

**All models:**
```bash
scp -r <your-host>:~/atari-rl-dashboard/saved_models ./saved_models_production
```

**Best models only:**
```bash
scp <your-host>:~/atari-rl-dashboard/saved_models/*/best_model.pt ./
```

### Backup Strategy

Models are critical! Back up regularly:

```bash
# On instance - create tarball
cd ~/atari-rl-dashboard
tar -czf models_backup_$(date +%Y%m%d_%H%M%S).tar.gz saved_models/

# Download to local
scp <your-host>:~/atari-rl-dashboard/models_backup_*.tar.gz ./backups/
```

## 📈 Expected Timeline

**Day 1 (0-24 hours):**
- Hour 0-6: First 6 games training
- Hour 0.4: Breakout completes → Pong starts
- Hour 0.8: Freeway completes → Boxing starts  
- Hour 1.5-3: Pong, Boxing complete → Next in queue
- Hour 6-8: First batch (6 games) completes
- Hour 8-24: Second batch training

**Day 2 (24-48 hours):**
- Hour 24-36: Remaining games
- Hour 36-48: Long games (MsPacman, Asteroids)

**Day 3 (48-72 hours):**
- Hour 48-60: Final long games
- Hour 60+: Buffer for any issues

**Total: 2.5-3 days** for all 10 games

## 🛡️ Safety & Best Practices

1. **Always use tmux/screen** - Training continues after disconnect
2. **Monitor first 2 hours** - Catch issues early
3. **Check logs regularly** - Look for errors or freezes
4. **Backup models daily** - Download to local machine
5. **Keep Prototyping running** - As backup until Production completes
6. **Monitor costs** - Production Mode is expensive per hour
7. **Delete instance when done** - Avoid unnecessary charges

## 🔄 Comparison with Current Setup

| Aspect | Current (Prototyping) | New (Production) |
|--------|----------------------|------------------|
| Instance Type | Prototyping Mode | Production Mode |
| Script | train.py | train_envpool.py |
| Games Active | 3 (Freeway, Enduro, BeamRider) | 6 simultaneous |
| Speed | 300-700 eps/hr | 6,000-24,000 eps/hr |
| Issues | Frequent crashes/freezes | Stable (no GPU restrictions) |
| Management | Manual restarts | Automatic queue |
| Duration | 10-14 days | 2-3 days |

## 📞 Support

**Issues with scripts:**
- Check logs: `tail -100 training_GAME.log`
- Monitor GPU: `nvidia-smi`
- Verify processes: `ps aux | grep train`

**Thunder Compute issues:**
- [Thunder Compute Docs](https://www.thundercompute.com/docs)
- [Production Mode Guide](https://www.thundercompute.com/docs/production-mode)

**EnvPool issues:**
- [EnvPool GitHub](https://github.com/sail-sg/envpool)
- Verify version: `python -c "import envpool; print(envpool.__version__)"`

## ✅ Success Criteria

Training is successful when:
- ✅ All 10 games complete target episodes
- ✅ No GPU compatibility errors
- ✅ 90-95% GPU utilization maintained
- ✅ Total time < 72 hours
- ✅ All checkpoints saved properly
- ✅ No manual intervention required

---

**Ready to start?** Follow the Quick Start steps above! 🚀

