# Atari RL Training Dashboard

Real-time visualization dashboard for training Rainbow DQN agents on Atari games.

## Features

- **Rainbow DQN Agent**: State-of-the-art algorithm combining 6 improvements
  - Dueling Network Architecture
  - Noisy Layers for exploration
  - Prioritized Experience Replay
  - N-step Returns
  - Distributional RL (C51)
  - Double DQN

- **Two Training Modes**:
  - **Headless** (`train.py`): Fast training on cloud GPUs (Thunder Compute)
  - **Web UI** (`run_server.py`): Watch AI play, continue training locally

- **Features**:
  - Live game visualization at ~30 FPS
  - Real-time training metrics and charts
  - 90-second autosave (never lose progress)
  - Model checkpoint management
  - GPU acceleration (CUDA/MPS)
  - 10 Atari games supported

## Project Structure

```
atari/
├── train.py                        # Headless training script (standard)
├── train_envpool.py                # Ultra-fast EnvPool training
├── train_production_batch.py       # Production parallel orchestrator (NEW)
├── monitor_production.py           # Production training monitor (NEW)
├── setup_production.sh             # Production instance setup (NEW)
├── launch_production_training.sh   # Production launcher (NEW)
├── add_production_instance.sh      # SSH config helper (NEW)
├── PRODUCTION_SETUP.md             # Production mode guide (NEW)
├── run_server.py                   # Web UI entry point (for local)
├── server.py                       # Flask-SocketIO server
├── rainbow_agent.py                # Rainbow DQN implementation
├── config.py                       # Training configuration
├── model_manager.py                # Checkpoint management
├── db_manager.py                   # Training metrics database
├── frame_streamer.py               # Frame encoding & streaming
├── game_environments.py            # Game management
├── Dockerfile                      # Docker image for cloud training
├── docker-compose.yml              # Docker orchestration
├── requirements.txt                # Dependencies
└── saved_models/                   # Trained model checkpoints
```

## 🚀 Quick Start Guides

**Setting up new Thunder Compute instance?** → [THUNDER_COMPUTE_SETUP.md](THUNDER_COMPUTE_SETUP.md) ⚡  
Or run: `bash setup_new_instance.sh` for step-by-step instructions.

### Local Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Run the web UI:
```bash
python run_server.py
```

3. Open http://localhost:5001 in your browser

Frontend build notes:
- The server will serve a built frontend if `FRONTEND_BUILD_DIR` points to the build output.
- If no build is present, the root route returns a JSON warning and the API/WebSocket still work.
- Frontend source is maintained separately; this repo only expects a built bundle (or an external host).

### Headless Training (Recommended for serious training)

```bash
# Train Pong (POC - ~2 hours)
python train.py --game Pong --episodes 3000

# Train Breakout
python train.py --game Breakout --episodes 10000

# Train all games with recommended episodes
python train.py --game all

# List available games
python train.py --list-games
```

## Training Workflow

### ⚡ Production Mode Training (Recommended - Thunder Compute)

**For maximum speed (6,000-24,000 eps/hr):**

📌 **Quick Reference:** [THUNDER_COMPUTE_SETUP.md](THUNDER_COMPUTE_SETUP.md) - **Bookmark this!**  
📖 **Full Guide:** [PRODUCTION_SETUP.md](PRODUCTION_SETUP.md)

**Quick Start:**
1. Create Thunder Compute Production Mode instance (A100 80GB recommended)
2. Configure SSH locally: `bash add_production_instance.sh`
3. Copy and run setup script on instance: `bash setup_production.sh`
4. Launch training: `bash launch_production_training.sh`
5. Monitor: `python monitor_production.py --host <your-host> --watch`

**Benefits:**
- ✅ 10-20x faster than standard training
- ✅ 6 games in parallel
- ✅ Complete all 10 games in 2-3 days
- ✅ No GPU restrictions
- ✅ A100 80GB GPU utilization: 50-80%

### Cloud Training (Standard - Thunder Compute / AWS / etc.)

1. **Clone and setup on cloud GPU**:
```bash
git clone https://github.com/cosentinor/atari-rl-dashboard.git
cd atari-rl-dashboard
pip install -r requirements.txt
```

2. **Train headlessly**:
```bash
# Train Pong as POC
python train.py --game Pong --episodes 3000

# Or train all games
python train.py --game all
```

3. **Download trained models**:
```bash
# From your local machine
scp -r user@cloud-server:~/atari-rl-dashboard/saved_models/* ./saved_models/
```

### Local Usage

After downloading trained models:

1. **Watch AI play**:
   - Start web UI: `python run_server.py`
   - Select game → Select checkpoint → Click Start
   - Watch the trained agent play!

2. **Continue training**:
   - Same as above, but the agent will keep learning (slower locally)

## Remote Management (for AI Agents)

This project is configured for automated management via Cursor AI agents.

- **Connection**: Connection details are specified in `.cursorrules`.
- **Deployment**: Agents can deploy code using `git pull` and `systemctl restart` over SSH.
- **Monitoring**: Logs are available at `/var/log/atari/` on the remote server.
- **Health**: The `deployment/health_check.sh` script monitors service availability.

To enable automation, ensure your SSH key is authorized on the VPS and the IP address is correctly referenced.

### Docker (Alternative)

```bash
# Build image
docker build -t atari-rl .

# Train with GPU
docker run --gpus all -v $(pwd)/saved_models:/app/saved_models atari-rl \
    python train.py --game Pong --episodes 3000

# Or use docker-compose
docker-compose run train python train.py --game Pong --episodes 3000
```

## Available Games

| Game | Difficulty | Recommended Episodes | Notes |
|------|------------|---------------------|-------|
| Pong | Easy | 3,000 | Great for testing |
| Freeway | Easy | 3,000 | Simple reward |
| Breakout | Medium | 10,000 | Classic benchmark |
| Boxing | Medium | 10,000 | Two-player style |
| Space Invaders | Medium | 15,000 | Iconic shooter |
| Enduro | Medium | 15,000 | Racing game |
| Beam Rider | Hard | 20,000 | Fast-paced |
| Seaquest | Hard | 25,000 | Resource management |
| Asteroids | Hard | 25,000 | 360° movement |
| Ms. Pac-Man | Very Hard | 30,000 | Complex navigation |

## Training Tips

1. **Start with Pong**: Validates setup, trains fast, clear success metrics
2. **Use Turbo mode** in web UI for faster local training (no visualization delay)
3. **Checkpoints save every 90 seconds** - safe to interrupt anytime
4. **Cloud GPUs recommended** for serious training (3-10x faster than Mac)

## Requirements

- Python 3.10+
- PyTorch 2.0+
- CUDA GPU (recommended) or Apple Silicon (MPS)
- 8GB+ RAM (replay buffer)

## License

MIT
