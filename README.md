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
├── train.py            # Headless training script (for cloud)
├── run_server.py       # Web UI entry point (for local)
├── server.py           # Flask-SocketIO server
├── rainbow_agent.py    # Rainbow DQN implementation
├── config.py           # Training configuration
├── model_manager.py    # Checkpoint management
├── db_manager.py       # Training metrics database
├── frame_streamer.py   # Frame encoding & streaming
├── game_environments.py # Game management
├── Dockerfile          # Docker image for cloud training
├── docker-compose.yml  # Docker orchestration
├── requirements.txt    # Dependencies
├── saved_models/       # Trained model checkpoints
└── frontend/
    ├── index.html
    ├── app.js
    └── styles.css
```

## Quick Start

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

### Cloud Training (Thunder Compute / AWS / etc.)

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
