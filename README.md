# Atari RL Training Dashboard

Real-time visualization dashboard for training reinforcement learning agents on Atari games.

## Features

- Live game visualization at ~30 FPS
- Real-time training metrics (Episode, Reward, Epsilon, FPS)
- Support for 10 Atari games (Pong, Breakout, Ms. Pac-Man, etc.)
- Modern React frontend with WebSocket streaming
- Clean Flask-SocketIO backend

## Project Structure

```
atari/
├── server.py           # Flask-SocketIO server
├── frame_streamer.py   # Efficient frame encoding & streaming
├── game_environments.py # Game management
├── atari_agent.py      # Basic RL agent
├── run_server.py       # Entry point
├── requirements.txt    # Dependencies
└── frontend/
    ├── index.html      # React entry point
    ├── app.js          # React components
    └── styles.css      # Styling
```

## Quick Start

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Run the server:
```bash
python run_server.py
```

3. Open http://localhost:5001 in your browser

## Usage

1. Select a game from the dropdown
2. Click "Start" to begin training
3. Watch the AI learn in real-time
4. Click "Stop" to end training

## Available Games

- Pong
- Breakout
- Space Invaders
- Asteroids
- Ms. Pac-Man
- Boxing
- Seaquest
- Beam Rider
- Enduro
- Freeway

