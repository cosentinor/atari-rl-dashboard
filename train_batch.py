#!/usr/bin/env python3
"""
Batch Training Orchestrator for Atari RL.
Runs up to N games in parallel to maximize GPU utilization on A100.
"""

import argparse
import subprocess
import time
import sys
import os
from pathlib import Path
from datetime import datetime

# Configuration
GAMES_TO_TRAIN = [
    "Freeway",
    "Breakout",
    "Boxing",
    "SpaceInvaders",
    "Enduro",
    "BeamRider",
    "Seaquest",
    "Asteroids"
]

def run_batch(games, max_parallel=3, episodes=3000):
    """Run games in parallel lanes."""
    processes = {}
    queue = list(games)
    completed = []
    
    print(f"🚀 Starting batch training for {len(games)} games...")
    print(f"Parallel lanes: {max_parallel}")
    print(f"Queue: {', '.join(queue)}")
    print("-" * 60)

    try:
        while queue or processes:
            # Check for finished processes
            finished_games = []
            for game, proc in processes.items():
                if proc.poll() is not None:
                    print(f"✅ Finished: {game} (Exit code: {proc.returncode})")
                    finished_games.append(game)
            
            for game in finished_games:
                processes.pop(game)
                completed.append(game)

            # Start new processes if lanes available
            while len(processes) < max_parallel and queue:
                game = queue.pop(0)
                log_file = f"training_{game.lower()}.log"
                print(f"🏃 Starting: {game} (Logging to {log_file})")
                
                # Optimized for A100: num-envs=64, batch-size=1024
                # We divide 128 envs by max_parallel to keep CPU usage sane (16 cores total)
                num_envs = 128 // max_parallel
                
                cmd = [
                    sys.executable, "train_envpool.py",
                    "--game", game,
                    "--episodes", str(episodes),
                    "--num-envs", str(num_envs),
                    "--batch-size", "1024"
                ]
                
                with open(log_file, "w") as f:
                    proc = subprocess.Popen(
                        cmd,
                        stdout=f,
                        stderr=subprocess.STDOUT,
                        text=True
                    )
                    processes[game] = proc

            if queue or processes:
                time.sleep(10)  # Check every 10 seconds

    except KeyboardInterrupt:
        print("\n🛑 Interrupt received. Shutting down parallel processes...")
        for game, proc in processes.items():
            print(f"Killing {game}...")
            proc.terminate()
        sys.exit(1)

    print("-" * 60)
    print(f"🎉 All training complete! Games finished: {', '.join(completed)}")

def main():
    parser = argparse.ArgumentParser(description="Atari Batch Trainer")
    parser.add_argument("--parallel", "-p", type=int, default=2, help="Max parallel games")
    parser.add_argument("--episodes", "-e", type=int, default=10000, help="Episodes per game")
    parser.add_argument("--games", "-g", nargs="+", help="Specific games to train (defaults to all 8)")
    
    args = parser.parse_args()
    
    games = args.games if args.games else GAMES_TO_TRAIN
    run_batch(games, max_parallel=args.parallel, episodes=args.episodes)

if __name__ == "__main__":
    main()

