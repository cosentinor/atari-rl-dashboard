#!/usr/bin/env python3
"""
Production Batch Training Orchestrator for Atari RL.
Optimized for Thunder Compute Production Mode with A100 GPU.

Runs 6 games in parallel to maximize GPU utilization (target: 95%).
Automatically queues and starts remaining games as slots become available.
"""

import argparse
import subprocess
import time
import sys
import os
import signal
from pathlib import Path
from datetime import datetime
import json

# Production game list - ordered by training duration (longest first)
PRODUCTION_GAMES = [
    "MsPacman",     # 30K episodes - ~4.5 hours
    "Asteroids",    # 25K episodes - ~3.7 hours
    "Seaquest",     # 25K episodes - ~3.7 hours
    "BeamRider",    # 20K episodes - ~3.0 hours
    "SpaceInvaders",# 15K episodes - ~2.2 hours
    "Enduro",       # 15K episodes - ~2.2 hours
    "Breakout",     # 10K episodes - ~0.4 hours (FAST!)
    "Boxing",       # 10K episodes - ~1.5 hours
    "Freeway",      # 3K episodes - ~0.4 hours
    "Pong"          # 3K episodes - ~0.4 hours
]

# Episode targets from config
EPISODE_TARGETS = {
    "MsPacman": 30000,
    "Asteroids": 25000,
    "Seaquest": 25000,
    "BeamRider": 20000,
    "SpaceInvaders": 15000,
    "Enduro": 15000,
    "Breakout": 10000,
    "Boxing": 10000,
    "Freeway": 3000,
    "Pong": 3000
}

class ProductionBatchTrainer:
    """Manages parallel training with intelligent queue management."""
    
    def __init__(self, max_parallel=6, num_envs=256, batch_size=1024):
        self.max_parallel = max_parallel
        self.num_envs = num_envs
        self.batch_size = batch_size
        self.processes = {}  # game_name -> subprocess
        self.start_times = {}  # game_name -> start_time
        self.completed = []
        self.failed = []
        self.last_report_time = time.time()
        self.report_interval = 1800  # 30 minutes
        
    def get_log_file(self, game):
        """Get log file path for a game."""
        return f"training_{game.lower()}.log"
    
    def get_episodes_target(self, game):
        """Get episode target for a game."""
        return EPISODE_TARGETS.get(game, 10000)
    
    def start_game(self, game):
        """Start training for a single game."""
        log_file = self.get_log_file(game)
        episodes = self.get_episodes_target(game)
        
        print(f"🏃 Starting: {game}")
        print(f"   Episodes: {episodes:,}")
        print(f"   Envs: {self.num_envs}")
        print(f"   Batch size: {self.batch_size}")
        print(f"   Log: {log_file}")
        
        cmd = [
            sys.executable, "train_envpool.py",
            "--game", game,
            "--episodes", str(episodes),
            "--num-envs", str(self.num_envs),
            "--batch-size", str(self.batch_size)
        ]
        
        try:
            with open(log_file, "w") as f:
                proc = subprocess.Popen(
                    cmd,
                    stdout=f,
                    stderr=subprocess.STDOUT,
                    text=True
                )
                self.processes[game] = proc
                self.start_times[game] = time.time()
                print(f"   ✅ Started (PID: {proc.pid})")
                return True
        except Exception as e:
            print(f"   ❌ Failed to start: {e}")
            self.failed.append(game)
            return False
    
    def check_finished_processes(self):
        """Check for finished processes and return list of finished games."""
        finished_games = []
        for game, proc in list(self.processes.items()):
            if proc.poll() is not None:
                exit_code = proc.returncode
                elapsed = time.time() - self.start_times[game]
                
                if exit_code == 0:
                    print(f"✅ Finished: {game}")
                    print(f"   Time: {elapsed/3600:.2f} hours")
                    print(f"   Exit code: {exit_code}")
                    self.completed.append(game)
                else:
                    print(f"❌ Failed: {game}")
                    print(f"   Exit code: {exit_code}")
                    print(f"   Check log: {self.get_log_file(game)}")
                    self.failed.append(game)
                
                finished_games.append(game)
        
        # Remove finished from active processes
        for game in finished_games:
            self.processes.pop(game)
            
        return finished_games
    
    def get_gpu_status(self):
        """Get current GPU utilization."""
        try:
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=utilization.gpu,memory.used,memory.total", 
                 "--format=csv,noheader,nounits"],
                capture_output=True,
                text=True,
                timeout=5
            )
            if result.returncode == 0:
                util, mem_used, mem_total = result.stdout.strip().split(", ")
                return {
                    "utilization": int(util),
                    "memory_used": int(mem_used),
                    "memory_total": int(mem_total)
                }
        except Exception as e:
            print(f"⚠️  Could not get GPU status: {e}")
        return None
    
    def print_progress_report(self):
        """Print detailed progress report."""
        print("\n" + "=" * 70)
        print(f"📊 PROGRESS REPORT - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("=" * 70)
        
        # GPU Status
        gpu = self.get_gpu_status()
        if gpu:
            mem_pct = (gpu['memory_used'] / gpu['memory_total']) * 100
            print(f"🎮 GPU: {gpu['utilization']}% utilization | "
                  f"{gpu['memory_used']}MB / {gpu['memory_total']}MB ({mem_pct:.1f}%)")
        
        # Active training
        print(f"\n🏃 Active Training ({len(self.processes)} games):")
        for game in self.processes.keys():
            elapsed = time.time() - self.start_times[game]
            target = self.get_episodes_target(game)
            print(f"   • {game:15} | {elapsed/3600:.1f}h elapsed | Target: {target:,} eps")
        
        # Completed
        if self.completed:
            print(f"\n✅ Completed ({len(self.completed)} games):")
            for game in self.completed:
                print(f"   • {game}")
        
        # Failed
        if self.failed:
            print(f"\n❌ Failed ({len(self.failed)} games):")
            for game in self.failed:
                print(f"   • {game}")
        
        print("=" * 70 + "\n")
    
    def run(self, games):
        """Run batch training with queue management."""
        queue = list(games)
        total_games = len(queue)
        
        print("=" * 70)
        print("🚀 PRODUCTION BATCH TRAINING")
        print("=" * 70)
        print(f"Games to train: {total_games}")
        print(f"Parallel slots: {self.max_parallel}")
        print(f"Queue: {', '.join(queue)}")
        print("=" * 70)
        print()
        
        try:
            while queue or self.processes:
                # Check for finished processes
                finished = self.check_finished_processes()
                
                # Start new games if slots available
                while len(self.processes) < self.max_parallel and queue:
                    game = queue.pop(0)
                    
                    # Add startup delay for stability
                    if self.processes:
                        print(f"⏳ Waiting 10 seconds before starting next game...")
                        time.sleep(10)
                    
                    self.start_game(game)
                    print()
                
                # Print progress report periodically
                current_time = time.time()
                if current_time - self.last_report_time >= self.report_interval:
                    self.print_progress_report()
                    self.last_report_time = current_time
                
                # Wait before next check
                if queue or self.processes:
                    time.sleep(15)  # Check every 15 seconds
        
        except KeyboardInterrupt:
            print("\n" + "=" * 70)
            print("🛑 INTERRUPT RECEIVED - Shutting down gracefully...")
            print("=" * 70)
            
            for game, proc in self.processes.items():
                print(f"Terminating {game} (PID: {proc.pid})...")
                proc.terminate()
                
            # Wait for processes to terminate
            print("Waiting for processes to finish...")
            time.sleep(5)
            
            # Force kill if still alive
            for game, proc in self.processes.items():
                if proc.poll() is None:
                    print(f"Force killing {game}...")
                    proc.kill()
            
            sys.exit(1)
        
        # Final report
        print("\n" + "=" * 70)
        print("🎉 BATCH TRAINING COMPLETE!")
        print("=" * 70)
        print(f"✅ Completed: {len(self.completed)}/{total_games} games")
        if self.completed:
            print(f"   {', '.join(self.completed)}")
        
        if self.failed:
            print(f"\n❌ Failed: {len(self.failed)} games")
            print(f"   {', '.join(self.failed)}")
            print("\n⚠️  Check logs for details:")
            for game in self.failed:
                print(f"   tail {self.get_log_file(game)}")
        
        print("=" * 70)
        
        return len(self.failed) == 0


def main():
    parser = argparse.ArgumentParser(
        description="Production Batch Trainer - Optimized for A100 GPU",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Train all 10 games with 6 parallel (recommended):
  python train_production_batch.py --parallel 6
  
  # Train specific games:
  python train_production_batch.py --games MsPacman Qbert Asteroids
  
  # Conservative mode (4 parallel):
  python train_production_batch.py --parallel 4 --num-envs 128
        """
    )
    
    parser.add_argument(
        "--parallel", "-p", 
        type=int, 
        default=6, 
        help="Max parallel games (default: 6 for A100)"
    )
    parser.add_argument(
        "--episodes", "-e", 
        type=int, 
        help="Override episodes for all games (default: per-game targets)"
    )
    parser.add_argument(
        "--games", "-g", 
        nargs="+", 
        help="Specific games to train (default: all 10)"
    )
    parser.add_argument(
        "--num-envs", "-n",
        type=int,
        default=256,
        help="Parallel environments per game (default: 256)"
    )
    parser.add_argument(
        "--batch-size", "-b",
        type=int,
        default=1024,
        help="Batch size for learning (default: 1024)"
    )
    
    args = parser.parse_args()
    
    # Override episodes if specified
    if args.episodes:
        for game in EPISODE_TARGETS:
            EPISODE_TARGETS[game] = args.episodes
    
    # Determine which games to train
    games = args.games if args.games else PRODUCTION_GAMES
    
    # Validate games
    for game in games:
        if game not in PRODUCTION_GAMES:
            print(f"❌ Unknown game: {game}")
            print(f"Available: {', '.join(PRODUCTION_GAMES)}")
            sys.exit(1)
    
    # Create trainer and run
    trainer = ProductionBatchTrainer(
        max_parallel=args.parallel,
        num_envs=args.num_envs,
        batch_size=args.batch_size
    )
    
    success = trainer.run(games)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()

