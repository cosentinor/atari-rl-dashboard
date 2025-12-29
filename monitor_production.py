#!/usr/bin/env python3
"""
Production Training Monitor for Atari RL.
Real-time monitoring of parallel training processes.

Can be run locally or on the server:
  Local:  python monitor_production.py --host tnr-prod
  Server: python monitor_production.py
"""

import argparse
import subprocess
import sys
import time
import re
from datetime import datetime, timedelta
from pathlib import Path
import json

# Game episode targets
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

# Expected speeds (episodes/hour) based on historical data
EXPECTED_SPEEDS = {
    "Breakout": 24000,
    "Pong": 6700,
    "Freeway": 6700,
    "SpaceInvaders": 6700,
    "Enduro": 6700,
    "BeamRider": 2400,
    "Boxing": 700,
    "MsPacman": 6700,
    "Asteroids": 6700,
    "Seaquest": 6700
}


class ProductionMonitor:
    """Monitor training progress on Production instance."""
    
    def __init__(self, host=None):
        self.host = host
        self.is_remote = host is not None
        
    def run_command(self, cmd):
        """Run command locally or via SSH."""
        if self.is_remote:
            ssh_cmd = ["ssh", self.host, cmd]
            result = subprocess.run(ssh_cmd, capture_output=True, text=True)
        else:
            result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        
        if result.returncode != 0:
            return None
        return result.stdout.strip()
    
    def get_gpu_status(self):
        """Get GPU utilization and memory."""
        cmd = "nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu --format=csv,noheader,nounits"
        output = self.run_command(cmd)
        
        if output:
            try:
                util, mem_used, mem_total, temp = output.split(", ")
                return {
                    "utilization": int(util),
                    "memory_used": int(mem_used),
                    "memory_total": int(mem_total),
                    "temperature": int(temp)
                }
            except:
                pass
        return None
    
    def get_training_processes(self):
        """Get active training processes."""
        cmd = "ps aux | grep 'train_envpool.py\\|train.py' | grep -v grep"
        output = self.run_command(cmd)
        
        processes = []
        if output:
            for line in output.split('\n'):
                if 'train_envpool.py' in line or 'train.py' in line:
                    parts = line.split()
                    pid = parts[1]
                    cpu = parts[2]
                    mem = parts[3]
                    
                    # Extract game name from command
                    game_match = re.search(r'--game (\w+)', line)
                    game = game_match.group(1) if game_match else "Unknown"
                    
                    script = "envpool" if "envpool" in line else "standard"
                    
                    processes.append({
                        "pid": pid,
                        "game": game,
                        "cpu": cpu,
                        "mem": mem,
                        "script": script
                    })
        
        return processes
    
    def get_log_progress(self, game):
        """Extract progress from training log."""
        log_file = f"~/atari-rl-dashboard/training_{game.lower()}.log"
        cmd = f"tail -50 {log_file} 2>/dev/null | grep 'Ep ' | tail -1"
        output = self.run_command(cmd)
        
        if output:
            # Parse log line like: "Ep  1234 | Reward:   352.0 | Avg100:   286.0 | Best:   352.0 | ..."
            ep_match = re.search(r'Ep\s+(\d+)', output)
            reward_match = re.search(r'Reward:\s+([-\d.]+)', output)
            avg_match = re.search(r'Avg100:\s+([-\d.]+)', output)
            best_match = re.search(r'Best:\s+([-\d.]+)', output)
            eps_hr_match = re.search(r'Eps/hr:\s+([\d.]+)', output)
            
            return {
                "episode": int(ep_match.group(1)) if ep_match else None,
                "reward": float(reward_match.group(1)) if reward_match else None,
                "avg100": float(avg_match.group(1)) if avg_match else None,
                "best": float(best_match.group(1)) if best_match else None,
                "eps_per_hour": float(eps_hr_match.group(1)) if eps_hr_match else None
            }
        
        return None
    
    def get_model_checkpoints(self, game):
        """Get count of saved checkpoints."""
        cmd = f"ls ~/atari-rl-dashboard/saved_models/{game}/*.pt 2>/dev/null | wc -l"
        output = self.run_command(cmd)
        
        if output:
            try:
                return int(output)
            except:
                pass
        return 0
    
    def estimate_completion(self, current_ep, target_ep, eps_per_hour):
        """Estimate time to completion."""
        if not eps_per_hour or eps_per_hour == 0:
            return None
        
        remaining = target_ep - current_ep
        hours_remaining = remaining / eps_per_hour
        
        return timedelta(hours=hours_remaining)
    
    def print_status(self, detailed=False):
        """Print comprehensive status."""
        print("\n" + "=" * 80)
        print(f"📊 PRODUCTION TRAINING MONITOR - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        if self.is_remote:
            print(f"🌐 Remote Host: {self.host}")
        print("=" * 80)
        
        # GPU Status
        gpu = self.get_gpu_status()
        if gpu:
            mem_pct = (gpu['memory_used'] / gpu['memory_total']) * 100
            util_icon = "🔥" if gpu['utilization'] > 80 else "⚡" if gpu['utilization'] > 50 else "💤"
            
            print(f"\n{util_icon} GPU Status:")
            print(f"   Utilization: {gpu['utilization']}%")
            print(f"   Memory: {gpu['memory_used']} MB / {gpu['memory_total']} MB ({mem_pct:.1f}%)")
            print(f"   Temperature: {gpu['temperature']}°C")
        else:
            print("\n⚠️  Could not fetch GPU status")
        
        # Training Processes
        processes = self.get_training_processes()
        print(f"\n🏃 Active Training: {len(processes)} processes")
        
        if not processes:
            print("   No training processes found!")
            print("   Run: python train_production_batch.py --parallel 6")
        else:
            print()
            for proc in processes:
                print(f"   • {proc['game']:15} | PID: {proc['pid']:6} | "
                      f"CPU: {proc['cpu']:5}% | Mem: {proc['mem']:5}% | "
                      f"Type: {proc['script']}")
                
                # Get detailed progress
                if detailed:
                    progress = self.get_log_progress(proc['game'])
                    if progress and progress['episode']:
                        target = EPISODE_TARGETS.get(proc['game'], 10000)
                        pct = (progress['episode'] / target) * 100
                        
                        print(f"     └─ Episode: {progress['episode']:,} / {target:,} ({pct:.1f}%)")
                        print(f"        Reward: {progress['reward']:.1f} | "
                              f"Avg100: {progress['avg100']:.1f} | "
                              f"Best: {progress['best']:.1f}")
                        
                        if progress['eps_per_hour']:
                            print(f"        Speed: {progress['eps_per_hour']:,.0f} eps/hr")
                            
                            eta = self.estimate_completion(
                                progress['episode'], 
                                target, 
                                progress['eps_per_hour']
                            )
                            
                            if eta:
                                hours = int(eta.total_seconds() / 3600)
                                minutes = int((eta.total_seconds() % 3600) / 60)
                                print(f"        ETA: {hours}h {minutes}m")
                    
                    # Checkpoint count
                    checkpoints = self.get_model_checkpoints(proc['game'])
                    if checkpoints > 0:
                        print(f"        Checkpoints: {checkpoints}")
                    
                    print()
        
        print("=" * 80)
    
    def watch(self, interval=30):
        """Continuously monitor and update."""
        print(f"👀 Watching training (updates every {interval}s, Ctrl+C to stop)...")
        
        try:
            while True:
                # Clear screen (works on Unix)
                if sys.platform != "win32":
                    subprocess.run(["clear"])
                
                self.print_status(detailed=True)
                
                print(f"\n⏳ Next update in {interval} seconds...")
                time.sleep(interval)
                
        except KeyboardInterrupt:
            print("\n\n👋 Monitoring stopped.")


def main():
    parser = argparse.ArgumentParser(
        description="Monitor Production Training Progress",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Monitor from server:
  python monitor_production.py
  
  # Monitor remotely from local machine:
  python monitor_production.py --host tnr-prod
  
  # Watch mode with detailed info:
  python monitor_production.py --watch --interval 60
  
  # Quick status check:
  python monitor_production.py --quick
        """
    )
    
    parser.add_argument(
        "--host",
        help="SSH host for remote monitoring (from ~/.ssh/config)"
    )
    parser.add_argument(
        "--watch", "-w",
        action="store_true",
        help="Continuously watch and update"
    )
    parser.add_argument(
        "--interval", "-i",
        type=int,
        default=30,
        help="Update interval in seconds for watch mode (default: 30)"
    )
    parser.add_argument(
        "--quick", "-q",
        action="store_true",
        help="Quick status (no detailed progress)"
    )
    
    args = parser.parse_args()
    
    monitor = ProductionMonitor(host=args.host)
    
    if args.watch:
        monitor.watch(interval=args.interval)
    else:
        monitor.print_status(detailed=not args.quick)


if __name__ == "__main__":
    main()

