#!/usr/bin/env python3
"""
GPU-OPTIMIZED Training Script for A100.
Uses large network architecture and aggressive batching to maximize GPU utilization.

Target: 60-80% GPU utilization on A100.
"""

import argparse
import logging
import time
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)

# Try to import EnvPool
try:
    import envpool
    HAS_ENVPOOL = True
    logger.info(f"✅ EnvPool version: {envpool.__version__}")
except ImportError:
    HAS_ENVPOOL = False
    logger.error("❌ EnvPool required. Run: pip install envpool")
    sys.exit(1)

from rainbow_agent_large import RainbowAgentLarge, get_device
from model_manager import ModelManager
from config import (
    GAME_PRESETS,
    AUTOSAVE_INTERVAL_SECONDS,
    get_game_env_id,
    get_recommended_episodes,
    list_games
)


ENVPOOL_GAMES = {
    "Pong": "Pong-v5",
    "Breakout": "Breakout-v5",
    "SpaceInvaders": "SpaceInvaders-v5",
    "MsPacman": "MsPacman-v5",
    "Qbert": "Qbert-v5",
    "Seaquest": "Seaquest-v5",
    "BeamRider": "BeamRider-v5",
    "Enduro": "Enduro-v5",
    "Asteroids": "Asteroids-v5",
    "Boxing": "Boxing-v5",
    "Freeway": "Freeway-v5",
}


def train_gpu_optimized(
    game_name: str,
    episodes: int,
    num_envs: int = 64,
    batch_size: int = 4096,
    learning_updates: int = 4,
    checkpoint_path: str = None,
    save_dir: str = "saved_models"
):
    """
    Train using LARGE network and aggressive GPU batching.
    
    Args:
        learning_updates: Number of gradient updates per environment step (increases GPU compute)
    """
    logger.info("=" * 60)
    logger.info(f"🚀 GPU-OPTIMIZED TRAINING: {game_name}")
    logger.info(f"Episodes: {episodes}")
    logger.info(f"Parallel Environments: {num_envs}")
    logger.info(f"Batch Size: {batch_size} (A100 optimized)")
    logger.info(f"Learning Updates per Step: {learning_updates}")
    logger.info("=" * 60)
    
    if game_name not in ENVPOOL_GAMES:
        logger.error(f"Game '{game_name}' not available in EnvPool")
        return None
    
    envpool_id = ENVPOOL_GAMES[game_name]
    logger.info(f"EnvPool Environment: {envpool_id}")
    
    # Setup device
    device = get_device()
    logger.info(f"Device: {device}")
    
    # Check GPU
    if torch.cuda.is_available():
        gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1e9
        gpu_name = torch.cuda.get_device_name(0)
        logger.info(f"GPU: {gpu_name} ({gpu_mem:.1f} GB)")
    else:
        logger.error("GPU not available! This script requires CUDA.")
        return None
    
    # Create EnvPool environment
    logger.info(f"Creating {num_envs} EnvPool environments...")
    try:
        num_threads = min(num_envs, 8)
        
        env = envpool.make(
            envpool_id,
            env_type="gymnasium",
            num_envs=num_envs,
            batch_size=num_envs,
            num_threads=num_threads,
            seed=42,
            episodic_life=True,
            reward_clip=True,
            stack_num=4,
            gray_scale=True,
            img_height=84,
            img_width=84,
        )
        logger.info(f"✅ EnvPool created with {num_envs} environments")
    except Exception as e:
        logger.error(f"Failed to create EnvPool: {e}")
        return None
    
    num_actions = env.action_space.n
    logger.info(f"Action space: {num_actions} actions")
    
    # Create model manager
    model_manager = ModelManager(base_dir=save_dir)
    env_id = get_game_env_id(game_name)
    
    # Create LARGE Rainbow agent
    agent = RainbowAgentLarge(
        state_shape=(4, 84, 84),
        num_actions=num_actions,
        device=device,
        batch_size=batch_size,
        use_augmentation=True
    )
    
    # Load checkpoint if specified
    start_episode = 0
    if checkpoint_path:
        try:
            model_manager.load_checkpoint(agent, env_id, checkpoint_path)
            start_episode = agent.episode_count
            logger.info(f"Resumed from checkpoint: episode {start_episode}")
        except Exception as e:
            logger.warning(f"Could not load checkpoint: {e}")
    
    # Training metrics
    best_reward = float('-inf')
    recent_rewards = []
    total_steps = agent.step_count
    total_frames = 0
    training_start = time.time()
    last_autosave = time.time()
    last_log_time = time.time()
    
    # Episode tracking
    episode_rewards = np.zeros(num_envs)
    episode_steps = np.zeros(num_envs, dtype=int)
    completed_episodes = 0
    
    logger.info("")
    logger.info("Starting GPU-optimized training loop...")
    logger.info("-" * 60)
    
    # Reset all environments
    obs, info = env.reset()
    states = obs.astype(np.float32) / 255.0
    
    try:
        while completed_episodes < episodes:
            # Batch action selection on GPU
            states_tensor = torch.FloatTensor(states).to(device)
            
            with torch.no_grad():
                q_values = agent.online_net.get_q_values(states_tensor)
                
                # Epsilon-greedy
                if np.random.random() < agent.current_epsilon:
                    actions = np.random.randint(0, num_actions, size=num_envs)
                else:
                    actions = q_values.argmax(dim=1).cpu().numpy()
            
            # Step ALL environments
            next_obs, rewards, terminated, truncated, infos = env.step(actions)
            dones = terminated | truncated
            total_frames += num_envs
            
            next_states = next_obs.astype(np.float32) / 255.0
            
            # Update episode tracking (vectorized)
            episode_rewards += rewards
            episode_steps += 1
            total_steps += num_envs
            
            # Find completed episodes
            done_indices = np.where(dones)[0]
            
            for i in done_indices:
                completed_episodes += 1
                agent.episode_count = start_episode + completed_episodes
                agent.update_epsilon()
                
                ep_reward = episode_rewards[i]
                recent_rewards.append(ep_reward)
                if len(recent_rewards) > 100:
                    recent_rewards.pop(0)
                
                is_best = ep_reward > best_reward
                if is_best:
                    best_reward = ep_reward
                
                # Log periodically
                current_time = time.time()
                if completed_episodes % 50 == 0 or is_best or (current_time - last_log_time > 10):
                    elapsed = current_time - training_start
                    eps_per_hour = completed_episodes / (elapsed / 3600) if elapsed > 0 else 0
                    fps = total_frames / elapsed if elapsed > 0 else 0
                    avg_reward = np.mean(recent_rewards) if recent_rewards else 0
                    
                    log_msg = (
                        f"Ep {start_episode + completed_episodes:5d} | "
                        f"Reward: {ep_reward:7.1f} | "
                        f"Avg100: {avg_reward:7.1f} | "
                        f"Best: {best_reward:7.1f} | "
                        f"ε: {agent.current_epsilon:.3f} | "
                        f"Eps/hr: {eps_per_hour:.0f} | "
                        f"FPS: {fps/1000:.1f}K"
                    )
                    
                    if is_best:
                        log_msg += " ⭐"
                    
                    logger.info(log_msg)
                    last_log_time = current_time
                
                # Save checkpoints
                if completed_episodes % 100 == 0 or is_best:
                    model_manager.save_checkpoint(
                        agent, env_id, start_episode + completed_episodes,
                        ep_reward, is_best=is_best
                    )
                    if completed_episodes % 100 == 0:
                        logger.info(f"  [Checkpoint] Episode {start_episode + completed_episodes}")
                
                # Reset episode tracking
                episode_rewards[i] = 0
                episode_steps[i] = 0
            
            # Store transitions (sample subset to avoid memory overhead)
            sample_indices = np.random.choice(num_envs, min(32, num_envs), replace=False)
            for i in sample_indices:
                agent.push_transition(
                    states[i], actions[i], rewards[i], next_states[i], dones[i]
                )
            
            states = next_states
            
            # AGGRESSIVE LEARNING: Multiple updates per step to keep GPU busy
            loss = agent.learn(num_updates=learning_updates)
            
            # Autosave
            current_time = time.time()
            if current_time - last_autosave >= AUTOSAVE_INTERVAL_SECONDS:
                model_manager.save_checkpoint(
                    agent, env_id, start_episode + completed_episodes,
                    recent_rewards[-1] if recent_rewards else 0
                )
                last_autosave = current_time
                elapsed = current_time - training_start
                fps = total_frames / elapsed
                
                # Report GPU memory usage
                if torch.cuda.is_available():
                    gpu_mem_used = torch.cuda.memory_allocated() / 1e9
                    gpu_mem_cached = torch.cuda.memory_reserved() / 1e9
                    logger.info(f"  [Autosave] Ep {start_episode + completed_episodes} | "
                               f"FPS: {fps/1000:.1f}K | "
                               f"GPU Mem: {gpu_mem_used:.1f}/{gpu_mem_cached:.1f}GB")
                else:
                    logger.info(f"  [Autosave] Ep {start_episode + completed_episodes} | FPS: {fps/1000:.1f}K")
    
    except KeyboardInterrupt:
        logger.info("\n" + "=" * 60)
        logger.info("Training interrupted")
        model_manager.save_checkpoint(
            agent, env_id, start_episode + completed_episodes,
            recent_rewards[-1] if recent_rewards else 0
        )
    
    finally:
        env.close()
    
    # Final summary
    total_time = time.time() - training_start
    fps = total_frames / total_time if total_time > 0 else 0
    
    logger.info("")
    logger.info("=" * 60)
    logger.info("🎉 TRAINING COMPLETE")
    logger.info(f"Game: {game_name}")
    logger.info(f"Episodes: {completed_episodes}")
    logger.info(f"Total frames: {total_frames:,}")
    logger.info(f"Best reward: {best_reward:.1f}")
    logger.info(f"Final avg (100): {np.mean(recent_rewards):.1f}")
    logger.info(f"Time: {total_time/3600:.2f} hours")
    logger.info(f"Speed: {completed_episodes / (total_time / 3600):.0f} episodes/hour")
    logger.info(f"Throughput: {fps/1000:.1f}K FPS")
    logger.info("=" * 60)
    
    return {
        "game": game_name,
        "episodes": completed_episodes,
        "best_reward": best_reward,
        "fps": fps,
        "time_hours": total_time / 3600
    }


def main():
    parser = argparse.ArgumentParser(
        description="GPU-Optimized Atari RL Training (A100)",
        epilog=f"Available games: {', '.join(ENVPOOL_GAMES.keys())}"
    )
    
    parser.add_argument("--game", "-g", type=str, required=True)
    parser.add_argument("--episodes", "-e", type=int, default=3000)
    parser.add_argument("--num-envs", "-n", type=int, default=64)
    parser.add_argument("--batch-size", "-b", type=int, default=4096)
    parser.add_argument("--learning-updates", "-u", type=int, default=4)
    parser.add_argument("--checkpoint", "-c", type=str)
    parser.add_argument("--save-dir", "-s", type=str, default="saved_models")
    
    args = parser.parse_args()
    
    train_gpu_optimized(
        game_name=args.game,
        episodes=args.episodes,
        num_envs=args.num_envs,
        batch_size=args.batch_size,
        learning_updates=args.learning_updates,
        checkpoint_path=args.checkpoint,
        save_dir=args.save_dir
    )


if __name__ == "__main__":
    main()

