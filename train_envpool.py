#!/usr/bin/env python3
"""
ULTRA-FAST Training Script using EnvPool.
EnvPool is a high-performance C++ environment pool that can run
Atari games 10-100x faster than gymnasium!

This will FULLY utilize your A100 GPU.
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
    logger.warning("❌ EnvPool not installed. Run: pip install envpool")

from rainbow_agent import RainbowAgent, get_device
from model_manager import ModelManager
from config import (
    GAME_PRESETS,
    AUTOSAVE_INTERVAL_SECONDS,
    get_game_env_id,
    get_recommended_episodes,
    list_games
)


# EnvPool game ID mapping
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


class FrameStackEnvPool:
    """Frame stacking wrapper for EnvPool environments."""
    
    def __init__(self, num_envs: int, num_frames: int = 4):
        self.num_envs = num_envs
        self.num_frames = num_frames
        self.frames = None
    
    def reset(self, obs):
        """Initialize frame stack with first observation."""
        # obs shape: (num_envs, 84, 84) or (num_envs, 4, 84, 84) if already stacked
        if len(obs.shape) == 4:
            # Already frame-stacked by EnvPool
            self.frames = obs
        else:
            # Stack the same frame num_frames times
            self.frames = np.stack([obs] * self.num_frames, axis=1)
        return self.frames
    
    def step(self, obs):
        """Update frame stack with new observation."""
        if len(obs.shape) == 4:
            # Already frame-stacked by EnvPool
            self.frames = obs
        else:
            # Shift frames and add new one
            self.frames = np.roll(self.frames, shift=-1, axis=1)
            self.frames[:, -1] = obs
        return self.frames


def train_envpool(
    game_name: str,
    episodes: int,
    num_envs: int = 64,
    batch_size: int = 512,
    checkpoint_path: str = None,
    save_dir: str = "saved_models"
):
    """
    Train using EnvPool for ultra-fast environment stepping.
    """
    if not HAS_ENVPOOL:
        logger.error("EnvPool not installed! Run: pip install envpool")
        logger.error("Falling back to standard training...")
        from train_fast import train_game_fast
        return train_game_fast(
            game_name,
            episodes,
            num_envs=num_envs,
            batch_size=batch_size,
            checkpoint_path=checkpoint_path,
            save_dir=save_dir
        )
    
    logger.info("=" * 60)
    logger.info(f"⚡ ENVPOOL ULTRA-FAST TRAINING: {game_name}")
    logger.info(f"Episodes: {episodes}")
    logger.info(f"Parallel Environments: {num_envs}")
    logger.info(f"Batch Size: {batch_size}")
    logger.info("=" * 60)
    
    # Get EnvPool game ID
    if game_name not in ENVPOOL_GAMES:
        logger.error(f"Game '{game_name}' not available in EnvPool")
        logger.info(f"Available games: {list(ENVPOOL_GAMES.keys())}")
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
        
        # Only scale up if the user didn't specify larger values
        if gpu_mem >= 40:
            if batch_size < 1024:
                batch_size = 1024
                logger.info(f"🚀 A100 detected! Auto-scaling batch_size to {batch_size}")
            if num_envs < 128:
                # But don't auto-scale if we are in batch mode (indicated by smaller num_envs)
                if num_envs > 64: 
                    num_envs = 128
                    logger.info(f"🚀 A100 detected! Auto-scaling num_envs to {num_envs}")
    
    # Create EnvPool environment
    logger.info(f"Creating {num_envs} EnvPool environments...")
    try:
        # Use num_threads=0 for envpool to automatically manage threads per environment,
        # but in parallel settings we want to be more explicit.
        # We'll use a safer default or the user-specified num_threads.
        num_threads = min(num_envs, 8) 
        
        env = envpool.make(
            envpool_id,
            env_type="gymnasium",
            num_envs=num_envs,
            batch_size=num_envs,  # Process all envs together
            num_threads=num_threads,
            seed=42,
            episodic_life=True,  # Standard Atari preprocessing
            reward_clip=True,
            stack_num=4,  # Frame stacking built-in!
            gray_scale=True,
            img_height=84,
            img_width=84,
        )
        logger.info(f"✅ EnvPool created with {num_envs} environments")
    except Exception as e:
        logger.error(f"Failed to create EnvPool: {e}")
        return None
    
    # Get action space
    num_actions = env.action_space.n
    logger.info(f"Action space: {num_actions} actions")
    
    # Create model manager
    model_manager = ModelManager(base_dir=save_dir)
    env_id = get_game_env_id(game_name)  # For saving
    
    # Create Rainbow agent with large batch size
    agent = RainbowAgent(
        state_shape=(4, 84, 84),
        num_actions=num_actions,
        device=device,
        batch_size=batch_size
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
    logger.info("Starting EnvPool training loop...")
    logger.info("-" * 60)
    
    # Reset all environments
    obs, info = env.reset()
    # obs shape: (num_envs, 4, 84, 84) - already frame-stacked!
    states = obs.astype(np.float32) / 255.0  # Normalize to [0, 1]
    
    try:
        while completed_episodes < episodes:
            # Convert states to tensor for batch action selection
            states_tensor = torch.FloatTensor(states).to(device)
            
            # Select actions for ALL environments at once (vectorized)
            with torch.no_grad():
                # Use the network to get Q-values for all states
                if hasattr(agent, 'online_net'):
                    # Get action values
                    q_values = agent.online_net(states_tensor)
                    if len(q_values.shape) == 3:  # Distributional
                        q_values = (q_values * agent.support).sum(dim=2)
                    
                    # Epsilon-greedy
                    if np.random.random() < agent.current_epsilon:
                        actions = np.random.randint(0, num_actions, size=num_envs)
                    else:
                        actions = q_values.argmax(dim=1).cpu().numpy()
                else:
                    actions = np.array([agent.select_action(s, training=True) for s in states])
            
            # Step ALL environments at once (this is where EnvPool shines!)
            next_obs, rewards, terminated, truncated, infos = env.step(actions)
            dones = terminated | truncated
            total_frames += num_envs
            
            # Normalize observations
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
                
                # Reset episode tracking for this env
                episode_rewards[i] = 0
                episode_steps[i] = 0
            
            # Store transitions in batches (only sample a subset to avoid memory issues)
            sample_indices = np.random.choice(num_envs, min(32, num_envs), replace=False)
            for i in sample_indices:
                agent.push_transition(
                    states[i], actions[i], rewards[i], next_states[i], dones[i]
                )
            
            states = next_states
            
            # Learn less frequently but with larger batches
            if total_steps % (num_envs * 4) == 0:
                loss = agent.learn()
            
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
    final_eps = completed_episodes
    fps = total_frames / total_time if total_time > 0 else 0
    
    logger.info("")
    logger.info("=" * 60)
    logger.info("🎉 TRAINING COMPLETE")
    logger.info(f"Game: {game_name}")
    logger.info(f"Episodes: {final_eps}")
    logger.info(f"Total frames: {total_frames:,}")
    logger.info(f"Best reward: {best_reward:.1f}")
    logger.info(f"Final avg (100): {np.mean(recent_rewards):.1f}")
    logger.info(f"Time: {total_time/3600:.2f} hours")
    logger.info(f"Speed: {final_eps / (total_time / 3600):.0f} episodes/hour")
    logger.info(f"Throughput: {fps/1000:.1f}K FPS")
    logger.info("=" * 60)
    
    return {
        "game": game_name,
        "episodes": final_eps,
        "best_reward": best_reward,
        "fps": fps,
        "time_hours": total_time / 3600
    }


def main():
    parser = argparse.ArgumentParser(
        description="EnvPool ULTRA-FAST Atari RL Training",
        epilog=f"Available games: {', '.join(ENVPOOL_GAMES.keys())}"
    )
    
    parser.add_argument("--game", "-g", type=str, required=True)
    parser.add_argument("--episodes", "-e", type=int, default=3000)
    parser.add_argument("--num-envs", "-n", type=int, default=64)
    parser.add_argument("--batch-size", "-b", type=int, default=512)
    parser.add_argument("--checkpoint", "-c", type=str)
    parser.add_argument("--save-dir", "-s", type=str, default="saved_models")
    
    args = parser.parse_args()
    
    train_envpool(
        game_name=args.game,
        episodes=args.episodes,
        num_envs=args.num_envs,
        batch_size=args.batch_size,
        checkpoint_path=args.checkpoint,
        save_dir=args.save_dir
    )


if __name__ == "__main__":
    main()
