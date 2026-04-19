#!/usr/bin/env python3
"""
FAST Headless Training Script for Atari RL.
Uses vectorized environments for GPU-optimized training.

This version runs MUCH faster on powerful GPUs like A100/H100.
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

# Import project modules
from rainbow_agent import RainbowAgent, FrameStack, get_device
from model_manager import ModelManager
from game_environments import GameEnvironments
from config import (
    GAME_PRESETS, 
    AUTOSAVE_INTERVAL_SECONDS,
    get_game_env_id,
    get_recommended_episodes,
    list_games
)

# Try to import gymnasium's vectorized environments
try:
    import gymnasium as gym
    from gymnasium.vector import AsyncVectorEnv, SyncVectorEnv
    # IMPORTANT: Import ale_py to register ALE environments
    import ale_py
    gym.register_envs(ale_py)
    HAS_VECTOR_ENV = True
except ImportError as e:
    HAS_VECTOR_ENV = False
    logger.warning(f"Vectorized environments not available: {e}")


def make_env(env_id):
    """Factory function to create environment."""
    def _init():
        # Import ale_py in each subprocess to register environments
        import gymnasium as gym
        import ale_py
        gym.register_envs(ale_py)
        env = gym.make(env_id, render_mode=None)
        return env
    return _init


def train_game_fast(
    game_name: str,
    episodes: int,
    num_envs: int = 8,
    batch_size: int = 256,
    checkpoint_path: str = None,
    save_dir: str = "saved_models"
):
    """
    Train a single game headlessly with parallel environments.
    
    Args:
        game_name: Name of the game
        episodes: Number of episodes to train
        num_envs: Number of parallel environments (default: 8)
        batch_size: Training batch size (default: 256 for GPU)
        checkpoint_path: Optional path to checkpoint to resume from
        save_dir: Directory to save models
    """
    logger.info("=" * 60)
    logger.info(f"🚀 FAST TRAINING: {game_name}")
    logger.info(f"Episodes: {episodes}")
    logger.info(f"Parallel Environments: {num_envs}")
    logger.info(f"Batch Size: {batch_size}")
    logger.info("=" * 60)
    
    # Get environment ID
    env_id = get_game_env_id(game_name)
    logger.info(f"Environment: {env_id}")
    
    # Setup
    device = get_device()
    logger.info(f"Device: {device}")
    
    # Check GPU memory
    if torch.cuda.is_available():
        gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1e9
        logger.info(f"GPU Memory: {gpu_mem:.1f} GB")
        
        # Adjust batch size based on GPU memory
        if gpu_mem >= 40:  # A100 80GB or similar
            batch_size = max(batch_size, 512)
            num_envs = max(num_envs, 16)
            logger.info(f"A100 detected! Scaling up: batch_size={batch_size}, num_envs={num_envs}")
    
    model_manager = ModelManager(base_dir=save_dir)
    
    # Create vectorized environments if available
    if HAS_VECTOR_ENV and num_envs > 1:
        logger.info(f"Creating {num_envs} parallel environments...")
        try:
            envs = AsyncVectorEnv([make_env(env_id) for _ in range(num_envs)])
            is_vectorized = True
            logger.info("✅ Async vectorized environments created")
        except Exception as e:
            logger.warning(f"Async envs failed: {e}, falling back to sync")
            try:
                envs = SyncVectorEnv([make_env(env_id) for _ in range(num_envs)])
                is_vectorized = True
                logger.info("✅ Sync vectorized environments created")
            except Exception as e2:
                logger.warning(f"Vectorized envs failed: {e2}, using single env")
                game_envs = GameEnvironments()
                envs = game_envs.create_environment(env_id)
                is_vectorized = False
    else:
        game_envs = GameEnvironments()
        envs = game_envs.create_environment(env_id)
        is_vectorized = False
    
    # Get action space
    if is_vectorized:
        num_actions = envs.single_action_space.n
    else:
        num_actions = envs.action_space.n
    logger.info(f"Action space: {num_actions} actions")
    
    # Create frame stackers (one per env if vectorized)
    if is_vectorized:
        frame_stacks = [FrameStack(num_frames=4, frame_size=(84, 84)) for _ in range(num_envs)]
    else:
        frame_stack = FrameStack(num_frames=4, frame_size=(84, 84))
    
    # Create Rainbow agent with larger batch size
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
            logger.info("Starting fresh training")
    
    # Training metrics
    best_reward = float('-inf')
    recent_rewards = []
    total_steps = agent.step_count
    training_start = time.time()
    last_autosave = time.time()
    
    # Episode tracking for vectorized envs
    if is_vectorized:
        episode_rewards = np.zeros(num_envs)
        episode_steps = np.zeros(num_envs, dtype=int)
        completed_episodes = 0
    
    logger.info("")
    logger.info("Starting training loop...")
    logger.info("-" * 60)
    
    try:
        if is_vectorized:
            # Vectorized training loop
            obs, _ = envs.reset()
            states = [fs.reset(o) for fs, o in zip(frame_stacks, obs)]
            states = np.array(states)
            
            while completed_episodes < episodes:
                # Select actions for all environments
                actions = []
                for state in states:
                    action = agent.select_action(state, training=True)
                    actions.append(action)
                actions = np.array(actions)
                
                # Step all environments
                next_obs, rewards, terminated, truncated, infos = envs.step(actions)
                dones = terminated | truncated
                
                # Process transitions
                next_states = []
                for i, (fs, next_o) in enumerate(zip(frame_stacks, next_obs)):
                    next_state = fs.push(next_o)
                    next_states.append(next_state)
                    
                    # Store transition
                    agent.push_transition(states[i], actions[i], rewards[i], next_state, dones[i])
                    
                    episode_rewards[i] += rewards[i]
                    episode_steps[i] += 1
                    total_steps += 1
                    
                    # Episode complete
                    if dones[i]:
                        completed_episodes += 1
                        agent.episode_count = start_episode + completed_episodes
                        agent.update_epsilon()
                        
                        episode_reward = episode_rewards[i]
                        recent_rewards.append(episode_reward)
                        if len(recent_rewards) > 100:
                            recent_rewards.pop(0)
                        
                        avg_reward = np.mean(recent_rewards)
                        is_best = episode_reward > best_reward
                        if is_best:
                            best_reward = episode_reward
                        
                        # Log progress
                        if completed_episodes % 10 == 0 or is_best:
                            elapsed = time.time() - training_start
                            eps_per_hour = completed_episodes / (elapsed / 3600) if elapsed > 0 else 0
                            
                            log_msg = (
                                f"Ep {start_episode + completed_episodes:5d} | "
                                f"Reward: {episode_reward:7.1f} | "
                                f"Avg100: {avg_reward:7.1f} | "
                                f"Best: {best_reward:7.1f} | "
                                f"ε: {agent.current_epsilon:.3f} | "
                                f"Eps/hr: {eps_per_hour:.0f}"
                            )
                            
                            if is_best:
                                log_msg += " ⭐"
                            
                            logger.info(log_msg)
                        
                        # Save checkpoints
                        if completed_episodes % 100 == 0 or is_best:
                            model_manager.save_checkpoint(
                                agent, env_id, start_episode + completed_episodes, 
                                episode_reward, is_best=is_best
                            )
                            if completed_episodes % 100 == 0:
                                logger.info(f"  [Checkpoint] Episode {start_episode + completed_episodes} saved")
                        
                        # Reset this environment's tracking
                        episode_rewards[i] = 0
                        episode_steps[i] = 0
                        next_states[i] = frame_stacks[i].reset(next_obs[i])
                
                next_states = np.array(next_states)
                states = next_states
                
                # Learn from batch
                loss = agent.learn()
                
                # Time-based autosave
                current_time = time.time()
                if current_time - last_autosave >= AUTOSAVE_INTERVAL_SECONDS:
                    model_manager.save_checkpoint(
                        agent, env_id, start_episode + completed_episodes, 
                        recent_rewards[-1] if recent_rewards else 0
                    )
                    last_autosave = current_time
                    logger.info(f"  [Autosave] Episode {start_episode + completed_episodes}")
        else:
            # Standard single-env training loop (fallback)
            for episode in range(start_episode + 1, start_episode + episodes + 1):
                agent.episode_count = episode
                
                obs, _ = envs.reset()
                state = frame_stack.reset(obs)
                
                episode_reward = 0
                episode_steps = 0
                episode_losses = []
                done = False
                
                while not done:
                    episode_steps += 1
                    total_steps += 1
                    
                    action = agent.select_action(state, training=True)
                    next_obs, reward, terminated, truncated, _ = envs.step(action)
                    done = terminated or truncated
                    episode_reward += reward
                    
                    next_state = frame_stack.push(next_obs)
                    agent.push_transition(state, action, reward, next_state, done)
                    loss = agent.learn()
                    
                    if loss is not None:
                        episode_losses.append(loss)
                    
                    state = next_state
                    
                    current_time = time.time()
                    if current_time - last_autosave >= AUTOSAVE_INTERVAL_SECONDS:
                        model_manager.save_checkpoint(agent, env_id, episode, episode_reward)
                        last_autosave = current_time
                        logger.info(f"  [Autosave] Episode {episode}")
                
                agent.update_epsilon()
                agent.step_count = total_steps
                recent_rewards.append(episode_reward)
                if len(recent_rewards) > 100:
                    recent_rewards.pop(0)
                
                avg_reward = np.mean(recent_rewards)
                is_best = episode_reward > best_reward
                if is_best:
                    best_reward = episode_reward
                
                if episode % 10 == 0 or is_best:
                    elapsed = time.time() - training_start
                    eps_per_hour = episode / (elapsed / 3600) if elapsed > 0 else 0
                    
                    log_msg = (
                        f"Ep {episode:5d} | "
                        f"Reward: {episode_reward:7.1f} | "
                        f"Avg100: {avg_reward:7.1f} | "
                        f"Best: {best_reward:7.1f} | "
                        f"ε: {agent.current_epsilon:.3f} | "
                        f"Eps/hr: {eps_per_hour:.0f}"
                    )
                    
                    if is_best:
                        log_msg += " ⭐"
                    
                    logger.info(log_msg)
                
                if episode % 100 == 0 or is_best:
                    model_manager.save_checkpoint(
                        agent, env_id, episode, episode_reward, is_best=is_best
                    )
    
    except KeyboardInterrupt:
        logger.info("\n" + "=" * 60)
        logger.info("Training interrupted by user")
        logger.info("Saving final checkpoint...")
        final_ep = start_episode + (completed_episodes if is_vectorized else episode - start_episode)
        model_manager.save_checkpoint(agent, env_id, final_ep, recent_rewards[-1] if recent_rewards else 0)
        logger.info("Checkpoint saved!")
    
    finally:
        envs.close()
    
    # Training complete
    total_time = time.time() - training_start
    final_episodes = completed_episodes if is_vectorized else (episode - start_episode)
    
    logger.info("")
    logger.info("=" * 60)
    logger.info("🎉 TRAINING COMPLETE")
    logger.info(f"Game: {game_name}")
    logger.info(f"Episodes: {final_episodes}")
    logger.info(f"Total steps: {total_steps:,}")
    logger.info(f"Best reward: {best_reward:.1f}")
    logger.info(f"Final avg (100): {np.mean(recent_rewards):.1f}")
    logger.info(f"Time: {total_time/3600:.2f} hours")
    logger.info(f"Speed: {final_episodes / (total_time / 3600):.0f} episodes/hour")
    logger.info("=" * 60)
    
    # Save final model
    model_manager.save_checkpoint(
        agent, env_id, start_episode + final_episodes, 
        recent_rewards[-1] if recent_rewards else 0, is_best=False,
        metadata={"training_complete": True, "final_avg_reward": np.mean(recent_rewards)}
    )
    
    return {
        "game": game_name,
        "episodes": final_episodes,
        "best_reward": best_reward,
        "final_avg_reward": np.mean(recent_rewards),
        "total_steps": total_steps,
        "time_hours": total_time / 3600,
        "episodes_per_hour": final_episodes / (total_time / 3600)
    }


def main():
    parser = argparse.ArgumentParser(
        description="FAST Headless Atari RL Training (GPU Optimized)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python train_fast.py --game Pong --episodes 3000
  python train_fast.py --game Breakout --episodes 5000 --num-envs 16
  python train_fast.py --game Pong --episodes 3000 --batch-size 512

Available games:
  """ + ", ".join(list_games())
    )
    
    parser.add_argument(
        "--game", "-g",
        type=str,
        required=True,
        help="Game to train (e.g., Pong, Breakout)"
    )
    
    parser.add_argument(
        "--episodes", "-e",
        type=int,
        help="Number of episodes to train"
    )
    
    parser.add_argument(
        "--num-envs", "-n",
        type=int,
        default=8,
        help="Number of parallel environments (default: 8, auto-scales for A100)"
    )
    
    parser.add_argument(
        "--batch-size", "-b",
        type=int,
        default=256,
        help="Training batch size (default: 256, auto-scales for A100)"
    )
    
    parser.add_argument(
        "--checkpoint", "-c",
        type=str,
        help="Path to checkpoint to resume from"
    )
    
    parser.add_argument(
        "--save-dir", "-s",
        type=str,
        default="saved_models",
        help="Directory to save models (default: saved_models)"
    )
    
    parser.add_argument(
        "--list-games",
        action="store_true",
        help="List all available games and exit"
    )
    
    args = parser.parse_args()
    
    if args.list_games:
        print("\nAvailable games:")
        print("-" * 50)
        for name, preset in GAME_PRESETS.items():
            print(f"  {name:15s} | {preset['difficulty']:10s} | {preset['recommended_episodes']:6d} eps")
        print()
        return
    
    episodes = args.episodes or get_recommended_episodes(args.game)
    
    train_game_fast(
        game_name=args.game,
        episodes=episodes,
        num_envs=args.num_envs,
        batch_size=args.batch_size,
        checkpoint_path=args.checkpoint,
        save_dir=args.save_dir
    )


if __name__ == "__main__":
    main()

