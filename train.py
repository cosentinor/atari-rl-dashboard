#!/usr/bin/env python3
"""
Headless Training Script for Atari RL.
Trains Rainbow DQN agents without web UI - optimized for cloud/GPU training.

Usage:
    python train.py --game Pong --episodes 3000
    python train.py --game Breakout --episodes 10000
    python train.py --game all --episodes-per-game 5000
"""

import argparse
import logging
import time
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

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


def train_game(
    game_name: str,
    episodes: int,
    checkpoint_path: str = None,
    save_dir: str = "saved_models"
):
    """
    Train a single game headlessly.
    
    Args:
        game_name: Name of the game (e.g., "Pong", "Breakout")
        episodes: Number of episodes to train
        checkpoint_path: Optional path to checkpoint to resume from
        save_dir: Directory to save models
    """
    logger.info("=" * 60)
    logger.info(f"TRAINING: {game_name}")
    logger.info(f"Episodes: {episodes}")
    logger.info("=" * 60)
    
    # Get environment ID
    env_id = get_game_env_id(game_name)
    logger.info(f"Environment: {env_id}")
    
    # Setup
    game_envs = GameEnvironments()
    model_manager = ModelManager(base_dir=save_dir)
    
    # Create environment
    try:
        env = game_envs.create_environment(env_id)
    except Exception as e:
        logger.error(f"Failed to create environment: {e}")
        return None
    
    num_actions = env.action_space.n
    logger.info(f"Action space: {num_actions} actions")
    
    # Create frame stacker
    frame_stack = FrameStack(num_frames=4, frame_size=(84, 84))
    
    # Create Rainbow agent
    device = get_device()
    logger.info(f"Device: {device}")
    
    agent = RainbowAgent(
        state_shape=(4, 84, 84),
        num_actions=num_actions,
        device=device
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
    
    logger.info("")
    logger.info("Starting training loop...")
    logger.info("-" * 60)
    
    try:
        for episode in range(start_episode + 1, start_episode + episodes + 1):
            agent.episode_count = episode
            
            # Reset environment
            obs, _ = env.reset()
            state = frame_stack.reset(obs)
            
            episode_reward = 0
            episode_steps = 0
            episode_losses = []
            done = False
            
            while not done:
                episode_steps += 1
                total_steps += 1
                
                # Select action
                action = agent.select_action(state, training=True)
                
                # Step environment
                next_obs, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated
                episode_reward += reward
                
                # Process next state
                next_state = frame_stack.push(next_obs)
                
                # Store transition and learn
                agent.push_transition(state, action, reward, next_state, done)
                loss = agent.learn()
                
                if loss is not None:
                    episode_losses.append(loss)
                
                state = next_state
                
                # Time-based autosave
                current_time = time.time()
                if current_time - last_autosave >= AUTOSAVE_INTERVAL_SECONDS:
                    model_manager.save_checkpoint(
                        agent, env_id, episode, episode_reward
                    )
                    last_autosave = current_time
                    logger.info(f"  [Autosave] Episode {episode}")
            
            # Episode complete
            agent.update_epsilon()
            agent.step_count = total_steps
            recent_rewards.append(episode_reward)
            if len(recent_rewards) > 100:
                recent_rewards.pop(0)
            
            avg_reward = np.mean(recent_rewards)
            avg_loss = np.mean(episode_losses) if episode_losses else 0
            
            # Check for best
            is_best = episode_reward > best_reward
            if is_best:
                best_reward = episode_reward
            
            # Log progress
            elapsed = time.time() - training_start
            eps_per_hour = episode / (elapsed / 3600) if elapsed > 0 else 0
            
            log_msg = (
                f"Ep {episode:5d} | "
                f"Reward: {episode_reward:7.1f} | "
                f"Avg100: {avg_reward:7.1f} | "
                f"Best: {best_reward:7.1f} | "
                f"Loss: {avg_loss:.4f} | "
                f"ε: {agent.current_epsilon:.3f} | "
                f"Steps: {episode_steps:4d}"
            )
            
            if is_best:
                log_msg += " ⭐"
            
            logger.info(log_msg)
            
            # Save checkpoints
            # Every 100 episodes or if best
            if episode % 100 == 0 or is_best:
                model_manager.save_checkpoint(
                    agent, env_id, episode, episode_reward, is_best=is_best
                )
                if episode % 100 == 0:
                    logger.info(f"  [Checkpoint] Episode {episode} saved")
    
    except KeyboardInterrupt:
        logger.info("\n" + "=" * 60)
        logger.info("Training interrupted by user")
        logger.info("Saving final checkpoint...")
        model_manager.save_checkpoint(
            agent, env_id, episode, episode_reward
        )
        logger.info("Checkpoint saved!")
    
    finally:
        env.close()
    
    # Training complete
    total_time = time.time() - training_start
    logger.info("")
    logger.info("=" * 60)
    logger.info("TRAINING COMPLETE")
    logger.info(f"Game: {game_name}")
    logger.info(f"Episodes: {episode - start_episode}")
    logger.info(f"Total steps: {total_steps:,}")
    logger.info(f"Best reward: {best_reward:.1f}")
    logger.info(f"Final avg (100): {avg_reward:.1f}")
    logger.info(f"Time: {total_time/3600:.2f} hours")
    logger.info("=" * 60)
    
    # Save final model
    model_manager.save_checkpoint(
        agent, env_id, episode, episode_reward, is_best=False,
        metadata={"training_complete": True, "final_avg_reward": avg_reward}
    )
    
    return {
        "game": game_name,
        "episodes": episode - start_episode,
        "best_reward": best_reward,
        "final_avg_reward": avg_reward,
        "total_steps": total_steps,
        "time_hours": total_time / 3600
    }


def train_all_games(episodes_per_game: int = None, save_dir: str = "saved_models"):
    """Train all available games sequentially."""
    games = list_games()
    results = []
    
    logger.info("=" * 60)
    logger.info("TRAINING ALL GAMES")
    logger.info(f"Games: {', '.join(games)}")
    logger.info("=" * 60)
    
    for game in games:
        episodes = episodes_per_game or get_recommended_episodes(game)
        logger.info(f"\n{'='*60}")
        logger.info(f"Next: {game} ({episodes} episodes)")
        logger.info(f"{'='*60}\n")
        
        result = train_game(game, episodes, save_dir=save_dir)
        if result:
            results.append(result)
    
    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("ALL TRAINING COMPLETE")
    logger.info("=" * 60)
    for r in results:
        logger.info(f"  {r['game']:15s} | Best: {r['best_reward']:7.1f} | Time: {r['time_hours']:.1f}h")
    
    return results


def main():
    parser = argparse.ArgumentParser(
        description="Headless Atari RL Training",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python train.py --game Pong --episodes 3000
  python train.py --game Breakout --episodes 10000
  python train.py --game all
  python train.py --game all --episodes-per-game 5000
  python train.py --list-games

Available games:
  """ + ", ".join(list_games())
    )
    
    parser.add_argument(
        "--game", "-g",
        type=str,
        help="Game to train (e.g., Pong, Breakout, or 'all')"
    )
    
    parser.add_argument(
        "--episodes", "-e",
        type=int,
        help="Number of episodes to train"
    )
    
    parser.add_argument(
        "--episodes-per-game",
        type=int,
        help="Episodes per game when training all (default: use recommended)"
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
    
    # List games and exit
    if args.list_games:
        print("\nAvailable games:")
        print("-" * 50)
        for name, preset in GAME_PRESETS.items():
            print(f"  {name:15s} | {preset['difficulty']:10s} | {preset['recommended_episodes']:6d} eps | {preset['description']}")
        print()
        return
    
    # Validate arguments
    if not args.game:
        parser.error("--game is required (use --list-games to see options)")
    
    # Train
    if args.game.lower() == "all":
        train_all_games(
            episodes_per_game=args.episodes_per_game,
            save_dir=args.save_dir
        )
    else:
        episodes = args.episodes or get_recommended_episodes(args.game)
        train_game(
            game_name=args.game,
            episodes=episodes,
            checkpoint_path=args.checkpoint,
            save_dir=args.save_dir
        )


if __name__ == "__main__":
    main()

