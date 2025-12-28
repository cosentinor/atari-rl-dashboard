"""
Configuration for Atari RL Training.
Centralized settings for games, training, and autosave.
"""

# ============== Autosave Settings ==============

AUTOSAVE_INTERVAL_SECONDS = 90  # Save checkpoint every 90 seconds
AUTOSAVE_INTERVAL_EPISODES = 100  # Also save every N episodes (backup)

# ============== Game Presets ==============
# Recommended training episodes and difficulty ratings

GAME_PRESETS = {
    "Pong": {
        "env_id": "ALE/Pong-v5",
        "recommended_episodes": 3000,
        "difficulty": "easy",
        "description": "Simple paddle game - great for testing"
    },
    "Freeway": {
        "env_id": "ALE/Freeway-v5",
        "recommended_episodes": 3000,
        "difficulty": "easy",
        "description": "Cross the road - simple reward structure"
    },
    "Breakout": {
        "env_id": "ALE/Breakout-v5",
        "recommended_episodes": 10000,
        "difficulty": "medium",
        "description": "Classic brick breaker"
    },
    "Boxing": {
        "env_id": "ALE/Boxing-v5",
        "recommended_episodes": 10000,
        "difficulty": "medium",
        "description": "Two-player boxing"
    },
    "SpaceInvaders": {
        "env_id": "ALE/SpaceInvaders-v5",
        "recommended_episodes": 15000,
        "difficulty": "medium",
        "description": "Shoot the aliens"
    },
    "Enduro": {
        "env_id": "ALE/Enduro-v5",
        "recommended_episodes": 15000,
        "difficulty": "medium",
        "description": "Racing endurance game"
    },
    "BeamRider": {
        "env_id": "ALE/BeamRider-v5",
        "recommended_episodes": 20000,
        "difficulty": "hard",
        "description": "Space shooter on rails"
    },
    "Seaquest": {
        "env_id": "ALE/Seaquest-v5",
        "recommended_episodes": 25000,
        "difficulty": "hard",
        "description": "Underwater rescue mission"
    },
    "Asteroids": {
        "env_id": "ALE/Asteroids-v5",
        "recommended_episodes": 25000,
        "difficulty": "hard",
        "description": "360-degree space shooter"
    },
    "MsPacman": {
        "env_id": "ALE/MsPacman-v5",
        "recommended_episodes": 30000,
        "difficulty": "very_hard",
        "description": "Maze navigation with ghosts"
    }
}

# ============== Rainbow DQN Hyperparameters ==============

RAINBOW_HYPERPARAMS = {
    # Network
    "hidden_size": 512,
    "noisy_std": 0.5,
    "atom_size": 51,
    "v_min": -10.0,
    "v_max": 10.0,
    
    # Training
    "batch_size": 32,
    "learning_rate": 6.25e-5,
    "gamma": 0.99,
    "n_step": 3,
    
    # Replay buffer
    "buffer_size": 100000,  # 100k for faster startup, increase for longer training
    "min_buffer_size": 10000,  # Start learning after this many transitions
    "alpha": 0.5,  # PER priority exponent
    "beta_start": 0.4,
    "beta_frames": 100000,
    
    # Target network
    "target_update_freq": 1000,
    
    # Exploration (epsilon-greedy fallback for early training)
    "epsilon_start": 1.0,
    "epsilon_final": 0.01,
    "epsilon_decay_episodes": 50
}

# ============== Training Presets ==============

TRAINING_PRESETS = {
    "quick_test": {
        "episodes": 100,
        "description": "Quick test to verify setup works"
    },
    "poc": {
        "episodes": 3000,
        "description": "Proof of concept - good for Pong"
    },
    "standard": {
        "episodes": 10000,
        "description": "Standard training for medium games"
    },
    "extended": {
        "episodes": 30000,
        "description": "Extended training for hard games"
    },
    "full": {
        "episodes": 50000,
        "description": "Full training run"
    }
}

# ============== Utility Functions ==============

def get_game_env_id(game_name: str) -> str:
    """Get environment ID from game name."""
    # Handle various input formats
    if game_name.startswith("ALE/"):
        return game_name
    
    # Check presets
    if game_name in GAME_PRESETS:
        return GAME_PRESETS[game_name]["env_id"]
    
    # Try to construct it
    clean_name = game_name.replace(" ", "").replace("-", "").replace("_", "")
    for name, preset in GAME_PRESETS.items():
        if clean_name.lower() == name.lower():
            return preset["env_id"]
    
    # Default construction
    return f"ALE/{game_name}-v5"


def get_recommended_episodes(game_name: str) -> int:
    """Get recommended episodes for a game."""
    if game_name in GAME_PRESETS:
        return GAME_PRESETS[game_name]["recommended_episodes"]
    return 10000  # Default


def list_games() -> list:
    """List all available games."""
    return list(GAME_PRESETS.keys())


def get_all_game_env_ids() -> dict:
    """Get all game environment IDs."""
    return {name: preset["env_id"] for name, preset in GAME_PRESETS.items()}

