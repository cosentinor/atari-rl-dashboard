"""
Game Environments Manager for Atari RL.
Manages available games and environment creation.
"""

import logging
from typing import Dict, List, Any, Optional
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class GameInfo:
    """Information about an Atari game."""
    id: str
    name: str
    display_name: str
    description: str
    action_space_size: int = 0
    observation_shape: tuple = (210, 160, 3)
    is_available: bool = True


# Default game catalog
ATARI_GAMES = [
    GameInfo(
        id='ALE/Pong-v5',
        name='Pong',
        display_name='Pong',
        description='Classic Pong paddle game. Move paddle to hit ball past opponent.',
        action_space_size=6
    ),
    GameInfo(
        id='ALE/Breakout-v5',
        name='Breakout',
        display_name='Breakout',
        description='Break bricks with a bouncing ball and paddle.',
        action_space_size=4
    ),
    GameInfo(
        id='ALE/SpaceInvaders-v5',
        name='SpaceInvaders',
        display_name='Space Invaders',
        description='Defend Earth from descending alien invaders.',
        action_space_size=6
    ),
    GameInfo(
        id='ALE/Asteroids-v5',
        name='Asteroids',
        display_name='Asteroids',
        description='Navigate and shoot asteroids in space.',
        action_space_size=14
    ),
    GameInfo(
        id='ALE/MsPacman-v5',
        name='MsPacman',
        display_name='Ms. Pac-Man',
        description='Navigate mazes eating dots while avoiding ghosts.',
        action_space_size=9
    ),
    GameInfo(
        id='ALE/Boxing-v5',
        name='Boxing',
        display_name='Boxing',
        description='Box against an opponent. Score points by landing punches.',
        action_space_size=18
    ),
    GameInfo(
        id='ALE/Seaquest-v5',
        name='Seaquest',
        display_name='Seaquest',
        description='Underwater action game. Rescue divers and shoot enemies.',
        action_space_size=18
    ),
    GameInfo(
        id='ALE/BeamRider-v5',
        name='BeamRider',
        display_name='Beam Rider',
        description='Sci-fi shooter on a grid of beams.',
        action_space_size=9
    ),
    GameInfo(
        id='ALE/Enduro-v5',
        name='Enduro',
        display_name='Enduro',
        description='Racing game with day/night cycles and weather.',
        action_space_size=9
    ),
    GameInfo(
        id='ALE/Freeway-v5',
        name='Freeway',
        display_name='Freeway',
        description='Guide a chicken across a busy highway.',
        action_space_size=3
    )
]


class GameEnvironments:
    """
    Manages Atari game environments.
    Provides game discovery, environment creation, and configuration.
    """
    
    def __init__(self):
        self.games: Dict[str, GameInfo] = {g.id: g for g in ATARI_GAMES}
        self._check_availability()
        logger.info(f"GameEnvironments initialized with {len(self.games)} games")
    
    def _check_availability(self):
        """Check which games are actually available."""
        try:
            import gymnasium as gym
            from ale_py import ALEInterface
            
            for game_id, game_info in self.games.items():
                try:
                    # Try to create environment briefly
                    env = gym.make(game_id)
                    game_info.action_space_size = env.action_space.n
                    game_info.observation_shape = env.observation_space.shape
                    game_info.is_available = True
                    env.close()
                except Exception as e:
                    logger.warning(f"Game {game_id} not available: {e}")
                    game_info.is_available = False
                    
        except ImportError as e:
            logger.warning(f"gymnasium/ale-py not fully available: {e}")
            # Mark all as unavailable if we can't check
            for game_info in self.games.values():
                game_info.is_available = False
    
    def get_available_games(self) -> List[str]:
        """Get list of available game IDs."""
        return [g.id for g in self.games.values() if g.is_available]
    
    def get_all_games(self) -> List[str]:
        """Get list of all game IDs."""
        return list(self.games.keys())
    
    def get_game_info(self, game_id: str) -> Optional[GameInfo]:
        """Get information about a specific game."""
        return self.games.get(game_id)
    
    def get_all_games_info(self) -> List[Dict[str, Any]]:
        """Get information about all games as dictionaries."""
        return [
            {
                'id': g.id,
                'name': g.name,
                'display_name': g.display_name,
                'description': g.description,
                'action_space_size': g.action_space_size,
                'observation_shape': list(g.observation_shape),
                'is_available': g.is_available
            }
            for g in self.games.values()
        ]
    
    def create_environment(self, game_id: str, render_mode: str = 'rgb_array'):
        """
        Create a game environment.
        
        Args:
            game_id: The game identifier (e.g., 'ALE/Pong-v5')
            render_mode: Render mode ('rgb_array', 'human', None)
            
        Returns:
            gymnasium.Env: The created environment
        """
        if game_id not in self.games:
            raise ValueError(f"Unknown game: {game_id}")
        
        game_info = self.games[game_id]
        if not game_info.is_available:
            raise RuntimeError(f"Game {game_id} is not available")
        
        try:
            import gymnasium as gym
            
            env = gym.make(game_id, render_mode=render_mode)
            logger.info(f"Created environment for {game_id}")
            return env
            
        except Exception as e:
            logger.error(f"Failed to create environment for {game_id}: {e}")
            raise
    
    def create_training_environment(self, game_id: str, frame_stack: int = 4):
        """
        Create a wrapped environment optimized for training.
        
        Args:
            game_id: The game identifier
            frame_stack: Number of frames to stack
            
        Returns:
            Wrapped gymnasium environment
        """
        try:
            import gymnasium as gym
            from gymnasium.wrappers import (
                GrayscaleObservation,
                ResizeObservation,
                FrameStackObservation
            )
            
            # Create base environment
            env = self.create_environment(game_id, render_mode='rgb_array')
            
            # Apply wrappers for training
            env = GrayscaleObservation(env)
            env = ResizeObservation(env, (84, 84))
            env = FrameStackObservation(env, frame_stack)
            
            logger.info(f"Created training environment for {game_id} with {frame_stack} frame stack")
            return env
            
        except ImportError as e:
            logger.error(f"Missing required wrapper: {e}")
            # Return basic environment if wrappers aren't available
            return self.create_environment(game_id)

