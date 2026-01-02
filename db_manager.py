"""
Database Manager for Atari RL Training.
Handles SQLite storage for training history and metrics.
"""

import sqlite3
import json
import logging
from typing import Dict, List, Optional, Tuple
from pathlib import Path
from contextlib import contextmanager

logger = logging.getLogger(__name__)


class TrainingDatabase:
    """
    SQLite database for storing training history and metrics.
    
    Tables:
    - sessions: Training session metadata
    - episodes: Per-episode metrics
    - step_metrics: Sampled per-step metrics
    - game_training_stats: Aggregated training time per game
    """
    
    def __init__(self, db_path: str = "data/rl_training.db"):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        
        self._init_db()
        logger.info(f"TrainingDatabase initialized at {self.db_path}")
    
    @contextmanager
    def _get_connection(self):
        """Context manager for database connections."""
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        try:
            yield conn
        finally:
            conn.close()
    
    def _init_db(self):
        """Initialize database schema."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            
            # Training sessions table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS sessions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    game_id TEXT NOT NULL,
                    started_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    ended_at TIMESTAMP,
                    total_episodes INTEGER DEFAULT 0,
                    total_steps INTEGER DEFAULT 0,
                    best_reward REAL DEFAULT 0,
                    avg_reward REAL DEFAULT 0,
                    device TEXT,
                    hyperparameters TEXT,
                    status TEXT DEFAULT 'running'
                )
            """)
            
            # Episode metrics table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS episodes (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id INTEGER NOT NULL,
                    episode_num INTEGER NOT NULL,
                    reward REAL NOT NULL,
                    steps INTEGER NOT NULL,
                    loss REAL,
                    q_value_mean REAL,
                    q_value_max REAL,
                    epsilon REAL,
                    duration_ms INTEGER,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (session_id) REFERENCES sessions(id)
                )
            """)
            
            # Per-step metrics (sampled)
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS step_metrics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id INTEGER NOT NULL,
                    episode_num INTEGER NOT NULL,
                    step_num INTEGER NOT NULL,
                    loss REAL,
                    q_values TEXT,
                    action INTEGER,
                    reward REAL,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (session_id) REFERENCES sessions(id)
                )
            """)
            
            # Aggregated training stats per game
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS game_training_stats (
                    game_id TEXT PRIMARY KEY,
                    total_training_seconds INTEGER DEFAULT 0,
                    total_sessions INTEGER DEFAULT 0,
                    total_episodes INTEGER DEFAULT 0,
                    total_steps INTEGER DEFAULT 0,
                    last_trained_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Create indices for faster queries
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_episodes_session 
                ON episodes(session_id)
            """)
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_step_metrics_session 
                ON step_metrics(session_id)
            """)
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_sessions_game 
                ON sessions(game_id)
            """)
            
            conn.commit()
    
    # ============== Session Management ==============
    
    def create_session(
        self,
        game_id: str,
        device: str = "cpu",
        hyperparameters: Optional[Dict] = None
    ) -> int:
        """Create a new training session."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO sessions (game_id, device, hyperparameters, status)
                VALUES (?, ?, ?, 'running')
            """, (game_id, device, json.dumps(hyperparameters or {})))
            conn.commit()
            session_id = cursor.lastrowid
            logger.info(f"Created session {session_id} for {game_id}")
            return session_id
    
    def end_session(self, session_id: int):
        """Mark a session as ended."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            
            # Calculate final stats
            cursor.execute("""
                SELECT 
                    COUNT(*) as total_episodes,
                    SUM(steps) as total_steps,
                    MAX(reward) as best_reward,
                    AVG(reward) as avg_reward
                FROM episodes WHERE session_id = ?
            """, (session_id,))
            
            stats = cursor.fetchone()
            
            cursor.execute("""
                UPDATE sessions SET
                    ended_at = CURRENT_TIMESTAMP,
                    total_episodes = ?,
                    total_steps = ?,
                    best_reward = ?,
                    avg_reward = ?,
                    status = 'completed'
                WHERE id = ?
            """, (
                stats['total_episodes'] or 0,
                stats['total_steps'] or 0,
                stats['best_reward'] or 0,
                stats['avg_reward'] or 0,
                session_id
            ))
            conn.commit()
            logger.info(f"Ended session {session_id}")
    
    def get_session(self, session_id: int) -> Optional[Dict]:
        """Get session details."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM sessions WHERE id = ?", (session_id,))
            row = cursor.fetchone()
            if row:
                return dict(row)
            return None
    
    def get_active_sessions(self) -> List[Dict]:
        """Get all running sessions."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM sessions WHERE status = 'running'")
            return [dict(row) for row in cursor.fetchall()]
    
    # ============== Episode Logging ==============
    
    def log_episode(
        self,
        session_id: int,
        episode_num: int,
        reward: float,
        steps: int,
        loss: Optional[float] = None,
        q_value_mean: Optional[float] = None,
        q_value_max: Optional[float] = None,
        epsilon: Optional[float] = None,
        duration_ms: Optional[int] = None
    ):
        """Log episode metrics."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO episodes 
                (session_id, episode_num, reward, steps, loss, 
                 q_value_mean, q_value_max, epsilon, duration_ms)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                session_id, episode_num, reward, steps,
                loss, q_value_mean, q_value_max, epsilon, duration_ms
            ))
            
            # Update session best reward if needed
            cursor.execute("""
                UPDATE sessions SET best_reward = MAX(best_reward, ?)
                WHERE id = ?
            """, (reward, session_id))
            
            conn.commit()
    
    def log_step_metrics(
        self,
        session_id: int,
        episode_num: int,
        step_num: int,
        loss: Optional[float] = None,
        q_values: Optional[List[float]] = None,
        action: Optional[int] = None,
        reward: Optional[float] = None
    ):
        """Log step-level metrics (sampled)."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO step_metrics 
                (session_id, episode_num, step_num, loss, q_values, action, reward)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                session_id, episode_num, step_num,
                loss, json.dumps(q_values) if q_values else None,
                action, reward
            ))
            conn.commit()
    
    # ============== Data Retrieval ==============
    
    def get_episode_history(
        self,
        session_id: int,
        limit: int = 1000
    ) -> List[Dict]:
        """Get episode history for a session."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT * FROM episodes 
                WHERE session_id = ?
                ORDER BY episode_num DESC
                LIMIT ?
            """, (session_id, limit))
            return [dict(row) for row in cursor.fetchall()]
    
    def get_recent_episodes(
        self,
        session_id: int,
        n: int = 100
    ) -> List[Dict]:
        """Get N most recent episodes."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT * FROM episodes 
                WHERE session_id = ?
                ORDER BY episode_num DESC
                LIMIT ?
            """, (session_id, n))
            results = [dict(row) for row in cursor.fetchall()]
            results.reverse()  # Return in chronological order
            return results
    
    def get_step_metrics(
        self,
        session_id: int,
        limit: int = 1000
    ) -> List[Dict]:
        """Get step metrics for a session."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT * FROM step_metrics 
                WHERE session_id = ?
                ORDER BY step_num DESC
                LIMIT ?
            """, (session_id, limit))
            results = [dict(row) for row in cursor.fetchall()]
            results.reverse()
            return results
    
    def get_reward_stats(self, session_id: int) -> Dict:
        """Get reward statistics for a session."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT 
                    COUNT(*) as total_episodes,
                    MIN(reward) as min_reward,
                    MAX(reward) as max_reward,
                    AVG(reward) as avg_reward,
                    SUM(steps) as total_steps
                FROM episodes WHERE session_id = ?
            """, (session_id,))
            row = cursor.fetchone()
            return dict(row) if row else {}
    
    def get_reward_distribution(
        self,
        session_id: int,
        bins: int = 20
    ) -> Tuple[List[float], List[int]]:
        """Get reward distribution histogram data."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT reward FROM episodes WHERE session_id = ?
            """, (session_id,))
            rewards = [row['reward'] for row in cursor.fetchall()]
            
            if not rewards:
                return [], []
            
            import numpy as np
            counts, bin_edges = np.histogram(rewards, bins=bins)
            bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
            
            return bin_centers.tolist(), counts.tolist()
    
    def get_action_distribution(self, session_id: int) -> Dict[int, int]:
        """Get action distribution for a session."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT action, COUNT(*) as count 
                FROM step_metrics 
                WHERE session_id = ? AND action IS NOT NULL
                GROUP BY action
            """, (session_id,))
            return {row['action']: row['count'] for row in cursor.fetchall()}
    
    # ============== Training Leaderboard ==============
    
    def record_training_activity(
        self,
        game_id: str,
        duration_seconds: int = 0,
        sessions: int = 0,
        episodes: int = 0,
        steps: int = 0
    ):
        """Record aggregated training activity for a game."""
        if not game_id:
            return
        
        duration_seconds = int(max(duration_seconds, 0))
        sessions = int(max(sessions, 0))
        episodes = int(max(episodes, 0))
        steps = int(max(steps, 0))
        
        if duration_seconds == 0 and sessions == 0 and episodes == 0 and steps == 0:
            return
        
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO game_training_stats (
                    game_id,
                    total_training_seconds,
                    total_sessions,
                    total_episodes,
                    total_steps,
                    last_trained_at
                )
                VALUES (?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
                ON CONFLICT(game_id) DO UPDATE SET
                    total_training_seconds = total_training_seconds + excluded.total_training_seconds,
                    total_sessions = total_sessions + excluded.total_sessions,
                    total_episodes = total_episodes + excluded.total_episodes,
                    total_steps = total_steps + excluded.total_steps,
                    last_trained_at = CURRENT_TIMESTAMP
            """, (game_id, duration_seconds, sessions, episodes, steps))
            conn.commit()
    
    def get_leaderboard(self, game_id: Optional[str] = None, limit: int = 10) -> List[Dict]:
        """Get leaderboard for games trained the most (by total training time)."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            
            if game_id:
                cursor.execute("""
                    SELECT 
                        game_id,
                        total_training_seconds,
                        total_sessions,
                        total_episodes,
                        total_steps,
                        last_trained_at
                    FROM game_training_stats
                    WHERE game_id = ?
                    ORDER BY total_training_seconds DESC
                    LIMIT ?
                """, (game_id, limit))
            else:
                cursor.execute("""
                    SELECT 
                        game_id,
                        total_training_seconds,
                        total_sessions,
                        total_episodes,
                        total_steps,
                        last_trained_at
                    FROM game_training_stats
                    ORDER BY total_training_seconds DESC
                    LIMIT ?
                """, (limit,))
            
            return [dict(row) for row in cursor.fetchall()]
    
    # ============== Cleanup ==============
    
    def cleanup_old_data(self, days: int = 30):
        """Remove data older than specified days."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            
            # Delete old step metrics first (most data)
            cursor.execute("""
                DELETE FROM step_metrics 
                WHERE session_id IN (
                    SELECT id FROM sessions 
                    WHERE ended_at < datetime('now', ?)
                )
            """, (f'-{days} days',))
            
            # Delete old episodes
            cursor.execute("""
                DELETE FROM episodes 
                WHERE session_id IN (
                    SELECT id FROM sessions 
                    WHERE ended_at < datetime('now', ?)
                )
            """, (f'-{days} days',))
            
            # Delete old sessions
            cursor.execute("""
                DELETE FROM sessions 
                WHERE ended_at < datetime('now', ?)
            """, (f'-{days} days',))
            
            conn.commit()
            logger.info(f"Cleaned up data older than {days} days")
    
    def get_database_stats(self) -> Dict:
        """Get database statistics."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            
            cursor.execute("SELECT COUNT(*) FROM sessions")
            num_sessions = cursor.fetchone()[0]
            
            cursor.execute("SELECT COUNT(*) FROM episodes")
            num_episodes = cursor.fetchone()[0]
            
            cursor.execute("SELECT COUNT(*) FROM step_metrics")
            num_steps = cursor.fetchone()[0]

            cursor.execute("SELECT COUNT(*) FROM game_training_stats")
            num_games = cursor.fetchone()[0]
            
            # Get file size
            file_size = self.db_path.stat().st_size if self.db_path.exists() else 0
            
            return {
                "sessions": num_sessions,
                "episodes": num_episodes,
                "step_metrics": num_steps,
                "games_tracked": num_games,
                "file_size_mb": round(file_size / (1024 * 1024), 2)
            }
    
