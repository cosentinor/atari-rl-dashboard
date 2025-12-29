"""
Database Manager for Atari RL Training.
Handles SQLite storage for training history and metrics.
"""

import sqlite3
import json
import logging
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple
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
                    status TEXT DEFAULT 'running',
                    visitor_id INTEGER,
                    FOREIGN KEY (visitor_id) REFERENCES visitors(id)
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
            
            # Visitors table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS visitors (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    email TEXT UNIQUE,
                    visitor_uuid TEXT UNIQUE NOT NULL,
                    first_visit TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    last_visit TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    total_sessions INTEGER DEFAULT 0,
                    preferred_mode TEXT,
                    opt_in_marketing BOOLEAN DEFAULT 0
                )
            """)
            
            # Analytics events table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS analytics_events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    visitor_id INTEGER,
                    event_type TEXT NOT NULL,
                    event_data TEXT,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    session_id INTEGER,
                    FOREIGN KEY (visitor_id) REFERENCES visitors(id),
                    FOREIGN KEY (session_id) REFERENCES sessions(id)
                )
            """)
            
            # Feedback table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS feedback (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    visitor_id INTEGER,
                    category TEXT,
                    rating INTEGER,
                    message TEXT,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (visitor_id) REFERENCES visitors(id)
                )
            """)
            
            # Challenges table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS challenges (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    game_id TEXT NOT NULL,
                    challenge_type TEXT NOT NULL,
                    target_value REAL NOT NULL,
                    description TEXT,
                    start_date DATE NOT NULL,
                    end_date DATE NOT NULL,
                    is_active BOOLEAN DEFAULT 1
                )
            """)
            
            # User challenge progress table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS user_challenges (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    visitor_id INTEGER NOT NULL,
                    challenge_id INTEGER NOT NULL,
                    progress REAL DEFAULT 0,
                    completed BOOLEAN DEFAULT 0,
                    completed_at TIMESTAMP,
                    FOREIGN KEY (visitor_id) REFERENCES visitors(id),
                    FOREIGN KEY (challenge_id) REFERENCES challenges(id),
                    UNIQUE(visitor_id, challenge_id)
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
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_visitors_uuid 
                ON visitors(visitor_uuid)
            """)
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_visitors_email 
                ON visitors(email)
            """)
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_analytics_visitor 
                ON analytics_events(visitor_id)
            """)
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_analytics_type 
                ON analytics_events(event_type)
            """)
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_feedback_visitor 
                ON feedback(visitor_id)
            """)
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_challenges_active 
                ON challenges(is_active, end_date)
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
    
    # ============== Leaderboard ==============
    
    def get_leaderboard(self, game_id: Optional[str] = None, limit: int = 10) -> List[Dict]:
        """Get top scores leaderboard."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            
            if game_id:
                cursor.execute("""
                    SELECT 
                        s.id as session_id,
                        s.game_id,
                        s.started_at,
                        s.total_episodes,
                        s.best_reward,
                        s.avg_reward
                    FROM sessions s
                    WHERE s.game_id = ? AND s.status = 'completed'
                    ORDER BY s.best_reward DESC
                    LIMIT ?
                """, (game_id, limit))
            else:
                cursor.execute("""
                    SELECT 
                        s.id as session_id,
                        s.game_id,
                        s.started_at,
                        s.total_episodes,
                        s.best_reward,
                        s.avg_reward
                    FROM sessions s
                    WHERE s.status = 'completed'
                    ORDER BY s.best_reward DESC
                    LIMIT ?
                """, (limit,))
            
            return [dict(row) for row in cursor.fetchall()]
    
    def get_game_best_scores(self) -> Dict[str, float]:
        """Get best score for each game."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT game_id, MAX(best_reward) as best_score
                FROM sessions
                WHERE status = 'completed'
                GROUP BY game_id
            """)
            return {row['game_id']: row['best_score'] for row in cursor.fetchall()}
    
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
            
            # Get file size
            file_size = self.db_path.stat().st_size if self.db_path.exists() else 0
            
            return {
                "sessions": num_sessions,
                "episodes": num_episodes,
                "step_metrics": num_steps,
                "file_size_mb": round(file_size / (1024 * 1024), 2)
            }
    
    # ============== Visitor Management ==============
    
    def create_or_update_visitor(
        self,
        visitor_uuid: str,
        email: Optional[str] = None,
        preferred_mode: Optional[str] = None,
        opt_in_marketing: bool = False
    ) -> int:
        """Create or update a visitor record."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            
            # Check if visitor exists
            cursor.execute("SELECT id FROM visitors WHERE visitor_uuid = ?", (visitor_uuid,))
            existing = cursor.fetchone()
            
            if existing:
                # Update existing visitor
                visitor_id = existing['id']
                updates = ["last_visit = CURRENT_TIMESTAMP"]
                params = []
                
                if email:
                    updates.append("email = ?")
                    params.append(email)
                if preferred_mode:
                    updates.append("preferred_mode = ?")
                    params.append(preferred_mode)
                if opt_in_marketing is not None:
                    updates.append("opt_in_marketing = ?")
                    params.append(1 if opt_in_marketing else 0)
                
                params.append(visitor_id)
                
                cursor.execute(f"""
                    UPDATE visitors SET {', '.join(updates)}
                    WHERE id = ?
                """, params)
            else:
                # Create new visitor
                cursor.execute("""
                    INSERT INTO visitors 
                    (visitor_uuid, email, preferred_mode, opt_in_marketing)
                    VALUES (?, ?, ?, ?)
                """, (visitor_uuid, email, preferred_mode, 1 if opt_in_marketing else 0))
                visitor_id = cursor.lastrowid
            
            conn.commit()
            return visitor_id
    
    def get_visitor_by_uuid(self, visitor_uuid: str) -> Optional[Dict]:
        """Get visitor by UUID."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM visitors WHERE visitor_uuid = ?", (visitor_uuid,))
            row = cursor.fetchone()
            return dict(row) if row else None
    
    def get_visitor_by_email(self, email: str) -> Optional[Dict]:
        """Get visitor by email."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM visitors WHERE email = ?", (email,))
            row = cursor.fetchone()
            return dict(row) if row else None
    
    def increment_visitor_sessions(self, visitor_id: int):
        """Increment session count for visitor."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                UPDATE visitors SET 
                    total_sessions = total_sessions + 1,
                    last_visit = CURRENT_TIMESTAMP
                WHERE id = ?
            """, (visitor_id,))
            conn.commit()
    
    def get_visitor_stats(self) -> Dict:
        """Get visitor statistics."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            
            cursor.execute("SELECT COUNT(*) as total FROM visitors")
            total = cursor.fetchone()['total']
            
            cursor.execute("SELECT COUNT(*) as with_email FROM visitors WHERE email IS NOT NULL")
            with_email = cursor.fetchone()['with_email']
            
            cursor.execute("""
                SELECT COUNT(*) as today FROM visitors 
                WHERE DATE(last_visit) = DATE('now')
            """)
            today = cursor.fetchone()['today']
            
            cursor.execute("""
                SELECT SUM(total_sessions) as total_sessions FROM visitors
            """)
            total_sessions = cursor.fetchone()['total_sessions'] or 0
            
            return {
                "total_visitors": total,
                "visitors_with_email": with_email,
                "email_collection_rate": round((with_email / total * 100) if total > 0 else 0, 1),
                "visitors_today": today,
                "total_sessions": total_sessions
            }
    
    # ============== Analytics Events ==============
    
    def log_analytics_event(
        self,
        event_type: str,
        visitor_id: Optional[int] = None,
        event_data: Optional[Dict] = None,
        session_id: Optional[int] = None
    ):
        """Log an analytics event."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO analytics_events 
                (visitor_id, event_type, event_data, session_id)
                VALUES (?, ?, ?, ?)
            """, (visitor_id, event_type, json.dumps(event_data or {}), session_id))
            conn.commit()
    
    def get_analytics_events(
        self,
        event_type: Optional[str] = None,
        visitor_id: Optional[int] = None,
        limit: int = 1000
    ) -> List[Dict]:
        """Get analytics events."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            
            query = "SELECT * FROM analytics_events WHERE 1=1"
            params = []
            
            if event_type:
                query += " AND event_type = ?"
                params.append(event_type)
            if visitor_id:
                query += " AND visitor_id = ?"
                params.append(visitor_id)
            
            query += " ORDER BY timestamp DESC LIMIT ?"
            params.append(limit)
            
            cursor.execute(query, params)
            return [dict(row) for row in cursor.fetchall()]
    
    def get_event_counts(self, days: int = 7) -> Dict[str, int]:
        """Get event type counts for the last N days."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT event_type, COUNT(*) as count
                FROM analytics_events
                WHERE timestamp >= datetime('now', ?)
                GROUP BY event_type
                ORDER BY count DESC
            """, (f'-{days} days',))
            return {row['event_type']: row['count'] for row in cursor.fetchall()}
    
    def get_conversion_funnel(self) -> Dict:
        """Get conversion funnel data."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            
            funnel = {}
            
            # Page views
            cursor.execute("""
                SELECT COUNT(DISTINCT visitor_id) as count
                FROM analytics_events
                WHERE event_type = 'page_view'
            """)
            funnel['page_views'] = cursor.fetchone()['count']
            
            # Email provided
            cursor.execute("""
                SELECT COUNT(DISTINCT visitor_id) as count
                FROM analytics_events
                WHERE event_type = 'email_provided'
            """)
            funnel['email_provided'] = cursor.fetchone()['count']
            
            # Mode selected
            cursor.execute("""
                SELECT COUNT(DISTINCT visitor_id) as count
                FROM analytics_events
                WHERE event_type = 'mode_selected'
            """)
            funnel['mode_selected'] = cursor.fetchone()['count']
            
            # Training started
            cursor.execute("""
                SELECT COUNT(DISTINCT visitor_id) as count
                FROM analytics_events
                WHERE event_type = 'training_started'
            """)
            funnel['training_started'] = cursor.fetchone()['count']
            
            return funnel
    
    # ============== Feedback Management ==============
    
    def submit_feedback(
        self,
        visitor_id: Optional[int],
        category: str,
        rating: Optional[int] = None,
        message: Optional[str] = None
    ) -> int:
        """Submit user feedback."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO feedback (visitor_id, category, rating, message)
                VALUES (?, ?, ?, ?)
            """, (visitor_id, category, rating, message))
            conn.commit()
            return cursor.lastrowid
    
    def get_feedback(
        self,
        category: Optional[str] = None,
        limit: int = 100
    ) -> List[Dict]:
        """Get feedback entries."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            
            if category:
                cursor.execute("""
                    SELECT f.*, v.email
                    FROM feedback f
                    LEFT JOIN visitors v ON f.visitor_id = v.id
                    WHERE f.category = ?
                    ORDER BY f.timestamp DESC
                    LIMIT ?
                """, (category, limit))
            else:
                cursor.execute("""
                    SELECT f.*, v.email
                    FROM feedback f
                    LEFT JOIN visitors v ON f.visitor_id = v.id
                    ORDER BY f.timestamp DESC
                    LIMIT ?
                """, (limit,))
            
            return [dict(row) for row in cursor.fetchall()]
    
    def get_feedback_stats(self) -> Dict:
        """Get feedback statistics."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            
            cursor.execute("SELECT COUNT(*) as total FROM feedback")
            total = cursor.fetchone()['total']
            
            cursor.execute("""
                SELECT AVG(rating) as avg_rating
                FROM feedback WHERE rating IS NOT NULL
            """)
            avg_rating = cursor.fetchone()['avg_rating']
            
            cursor.execute("""
                SELECT category, COUNT(*) as count
                FROM feedback
                GROUP BY category
            """)
            by_category = {row['category']: row['count'] for row in cursor.fetchall()}
            
            return {
                "total_feedback": total,
                "average_rating": round(avg_rating, 2) if avg_rating else None,
                "by_category": by_category
            }
    
    # ============== Challenges Management ==============
    
    def create_challenge(
        self,
        game_id: str,
        challenge_type: str,
        target_value: float,
        description: str,
        start_date: str,
        end_date: str
    ) -> int:
        """Create a new challenge."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO challenges 
                (game_id, challenge_type, target_value, description, start_date, end_date)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (game_id, challenge_type, target_value, description, start_date, end_date))
            conn.commit()
            return cursor.lastrowid
    
    def get_active_challenges(self) -> List[Dict]:
        """Get currently active challenges."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT * FROM challenges
                WHERE is_active = 1
                AND DATE('now') BETWEEN start_date AND end_date
                ORDER BY end_date ASC
            """)
            return [dict(row) for row in cursor.fetchall()]
    
    def get_challenge_progress(self, visitor_id: int, challenge_id: int) -> Optional[Dict]:
        """Get visitor's progress on a challenge."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT * FROM user_challenges
                WHERE visitor_id = ? AND challenge_id = ?
            """, (visitor_id, challenge_id))
            row = cursor.fetchone()
            return dict(row) if row else None
    
    def update_challenge_progress(
        self,
        visitor_id: int,
        challenge_id: int,
        progress: float,
        completed: bool = False
    ):
        """Update visitor's challenge progress."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            
            # Check if record exists
            cursor.execute("""
                SELECT id FROM user_challenges
                WHERE visitor_id = ? AND challenge_id = ?
            """, (visitor_id, challenge_id))
            
            existing = cursor.fetchone()
            
            if existing:
                cursor.execute("""
                    UPDATE user_challenges
                    SET progress = ?,
                        completed = ?,
                        completed_at = CASE WHEN ? = 1 THEN CURRENT_TIMESTAMP ELSE completed_at END
                    WHERE visitor_id = ? AND challenge_id = ?
                """, (progress, 1 if completed else 0, 1 if completed else 0, visitor_id, challenge_id))
            else:
                cursor.execute("""
                    INSERT INTO user_challenges
                    (visitor_id, challenge_id, progress, completed, completed_at)
                    VALUES (?, ?, ?, ?, ?)
                """, (
                    visitor_id, challenge_id, progress,
                    1 if completed else 0,
                    datetime.now() if completed else None
                ))
            
            conn.commit()
    
    def get_visitor_challenges(self, visitor_id: int) -> List[Dict]:
        """Get all challenges for a visitor with their progress."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT 
                    c.*,
                    uc.progress,
                    uc.completed,
                    uc.completed_at
                FROM challenges c
                LEFT JOIN user_challenges uc 
                    ON c.id = uc.challenge_id AND uc.visitor_id = ?
                WHERE c.is_active = 1
                AND DATE('now') BETWEEN c.start_date AND c.end_date
                ORDER BY c.end_date ASC
            """, (visitor_id,))
            return [dict(row) for row in cursor.fetchall()]

