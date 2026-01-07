"""
Model Manager for Atari RL Training.
Handles saving, loading, and managing model checkpoints.
"""

import os
import json
import logging
import shutil
import re
from datetime import datetime
from typing import Dict, List, Optional, Any
from pathlib import Path

logger = logging.getLogger(__name__)


class ModelManager:
    """
    Manages model checkpoints for RL training.
    
    Features:
    - Organized storage by game
    - Auto-save at intervals
    - Best model tracking
    - Model registry for quick lookup
    """
    
    def __init__(self, base_dir: str = "saved_models"):
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)
        
        self.registry_path = self.base_dir / "model_registry.json"
        self.registry = self._load_registry()
        self._sync_registry_from_disk()
        
        # Auto-save settings
        self.auto_save_interval = 100  # Episodes
        self.max_checkpoints_per_game = 5
        
        logger.info(f"ModelManager initialized at {self.base_dir}")
    
    def _load_registry(self) -> Dict:
        """Load or create model registry."""
        if self.registry_path.exists():
            try:
                with open(self.registry_path, 'r') as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"Failed to load registry: {e}")
        return {"games": {}, "last_updated": None}
    
    def _save_registry(self):
        """Save registry to disk with error handling."""
        try:
            self.registry["last_updated"] = datetime.now().isoformat()
            # Write to temp file first, then rename for atomicity
            temp_path = self.registry_path.with_suffix('.tmp')
            with open(temp_path, 'w') as f:
                json.dump(self.registry, f, indent=2)
            temp_path.replace(self.registry_path)
            logger.debug(f"Registry saved to {self.registry_path}")
        except Exception as e:
            logger.error(f"Failed to save registry: {e}")
            # Try direct write as fallback
            try:
                with open(self.registry_path, 'w') as f:
                    json.dump(self.registry, f, indent=2)
            except Exception as e2:
                logger.error(f"Fallback save also failed: {e2}")

    def _sync_registry_from_disk(self):
        """Ensure registry includes checkpoints present on disk."""
        changed = False

        if "games" not in self.registry:
            self.registry["games"] = {}
            changed = True

        for game_dir in self.base_dir.iterdir():
            if not game_dir.is_dir():
                continue

            clean_name = game_dir.name
            game_entry = self.registry["games"].get(clean_name)
            if not game_entry:
                game_entry = {
                    "game_id": f"ALE/{clean_name}-v5",
                    "checkpoints": [],
                    "best_checkpoint": None,
                    "best_reward": float("-inf"),
                }
                self.registry["games"][clean_name] = game_entry
                changed = True
            elif not game_entry.get("game_id"):
                game_entry["game_id"] = f"ALE/{clean_name}-v5"
                changed = True

            existing = {
                cp.get("filename"): cp
                for cp in game_entry.get("checkpoints", [])
                if cp.get("filename")
            }
            disk_checkpoints = {p.name: p for p in game_dir.glob("checkpoint_ep*.pt")}

            next_checkpoints = []
            for filename, path in disk_checkpoints.items():
                episode = 0
                match = re.search(r"checkpoint_ep(\d+)_", filename)
                if match:
                    episode = int(match.group(1))

                timestamp_match = re.search(r"checkpoint_ep\d+_(\d{8}_\d{6})", filename)
                if timestamp_match:
                    try:
                        timestamp = datetime.strptime(timestamp_match.group(1), "%Y%m%d_%H%M%S").isoformat()
                    except ValueError:
                        timestamp = datetime.fromtimestamp(path.stat().st_mtime).isoformat()
                else:
                    timestamp = datetime.fromtimestamp(path.stat().st_mtime).isoformat()

                existing_checkpoint = existing.get(filename, {})
                reward = existing_checkpoint.get("reward")
                prev_timestamp = existing_checkpoint.get("timestamp")
                entry = {
                    "filename": filename,
                    "episode": episode,
                    "reward": reward,
                    "timestamp": prev_timestamp or timestamp,
                }
                next_checkpoints.append(entry)

            if set(existing) != set(disk_checkpoints):
                changed = True
            else:
                for cp in next_checkpoints:
                    existing_cp = existing.get(cp["filename"], {})
                    if (
                        existing_cp.get("episode") != cp["episode"]
                        or existing_cp.get("reward") != cp["reward"]
                        or existing_cp.get("timestamp") != cp["timestamp"]
                    ):
                        changed = True
                        break

            game_entry["checkpoints"] = next_checkpoints

            best_filename = game_entry.get("best_checkpoint")
            if best_filename and best_filename not in disk_checkpoints:
                game_entry["best_checkpoint"] = None
                game_entry["best_reward"] = float("-inf")
                changed = True

        if changed:
            self._save_registry()
    
    def get_game_dir(self, game_id: str) -> Path:
        """Get directory for a specific game."""
        # Clean game_id for filesystem (e.g., "ALE/Pong-v5" -> "Pong")
        clean_name = game_id.split("/")[-1].replace("-v5", "").replace("-", "_")
        game_dir = self.base_dir / clean_name
        game_dir.mkdir(parents=True, exist_ok=True)
        return game_dir
    
    def save_checkpoint(
        self,
        agent,
        game_id: str,
        episode: int,
        reward: float,
        is_best: bool = False,
        metadata: Optional[Dict] = None
    ) -> str:
        """
        Save a model checkpoint.
        
        Args:
            agent: RainbowAgent instance
            game_id: Game identifier
            episode: Current episode number
            reward: Episode reward
            is_best: Whether this is the best model so far
            metadata: Additional metadata to save
            
        Returns:
            Path to saved checkpoint
        """
        game_dir = self.get_game_dir(game_id)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Create checkpoint filename
        filename = f"checkpoint_ep{episode}_{timestamp}.pt"
        filepath = game_dir / filename
        
        # Prepare metadata
        full_metadata = {
            "game_id": game_id,
            "episode": episode,
            "reward": reward,
            "timestamp": datetime.now().isoformat(),
            "is_best": is_best,
            **(metadata or {})
        }
        
        # Save using agent's save method (with cleanup fallback)
        self._ensure_disk_space()
        if not self._safe_save(agent, str(filepath), episode, reward, full_metadata):
            logger.error(f"Checkpoint save failed after cleanup; skipping save for {filepath}")
            return ""
        
        # Update registry
        clean_name = game_id.split("/")[-1].replace("-v5", "").replace("-", "_")
        if clean_name not in self.registry["games"]:
            self.registry["games"][clean_name] = {
                "game_id": game_id,
                "checkpoints": [],
                "best_checkpoint": None,
                "best_reward": float("-inf")
            }
        
        game_entry = self.registry["games"][clean_name]
        checkpoint_info = {
            "filename": filename,
            "episode": episode,
            "reward": reward,
            "timestamp": datetime.now().isoformat()
        }
        game_entry["checkpoints"].append(checkpoint_info)
        
        # Update best if needed
        if is_best or reward > game_entry["best_reward"]:
            game_entry["best_checkpoint"] = filename
            game_entry["best_reward"] = reward
            
            # Also save as best_model.pt
            best_path = game_dir / "best_model.pt"
            if not self._safe_save(agent, str(best_path), episode, reward, full_metadata):
                logger.warning(f"Failed to save best_model for {game_id}")
        
        # Cleanup old checkpoints (keep max_checkpoints_per_game)
        self._cleanup_old_checkpoints(game_dir, clean_name)
        
        self._save_registry()
        logger.info(f"Checkpoint saved: {filepath}")
        
        return str(filepath)
    
    def _cleanup_old_checkpoints(self, game_dir: Path, game_name: str):
        """Remove old checkpoints, keeping only the most recent ones."""
        game_entry = self.registry["games"].get(game_name, {})
        checkpoints = game_entry.get("checkpoints", [])
        
        if len(checkpoints) > self.max_checkpoints_per_game:
            # Sort by timestamp (oldest first)
            sorted_checkpoints = sorted(checkpoints, key=lambda x: x["timestamp"])
            
            # Keep only the most recent + best
            to_remove = sorted_checkpoints[:-self.max_checkpoints_per_game]
            best_filename = game_entry.get("best_checkpoint")
            
            removed_any = False
            for cp in to_remove:
                if cp["filename"] != best_filename:
                    filepath = game_dir / cp["filename"]
                    try:
                        if filepath.exists():
                            filepath.unlink()
                            logger.info(f"Removed old checkpoint: {filepath}")
                        checkpoints.remove(cp)
                        removed_any = True
                    except Exception as e:
                        logger.warning(f"Failed to remove checkpoint {filepath}: {e}")
            
            game_entry["checkpoints"] = checkpoints
            
            # Save registry after cleanup if anything changed
            if removed_any:
                self._save_registry()

        # Also prune on-disk checkpoints in case registry is stale
        self._prune_game_dir(game_dir)

    def _prune_game_dir(self, game_dir: Path):
        """Prune checkpoints on disk by mtime to enforce retention."""
        try:
            checkpoints = list(game_dir.glob("checkpoint_ep*.pt"))
            if not checkpoints:
                return

            # Keep the highest-episode checkpoint to avoid losing progress.
            highest_path = None
            highest_ep = -1
            for path in checkpoints:
                match = re.search(r"checkpoint_ep(\d+)_", path.name)
                if match:
                    ep = int(match.group(1))
                    if ep > highest_ep:
                        highest_ep = ep
                        highest_path = path

            # Keep newest checkpoints by mtime.
            newest = sorted(
                checkpoints,
                key=lambda p: p.stat().st_mtime,
                reverse=True
            )[:self.max_checkpoints_per_game]

            keep = set(newest)
            if highest_path:
                keep.add(highest_path)

            for path in checkpoints:
                if path in keep:
                    continue
                try:
                    path.unlink()
                    logger.info(f"Removed old checkpoint: {path}")
                except Exception as e:
                    logger.warning(f"Failed to remove checkpoint {path}: {e}")
        except Exception as e:
            logger.warning(f"Failed to prune checkpoints in {game_dir}: {e}")

    def _prune_all_checkpoints(self):
        """Prune checkpoints across all games to enforce retention."""
        for game_dir in self.base_dir.iterdir():
            if game_dir.is_dir():
                self._prune_game_dir(game_dir)

    def _ensure_disk_space(self, min_free_gb: float = 5.0):
        """Ensure enough free disk space for saving checkpoints."""
        try:
            usage = shutil.disk_usage(self.base_dir)
            free_gb = usage.free / (1024 ** 3)
            if free_gb < min_free_gb:
                logger.warning(
                    f"Low disk space ({free_gb:.1f} GB free). Pruning checkpoints."
                )
                self._prune_all_checkpoints()
        except Exception as e:
            logger.warning(f"Disk space check failed: {e}")

    def _safe_save(self, agent, filepath: str, episode: int, reward: float, metadata: Dict) -> bool:
        """Save a checkpoint with a cleanup + retry fallback."""
        try:
            agent.save(filepath, episode, reward, metadata)
            return True
        except Exception as e:
            logger.warning(f"Checkpoint save failed: {e}. Pruning and retrying.")
            self._prune_all_checkpoints()
            try:
                agent.save(filepath, episode, reward, metadata)
                return True
            except Exception as e2:
                logger.error(f"Checkpoint save failed after cleanup: {e2}")
                return False

    def _find_latest_checkpoint(self, game_dir: Path) -> tuple[Optional[str], int]:
        """Find the latest checkpoint filename and episode in a game directory."""
        latest_ep = -1
        latest_name = None
        latest_mtime = -1.0
        for path in game_dir.glob("checkpoint_ep*.pt"):
            match = re.search(r"checkpoint_ep(\d+)_", path.name)
            if not match:
                continue
            ep = int(match.group(1))
            mtime = path.stat().st_mtime
            if ep > latest_ep or (ep == latest_ep and mtime > latest_mtime):
                latest_ep = ep
                latest_name = path.name
                latest_mtime = mtime
        return latest_name, latest_ep
    
    def load_checkpoint(
        self,
        agent,
        game_id: str,
        checkpoint_name: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Load a model checkpoint.
        
        Args:
            agent: RainbowAgent instance to load into
            game_id: Game identifier
            checkpoint_name: Specific checkpoint filename (None = load best)
            
        Returns:
            Checkpoint data including metadata
        """
        game_dir = self.get_game_dir(game_id)
        
        if checkpoint_name is None:
            # Load best model
            filepath = game_dir / "best_model.pt"
            if not filepath.exists():
                raise FileNotFoundError(f"No best model found for {game_id}")
        else:
            filepath = game_dir / checkpoint_name
            if not filepath.exists():
                raise FileNotFoundError(f"Checkpoint not found: {filepath}")
        
        checkpoint = agent.load(str(filepath))
        logger.info(f"Loaded checkpoint from {filepath}")
        
        return checkpoint
    
    def get_available_checkpoints(self, game_id: str) -> List[Dict]:
        """Get list of available checkpoints for a game."""
        clean_name = game_id.split("/")[-1].replace("-v5", "").replace("-", "_")
        game_entry = self.registry["games"].get(clean_name, {})
        
        checkpoints = game_entry.get("checkpoints", [])
        best_filename = game_entry.get("best_checkpoint")
        
        # Mark which one is best
        result = []
        for cp in checkpoints:
            cp_copy = cp.copy()
            cp_copy["is_best"] = cp["filename"] == best_filename
            result.append(cp_copy)
        
        # Sort by episode (newest first)
        result.sort(key=lambda x: x["episode"], reverse=True)
        
        return result

    def get_latest_checkpoint_name(self, game_id: str) -> Optional[str]:
        """Get the latest checkpoint filename for a game (highest episode)."""
        game_dir = self.get_game_dir(game_id)
        latest_name, _ = self._find_latest_checkpoint(game_dir)
        return latest_name

    def get_latest_checkpoint_path(self, game_id: str) -> Optional[str]:
        """Get the full path to the latest checkpoint for a game."""
        game_dir = self.get_game_dir(game_id)
        latest_name, _ = self._find_latest_checkpoint(game_dir)
        if latest_name:
            return str(game_dir / latest_name)
        return None

    def has_checkpoints(self, game_id: str) -> bool:
        """Check if any checkpoints exist for a game."""
        game_dir = self.get_game_dir(game_id)
        if (game_dir / "best_model.pt").exists():
            return True
        return any(game_dir.glob("checkpoint_ep*.pt"))

    def resolve_checkpoint_path(self, game_id: str, checkpoint_name: str) -> Optional[str]:
        """Resolve a checkpoint filename to a safe on-disk path."""
        if not checkpoint_name:
            return None
        game_dir = self.get_game_dir(game_id)
        candidate = (game_dir / checkpoint_name).resolve()
        if not candidate.exists():
            return None
        if game_dir.resolve() not in candidate.parents:
            return None
        return str(candidate)
    
    def get_all_games(self) -> List[Dict]:
        """Get list of all games with saved models."""
        result = []
        for game_name, game_data in self.registry["games"].items():
            result.append({
                "name": game_name,
                "game_id": game_data.get("game_id", ""),
                "num_checkpoints": len(game_data.get("checkpoints", [])),
                "best_reward": game_data.get("best_reward", 0),
                "has_best_model": game_data.get("best_checkpoint") is not None
            })
        return result
    
    def should_auto_save(self, episode: int) -> bool:
        """Check if auto-save should trigger."""
        return episode > 0 and episode % self.auto_save_interval == 0
    
    def delete_checkpoint(self, game_id: str, checkpoint_name: str) -> bool:
        """Delete a specific checkpoint."""
        game_dir = self.get_game_dir(game_id)
        filepath = game_dir / checkpoint_name
        
        if filepath.exists():
            filepath.unlink()
            
            # Update registry
            clean_name = game_id.split("/")[-1].replace("-v5", "").replace("-", "_")
            game_entry = self.registry["games"].get(clean_name, {})
            checkpoints = game_entry.get("checkpoints", [])
            
            game_entry["checkpoints"] = [
                cp for cp in checkpoints if cp["filename"] != checkpoint_name
            ]
            
            self._save_registry()
            logger.info(f"Deleted checkpoint: {filepath}")
            return True
        
        return False
    
    def get_best_model_path(self, game_id: str) -> Optional[str]:
        """Get path to best model for a game."""
        game_dir = self.get_game_dir(game_id)
        best_path = game_dir / "best_model.pt"
        
        if best_path.exists():
            return str(best_path)
        return None
    
    def export_model(self, agent, game_id: str, export_path: str):
        """Export model to a custom path."""
        agent.save(export_path, agent.episode_count, 0, {
            "game_id": game_id,
            "exported_at": datetime.now().isoformat()
        })
        logger.info(f"Model exported to {export_path}")
