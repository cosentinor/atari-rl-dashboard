"""
Pretrained model registry loader for the Atari dashboard.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, Optional

logger = logging.getLogger(__name__)


class PretrainedManager:
    """Loads and serves metadata about pre-trained models."""

    def __init__(self, base_dir: str = "pretrained_models"):
        self.base_dir = Path(base_dir)
        self.registry_path = self.base_dir / "registry.json"
        self.registry: Dict = self._load_registry()
        self.models: Dict = self.registry.get("models", {})
        self.games: Dict = self.registry.get("games", {})

    def _load_registry(self) -> Dict:
        if self.registry_path.exists():
            try:
                with open(self.registry_path, "r") as handle:
                    return json.load(handle)
            except Exception as exc:
                logger.warning(f"Failed to load pretrained registry: {exc}")
        return {"version": 1, "models": {}, "games": {}}

    def reload(self) -> None:
        self.registry = self._load_registry()
        self.models = self.registry.get("models", {})
        self.games = self.registry.get("games", {})

    def get_model(self, model_id: str) -> Optional[Dict]:
        if not model_id:
            return None
        return self.models.get(model_id)

    def _normalize_game_key(self, game_id: str) -> Optional[str]:
        if not game_id:
            return None
        if game_id in self.games:
            return game_id
        clean = game_id.split("/")[-1].replace("-v5", "").replace("-v4", "")
        if clean in self.games:
            return clean
        lower_map = {key.lower(): key for key in self.games.keys()}
        return lower_map.get(clean.lower())

    def get_game_entry(self, game_id: str) -> Optional[Dict]:
        key = self._normalize_game_key(game_id)
        if not key:
            return None
        return self.games.get(key)

    def get_game_levels(self, game_id: str) -> Dict[str, Optional[Dict]]:
        entry = self.get_game_entry(game_id)
        if not entry:
            return {"low": None, "medium": None, "high": None}
        levels = entry.get("levels", {})
        return {
            "low": self.get_model(levels.get("low")),
            "medium": self.get_model(levels.get("medium")),
            "high": self.get_model(levels.get("high")),
        }

    def get_game_models(self, game_id: str) -> list[Dict]:
        entry = self.get_game_entry(game_id)
        if not entry:
            return []
        model_ids = entry.get("models", [])
        return [self.models[m_id] for m_id in model_ids if m_id in self.models]

    def has_pretrained(self, game_id: str) -> bool:
        levels = self.get_game_levels(game_id)
        return any(levels.values())
