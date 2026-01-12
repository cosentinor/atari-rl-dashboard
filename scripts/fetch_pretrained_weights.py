#!/usr/bin/env python3
"""Download pre-trained Atari model weights and build a registry."""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional
from urllib.parse import urlencode
from urllib.request import urlopen, urlretrieve

BITDEFENDER_BUCKET = "bitdefender_ml_artifacts"
BITDEFENDER_ALGOS = ["DQN_modern", "MDQN_modern", "C51_classic", "DQN_classic_adam"]
SB3_ALGOS = ["dqn", "a2c", "ppo", "qrdqn"]
PFRL_ALGOS = ["DQN", "Rainbow", "A3C"]

GAMES = [
    "Pong",
    "Breakout",
    "SpaceInvaders",
    "MsPacman",
    "Asteroids",
    "Boxing",
    "BeamRider",
    "Seaquest",
    "Enduro",
    "Freeway",
]

GAME_ID_MAP = {
    "Pong": "ALE/Pong-v5",
    "Breakout": "ALE/Breakout-v5",
    "SpaceInvaders": "ALE/SpaceInvaders-v5",
    "MsPacman": "ALE/MsPacman-v5",
    "Asteroids": "ALE/Asteroids-v5",
    "Boxing": "ALE/Boxing-v5",
    "BeamRider": "ALE/BeamRider-v5",
    "Seaquest": "ALE/Seaquest-v5",
    "Enduro": "ALE/Enduro-v5",
    "Freeway": "ALE/Freeway-v5",
}

LEVEL_TARGETS = {
    "low": 10_000_000,
    "medium": 40_000_000,
    "high": 50_000_000,
}

PREFERRED_BITDEFENDER_ALGOS = ["DQN_modern", "MDQN_modern", "DQN_classic_adam", "C51_classic"]


def fetch_gcs_items(prefix: str) -> List[Dict]:
    items: List[Dict] = []
    page_token: Optional[str] = None

    while True:
        params = {"prefix": prefix}
        if page_token:
            params["pageToken"] = page_token
        url = f"https://storage.googleapis.com/storage/v1/b/{BITDEFENDER_BUCKET}/o?{urlencode(params)}"
        with urlopen(url) as resp:
            payload = json.load(resp)
        items.extend(payload.get("items", []))
        page_token = payload.get("nextPageToken")
        if not page_token:
            break
    return items


def select_closest_step(steps: List[int], target: int) -> Optional[int]:
    if not steps:
        return None
    return min(steps, key=lambda step: (abs(step - target), step))


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def download_file(url: str, dest: Path) -> None:
    ensure_dir(dest.parent)
    if dest.exists():
        return
    urlretrieve(url, dest)


def register_model(registry: Dict, model: Dict) -> None:
    model_id = model["id"]
    registry["models"][model_id] = model
    game_key = model["game"]
    game_entry = registry["games"].setdefault(
        game_key,
        {
            "game_id": GAME_ID_MAP.get(game_key),
            "models": [],
            "levels": {},
        },
    )
    if model_id not in game_entry["models"]:
        game_entry["models"].append(model_id)


def build_bitdefender_models(base_dir: Path, seed: str, include_algos: List[str]) -> Dict[str, Dict]:
    models_by_game_algo: Dict[str, Dict[str, List[Dict]]] = {game: {} for game in GAMES}
    registry = {"models": {}, "games": {}}

    for algo in include_algos:
        for game in GAMES:
            prefix = f"atari/{algo}/{game}/"
            items = fetch_gcs_items(prefix)
            if not items:
                continue

            step_items: Dict[int, Dict] = {}
            for item in items:
                name = item.get("name") or ""
                parts = name.split("/")
                if len(parts) < 5:
                    continue
                if parts[3] != seed:
                    continue
                filename = parts[-1]
                if not filename.startswith("model_") or not filename.endswith(".gz"):
                    continue
                step_str = filename[len("model_") : -len(".gz")]
                try:
                    step = int(step_str)
                except ValueError:
                    continue
                step_items[step] = item

            if not step_items:
                continue

            steps = sorted(step_items.keys())
            selected_steps = {
                level: select_closest_step(steps, target)
                for level, target in LEVEL_TARGETS.items()
            }

            for level, step in selected_steps.items():
                if step is None:
                    continue
                item = step_items.get(step)
                if not item:
                    continue
                object_name = item["name"]
                filename = object_name.split("/")[-1]
                url = f"https://storage.googleapis.com/{BITDEFENDER_BUCKET}/{object_name}"
                dest = base_dir / "bitdefender" / algo / game / seed / filename
                download_file(url, dest)

                model_id = f"bitdefender:{algo}:{game}:{seed}:{step}"
                model = {
                    "id": model_id,
                    "source": "bitdefender",
                    "algorithm": algo,
                    "game": game,
                    "game_id": GAME_ID_MAP.get(game),
                    "seed": int(seed) if seed.isdigit() else seed,
                    "step": step,
                    "level": level,
                    "filename": filename,
                    "path": str(dest),
                    "size_bytes": int(item.get("size") or 0),
                    "format": "gz",
                    "framework": "pytorch",
                    "url": url,
                }
                register_model(registry, model)
                models_by_game_algo.setdefault(game, {}).setdefault(algo, []).append(model)

    # Assign levels based on preferred algos
    for game, algo_map in models_by_game_algo.items():
        levels = {}
        for level, target in LEVEL_TARGETS.items():
            chosen = None
            for algo in PREFERRED_BITDEFENDER_ALGOS:
                candidates = algo_map.get(algo, [])
                if not candidates:
                    continue
                chosen = min(candidates, key=lambda m: (abs(m["step"] - target), m["step"]))
                break
            if chosen:
                levels[level] = chosen["id"]
        if levels:
            registry["games"].setdefault(game, {"game_id": GAME_ID_MAP.get(game), "models": [], "levels": {}})
            registry["games"][game]["levels"].update(levels)

    return registry


def fetch_sb3_models(base_dir: Path, registry: Dict) -> None:
    for algo in SB3_ALGOS:
        for game in GAMES:
            env_id = f"{game}NoFrameskip-v4"
            repo_id = f"sb3/{algo}-{env_id}"
            api_url = f"https://huggingface.co/api/models/{repo_id}"
            try:
                with urlopen(api_url) as resp:
                    payload = json.load(resp)
            except Exception:
                continue

            siblings = payload.get("siblings", [])
            zip_files = [s["rfilename"] for s in siblings if s.get("rfilename", "").endswith(".zip")]
            if not zip_files:
                continue
            filename = zip_files[0]
            url = f"https://huggingface.co/{repo_id}/resolve/main/{filename}"
            dest = base_dir / "sb3" / algo / game / filename
            download_file(url, dest)

            model_id = f"sb3:{algo}:{game}:{filename}"
            model = {
                "id": model_id,
                "source": "sb3",
                "algorithm": algo,
                "game": game,
                "game_id": GAME_ID_MAP.get(game),
                "filename": filename,
                "path": str(dest),
                "format": "zip",
                "framework": "stable-baselines3",
                "url": url,
            }
            register_model(registry, model)


def fetch_pfrl_models(base_dir: Path, registry: Dict) -> None:
    for algo in PFRL_ALGOS:
        for game in GAMES:
            for model_type in ("best", "final"):
                url = f"https://pfrl-assets.preferred.jp/{algo}/{game}/{model_type}.zip"
                dest = base_dir / "pfrl" / algo / game / f"{model_type}.zip"
                try:
                    download_file(url, dest)
                except Exception:
                    continue
                model_id = f"pfrl:{algo}:{game}:{model_type}"
                model = {
                    "id": model_id,
                    "source": "pfrl",
                    "algorithm": algo,
                    "game": game,
                    "game_id": GAME_ID_MAP.get(game),
                    "filename": dest.name,
                    "path": str(dest),
                    "format": "zip",
                    "framework": "pfrl",
                    "url": url,
                }
                register_model(registry, model)


def main() -> None:
    parser = argparse.ArgumentParser(description="Download pre-trained Atari weights.")
    parser.add_argument("--base-dir", default="pretrained_models", help="Destination folder for weights")
    parser.add_argument("--seed", default="0", help="Seed folder to use for Bitdefender models")
    parser.add_argument(
        "--bitdefender-algos",
        nargs="+",
        default=BITDEFENDER_ALGOS,
        help="Bitdefender algorithms to include",
    )
    parser.add_argument("--include-sb3", action="store_true", help="Download SB3 RL Zoo models")
    parser.add_argument("--include-pfrl", action="store_true", help="Download PFRL model zoo weights")
    args = parser.parse_args()

    base_dir = Path(args.base_dir)
    base_dir.mkdir(parents=True, exist_ok=True)

    registry = {
        "version": 1,
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "models": {},
        "games": {},
    }

    bitdefender_registry = build_bitdefender_models(base_dir, str(args.seed), args.bitdefender_algos)
    registry["models"].update(bitdefender_registry.get("models", {}))
    registry["games"].update(bitdefender_registry.get("games", {}))

    if args.include_sb3:
        fetch_sb3_models(base_dir, registry)
    if args.include_pfrl:
        fetch_pfrl_models(base_dir, registry)

    registry_path = base_dir / "registry.json"
    with open(registry_path, "w") as handle:
        json.dump(registry, handle, indent=2)

    print(f"Saved registry to {registry_path}")
    print(f"Total models: {len(registry['models'])}")


if __name__ == "__main__":
    main()
