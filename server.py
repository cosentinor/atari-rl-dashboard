"""
Atari RL Training Server with Rainbow DQN.
Flask-SocketIO server for real-time game streaming and training.
"""

import logging
import os
import threading
import time
from pathlib import Path
from flask import Flask, send_from_directory, jsonify, request, send_file
from flask_socketio import SocketIO, emit
from flask_cors import CORS

from game_environments import GameEnvironments
from frame_streamer import FrameStreamer
from rainbow_agent import RainbowAgent, FrameStack, get_device
from model_manager import ModelManager
from pretrained_manager import PretrainedManager
from pretrained_policies import BitdefenderPolicy, SB3Policy, PFRLPolicy, LocalCheckpointPolicy
from db_manager import TrainingDatabase
from config import AUTOSAVE_INTERVAL_SECONDS, TRAINING_LEVELS

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Frontend build location
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FRONTEND_BUILD_DIR = os.environ.get(
    "FRONTEND_BUILD_DIR",
    os.path.join(BASE_DIR, "frontend", "build"),
)

# Flask app
app = Flask(__name__, static_folder=FRONTEND_BUILD_DIR, static_url_path="")

# CORS Configuration - Allow both development and production origins
cors_origins = [
    "http://localhost:3000",  # React development server
    "http://localhost:5001",  # React build served by this backend
    "http://127.0.0.1:5001",  # Loopback for local access
    "https://atari.riccardocosentino.com",  # Production domain
    "http://atari.riccardocosentino.com",  # Production HTTP fallback
]

CORS(app, resources={
    r"/*": {
        "origins": cors_origins,
        "methods": ["GET", "POST", "PUT", "DELETE", "OPTIONS"],
        "allow_headers": ["Content-Type", "Authorization"],
        "supports_credentials": True
    }
})

socketio = SocketIO(
    app, 
    cors_allowed_origins=cors_origins,
    async_mode='threading',
    ping_timeout=60,
    ping_interval=25,
    allow_upgrades=False,
    transports=['polling']
)


def _build_init_payload() -> dict:
    """Build the init payload for new clients."""
    games = [
        {
            'id': g.id,
            'name': g.name,
            'display_name': g.display_name,
            'action_space_size': int(g.action_space_size) if g.action_space_size is not None else 0,
            'action_names': g.action_names or [],
            'is_available': g.is_available
        }
        for g in game_envs.games.values()
        if g.is_available
    ]

    device = get_device()

    return {
        'games': games,
        'isTraining': is_training,
        'currentGame': current_game,
        'sessionId': current_session_id,
        'savedModels': model_manager.get_all_games(),
        'device': str(device),
        'trainingSpeed': training_speed,
        'trainingLevel': training_level,
        'vizFrameSkip': viz_frame_skip,
        'vizTargetFps': viz_target_fps
    }

# Global state
game_envs = GameEnvironments()
model_manager = ModelManager()
pretrained_manager = PretrainedManager()
db = TrainingDatabase()

# Training state
streamer = None
training_thread = None
is_training = False
training_stop_event = threading.Event()
training_time_lock = threading.Lock()
last_training_save_time = None
current_game = None
current_session_id = None
rainbow_agent = None
frame_stack = None
current_run_mode = "train"
current_pretrained_model = None
pretrained_policy = None
current_num_actions = None

# Watch mode state
watch_mode_active = False
watch_mode_game = None
watch_mode_thread = None

# Speed control settings
training_speed = "1x"  # 1x, 2x, 4x
training_level = "medium"  # low, medium, high
LOCAL_MODEL_ID_PREFIX = "rc_model"
LOCAL_MODEL_SOURCE = "local"
LOCAL_MODEL_ALGORITHM = "RC_model"
LOCAL_MODEL_MIN_EPISODES = 10000
LOCAL_MODEL_MAX_EPISODES = 50000
BITDEFENDER_LEVEL_STEPS = {"medium": 10000000, "high": 50000000}
BITDEFENDER_ALGORITHM_PREFERENCE = ("DQN_modern", "DQN")
viz_frame_skip = 1  # 1 = every frame, 2 = every 2nd, etc.
viz_target_fps = 24  # Visualization FPS target for streaming

# Per-game visualization settings (tuned for readability)
DEFAULT_VIZ_SETTINGS = {
    'target_fps': 24,
    'frame_skip': 2,
}

GAME_VIZ_SETTINGS = {
    'ALE/Pong-v5': {'target_fps': 24, 'frame_skip': 2},
    'ALE/Breakout-v5': {'target_fps': 22, 'frame_skip': 2},
    'ALE/SpaceInvaders-v5': {'target_fps': 20, 'frame_skip': 2},
    'ALE/MsPacman-v5': {'target_fps': 18, 'frame_skip': 2},
    'ALE/Asteroids-v5': {'target_fps': 20, 'frame_skip': 2},
    'ALE/Boxing-v5': {'target_fps': 22, 'frame_skip': 2},
    'ALE/Seaquest-v5': {'target_fps': 18, 'frame_skip': 2},
    'ALE/BeamRider-v5': {'target_fps': 18, 'frame_skip': 2},
    'ALE/Enduro-v5': {'target_fps': 20, 'frame_skip': 2},
    'ALE/Freeway-v5': {'target_fps': 16, 'frame_skip': 2},
}

# Performance: Connection pooling
MAX_CONCURRENT_TRAINING = 3  # Max simultaneous training sessions
active_training_sessions = 0
training_queue = []

def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None or raw == "":
        return default
    try:
        return int(raw)
    except ValueError:
        logger.warning("Invalid int for %s=%r; using default %s", name, raw, default)
        return default


def _env_bool(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}

def _resolve_training_level(value: str | None) -> str:
    """Normalize training level input."""
    if not value:
        return "medium"
    key = str(value).strip().lower()
    if key in {"low", "light"}:
        return "low"
    if key in {"medium", "med"}:
        return "medium"
    if key in {"high", "heavy"}:
        return "high"
    return "medium"





def _format_game_label(game_id: str) -> str:
    info = game_envs.get_game_info(game_id)
    if info:
        return info.display_name or info.name or game_id
    return game_id.split("/")[-1].replace("-v5", "")


def _local_model_id(game_id: str, checkpoint_name: str) -> str:
    return f"{LOCAL_MODEL_ID_PREFIX}:{game_id}:{checkpoint_name}"


def _parse_local_model_id(model_id: str):
    if not model_id:
        return None
    prefix = f"{LOCAL_MODEL_ID_PREFIX}:"
    if not model_id.startswith(prefix):
        return None
    payload = model_id[len(prefix):]
    game_id, sep, checkpoint = payload.partition(":")
    if not sep:
        return None
    return game_id, checkpoint


def _build_local_model_info(game_id: str, checkpoint: dict, path: str) -> dict:
    game_label = _format_game_label(game_id)
    return {
        "id": _local_model_id(game_id, checkpoint.get("filename")),
        "source": LOCAL_MODEL_SOURCE,
        "algorithm": LOCAL_MODEL_ALGORITHM,
        "game": game_label,
        "game_id": game_id,
        "seed": None,
        "step": None,
        "level": "low",
        "filename": checkpoint.get("filename"),
        "format": "pt",
        "framework": "pytorch",
        "path": path,
        "episode": checkpoint.get("episode"),
        "reward": checkpoint.get("reward"),
        "timestamp": checkpoint.get("timestamp"),
    }


def _resolve_local_checkpoint(game_id: str) -> dict | None:
    checkpoints = model_manager.get_available_checkpoints(game_id)
    candidates = [
        cp
        for cp in checkpoints
        if isinstance(cp.get("episode"), int)
        and LOCAL_MODEL_MIN_EPISODES <= cp.get("episode") <= LOCAL_MODEL_MAX_EPISODES
    ]
    if not candidates:
        return None
    checkpoint = max(candidates, key=lambda cp: cp.get("episode") or 0)
    filename = checkpoint.get("filename")
    if not filename:
        return None
    path = model_manager.resolve_checkpoint_path(game_id, filename)
    if not path:
        return None
    return _build_local_model_info(game_id, checkpoint, path)


def _resolve_local_model_by_id(model_id: str) -> dict | None:
    parsed = _parse_local_model_id(model_id)
    if not parsed:
        return None
    game_id, checkpoint_name = parsed
    path = model_manager.resolve_checkpoint_path(game_id, checkpoint_name)
    if not path:
        return None
    checkpoint = next(
        (cp for cp in model_manager.get_available_checkpoints(game_id) if cp.get("filename") == checkpoint_name),
        {"filename": checkpoint_name},
    )
    return _build_local_model_info(game_id, checkpoint, path)


def _select_bitdefender_model(game_id: str, step_target: int) -> dict | None:
    models = pretrained_manager.get_game_models(game_id)
    candidates = [
        model
        for model in models
        if (model.get("source") or "").lower() == "bitdefender"
        and model.get("step") == step_target
    ]
    if not candidates:
        return None
    def algo_rank(model: dict) -> int:
        algo = (model.get("algorithm") or "").lower()
        for idx, pref in enumerate(BITDEFENDER_ALGORITHM_PREFERENCE):
            if algo == pref.lower():
                return idx
        return len(BITDEFENDER_ALGORITHM_PREFERENCE)
    def seed_rank(model: dict) -> int:
        seed = model.get("seed")
        return seed if isinstance(seed, int) else 999
    return min(candidates, key=lambda model: (algo_rank(model), seed_rank(model)))


def _resolve_pretrained_level(game_id: str, level: str) -> dict | None:
    resolved = _resolve_training_level(level)
    if resolved == "low":
        return _resolve_local_checkpoint(game_id)
    step_target = BITDEFENDER_LEVEL_STEPS.get(resolved)
    if not step_target:
        return None
    model = _select_bitdefender_model(game_id, step_target)
    if not model:
        return None
    model_copy = dict(model)
    model_copy["level"] = resolved
    return model_copy


def _resolve_pretrained_model(game_id: str, level: str, pretrained_id: str | None) -> dict | None:
    if pretrained_id:
        local_info = _resolve_local_model_by_id(pretrained_id)
        if local_info:
            return local_info
        model_info = pretrained_manager.get_model(pretrained_id)
        if model_info:
            model_copy = dict(model_info)
            model_copy.setdefault("level", _resolve_training_level(level))
            return model_copy
    return _resolve_pretrained_level(game_id, level)


def _trim_memory():
    """Best-effort release of unused heap memory after training ends."""
    try:
        import gc
        gc.collect()
        if os.name == "posix":
            import ctypes
            libc = ctypes.CDLL("libc.so.6")
            libc.malloc_trim(0)
    except Exception:
        pass


# ============== HTTP Routes ==============

def _frontend_ready():
    return os.path.isfile(os.path.join(FRONTEND_BUILD_DIR, "index.html"))


def _frontend_missing_response():
    return jsonify({
        "success": False,
        "message": (
            "Frontend build not found. Run `npm run build` in "
            "frontend or set "
            "FRONTEND_BUILD_DIR to the build output."
        ),
    }), 404


def _send_frontend_index():
    response = send_from_directory(FRONTEND_BUILD_DIR, 'index.html', conditional=False)
    response.headers["Cache-Control"] = "no-store, max-age=0"
    response.headers["Pragma"] = "no-cache"
    response.headers["Expires"] = "0"
    return response


def _resolve_visitor_id(raw_visitor_id, email: str | None = None) -> int | None:
    """Resolve a visitor identifier (UUID or numeric id) to a visitor row id."""
    if raw_visitor_id is None and not email:
        return None

    if isinstance(raw_visitor_id, int):
        return raw_visitor_id

    visitor_uuid = None
    if raw_visitor_id is not None:
        visitor_uuid = str(raw_visitor_id).strip()
        if visitor_uuid.isdigit():
            return int(visitor_uuid)

    if email and not visitor_uuid:
        visitor = db.get_visitor_by_email(email)
        if visitor:
            return visitor["id"]

    if visitor_uuid:
        visitor = db.get_visitor_by_uuid(visitor_uuid)
        if visitor:
            if email and not visitor.get("email"):
                db.create_or_update_visitor(visitor_uuid=visitor_uuid, email=email)
            return visitor["id"]
        return db.create_or_update_visitor(visitor_uuid=visitor_uuid, email=email)

    return None


@app.route('/')
def index():
    """Serve the React frontend."""
    if not _frontend_ready():
        return _frontend_missing_response()
    return _send_frontend_index()


@app.route('/<path:filename>')
def serve_static(filename):
    """Serve static files from frontend folder."""
    if filename.startswith('api/'):
        return jsonify({'success': False, 'message': 'Not found'}), 404
    if not _frontend_ready():
        return _frontend_missing_response()
    target_path = os.path.join(FRONTEND_BUILD_DIR, filename)
    if os.path.isfile(target_path):
        return send_from_directory(FRONTEND_BUILD_DIR, filename)
    return _send_frontend_index()


@app.route('/api/games')
def get_games():
    """Get list of available games."""
    import json
    games = game_envs.get_all_games_info()
    games_json = json.loads(json.dumps(games, default=lambda x: int(x) if hasattr(x, 'item') else x))
    return {'success': True, 'games': games_json}


@app.route('/api/models')
def get_models():
    """Get all saved models."""
    return jsonify({
        'success': True,
        'games': model_manager.get_all_games()
    })


@app.route('/api/models/<game_id>')
def get_game_models(game_id):
    """Get checkpoints for a specific game."""
    # Handle URL encoding
    game_id = game_id.replace('_', '/')
    return jsonify({
        'success': True,
        'checkpoints': model_manager.get_available_checkpoints(game_id)
    })


@app.route('/api/models/<game_id>/download')
def download_model(game_id):
    """Download a model checkpoint for a specific game."""
    game_id = game_id.replace('_', '/')
    mode = (request.args.get('mode') or 'latest').strip().lower()
    checkpoint_name = request.args.get('checkpoint')

    try:
        model_path = None
        if mode == 'checkpoint':
            if not checkpoint_name:
                return jsonify({'success': False, 'message': 'Missing checkpoint name'}), 400
            model_path = model_manager.resolve_checkpoint_path(game_id, checkpoint_name)
        elif mode == 'best':
            model_path = model_manager.get_best_model_path(game_id)
        else:
            model_path = model_manager.get_latest_checkpoint_path(game_id)
            if not model_path:
                model_path = model_manager.get_best_model_path(game_id)

        if not model_path:
            return jsonify({'success': False, 'message': 'No checkpoint available'}), 404

        filename = os.path.basename(model_path)
        return send_file(model_path, as_attachment=True, download_name=filename)
    except Exception as e:
        logger.error(f"Failed to download model: {e}")
        return jsonify({'success': False, 'message': str(e)}), 500




@app.route('/api/pretrained/<game_id>')
def get_pretrained_models(game_id):
    """Get pretrained model levels for a specific game."""
    game_id = game_id.replace('_', '/')
    levels = {
        "low": _resolve_pretrained_level(game_id, "low"),
        "medium": _resolve_pretrained_level(game_id, "medium"),
        "high": _resolve_pretrained_level(game_id, "high"),
    }
    serialized_levels = {
        "low": serialize_pretrained_model(levels.get("low")),
        "medium": serialize_pretrained_model(levels.get("medium")),
        "high": serialize_pretrained_model(levels.get("high")),
    }
    models = [serialize_pretrained_model(m) for m in pretrained_manager.get_game_models(game_id)]
    models = [m for m in models if m]
    return jsonify({
        'success': True,
        'game_id': game_id,
        'levels': serialized_levels,
        'models': models,
    })


@app.route('/api/pretrained/download')
def download_pretrained_model():
    """Download a pretrained model file by model id."""
    model_id = request.args.get('id')
    if not model_id:
        return jsonify({'success': False, 'message': 'Missing model id'}), 400

    local_info = _resolve_local_model_by_id(model_id)
    if local_info:
        model_path = Path(local_info.get('path') or '').resolve()
        base_dir = model_manager.base_dir.resolve()
        if not model_path.exists():
            return jsonify({'success': False, 'message': 'Model file missing on disk'}), 404
        if base_dir not in model_path.parents:
            return jsonify({'success': False, 'message': 'Invalid model path'}), 400
        return send_file(model_path, as_attachment=True, download_name=model_path.name)

    model_info = pretrained_manager.get_model(model_id)
    if not model_info:
        return jsonify({'success': False, 'message': 'Pretrained model not found'}), 404

    model_path = Path(model_info.get('path') or '')
    if not model_path.is_absolute():
        model_path = pretrained_manager.base_dir / model_path
    model_path = model_path.resolve()
    base_dir = pretrained_manager.base_dir.resolve()
    if not model_path.exists():
        return jsonify({'success': False, 'message': 'Model file missing on disk'}), 404
    if base_dir not in model_path.parents:
        return jsonify({'success': False, 'message': 'Invalid model path'}), 400

    return send_file(model_path, as_attachment=True, download_name=model_path.name)


@app.route('/api/leaderboard')
def get_leaderboard():
    """Get training leaderboard."""
    game_id = request.args.get('game_id')
    limit = request.args.get('limit', default=10, type=int)
    return jsonify({
        'success': True,
        'leaderboard': db.get_leaderboard(game_id, limit=limit)
    })


@app.route('/api/session/<int:session_id>/history')
def get_session_history(session_id):
    """Get episode history for a session."""
    return jsonify({
        'success': True,
        'episodes': db.get_recent_episodes(session_id, n=500)
    })


@app.route('/api/session/<int:session_id>/stats')
def get_session_stats(session_id):
    """Get session statistics."""
    return jsonify({
        'success': True,
        'stats': db.get_reward_stats(session_id)
    })


@app.route('/api/device')
def get_device_info():
    """Get compute device info."""
    device = get_device()
    return jsonify({
        'success': True,
        'device': str(device),
        'cuda_available': hasattr(__import__('torch'), 'cuda') and __import__('torch').cuda.is_available()
    })


# ============== Visitor Management ==============

def _register_visitor_from_payload(data: dict) -> tuple[str, int]:
    import uuid

    visitor_uuid = data.get("visitor_uuid") or data.get("visitor_id")
    if visitor_uuid:
        visitor_uuid = str(visitor_uuid).strip()
    else:
        visitor_uuid = str(uuid.uuid4())

    email = data.get("email")
    opt_in_marketing = bool(data.get("opt_in_marketing", False))
    preferred_mode = data.get("preferred_mode")

    visitor_id = db.create_or_update_visitor(
        visitor_uuid=visitor_uuid,
        email=email,
        preferred_mode=preferred_mode,
        opt_in_marketing=opt_in_marketing
    )

    event_type = "email_provided" if email else "visitor_registered"
    event_data = {
        "user_agent": data.get("user_agent"),
        "screen_resolution": data.get("screen_resolution"),
        "referrer": data.get("referrer"),
        "email_collected": data.get("email_collected"),
        "opt_in_marketing": opt_in_marketing
    }
    event_data = {k: v for k, v in event_data.items() if v is not None}
    db.log_analytics_event(
        event_type=event_type,
        visitor_id=visitor_id,
        event_data=event_data
    )

    return visitor_uuid, visitor_id


@app.route('/api/analytics/register', methods=['POST'])
def register_analytics_visitor():
    """Register or update a visitor from the analytics client."""
    data = request.get_json() or {}
    try:
        visitor_uuid, visitor_id = _register_visitor_from_payload(data)
        return jsonify({
            'success': True,
            'visitor_uuid': visitor_uuid,
            'visitor_id': visitor_id
        })
    except Exception as e:
        logger.error(f"Failed to register visitor: {e}")
        return jsonify({'success': False, 'message': str(e)}), 500


@app.route('/api/visitor/register', methods=['POST'])
def register_visitor():
    """Register a new visitor or update existing (legacy endpoint)."""
    data = request.get_json() or {}
    try:
        visitor_uuid, visitor_id = _register_visitor_from_payload(data)
        return jsonify({
            'success': True,
            'visitor_uuid': visitor_uuid,
            'visitor_id': visitor_id
        })
    except Exception as e:
        logger.error(f"Failed to register visitor: {e}")
        return jsonify({'success': False, 'message': str(e)}), 500


# ============== Analytics ==============

@app.route('/api/analytics/batch', methods=['POST'])
def log_analytics_batch():
    """Log multiple analytics events."""
    data = request.get_json() or {}
    events = data.get('events', [])

    try:
        for event in events:
            event_type = event.get('event_type')
            if not event_type:
                continue

            visitor_id = _resolve_visitor_id(
                event.get('visitor_id') or event.get('visitor_uuid')
            )
            db.log_analytics_event(
                event_type=event_type,
                visitor_id=visitor_id,
                event_data=event.get('event_data'),
                session_id=event.get('session_id')
            )

        return jsonify({'success': True})
    except Exception as e:
        logger.error(f"Failed to log analytics: {e}")
        return jsonify({'success': False, 'message': str(e)}), 500


@app.route('/api/analytics/funnel')
def get_conversion_funnel():
    """Get conversion funnel data."""
    try:
        funnel = db.get_conversion_funnel()
        return jsonify({'success': True, 'funnel': funnel})
    except Exception as e:
        logger.error(f"Failed to get funnel: {e}")
        return jsonify({'success': False, 'message': str(e)}), 500


# ============== Feedback ==============

@app.route('/api/feedback', methods=['POST'])
def submit_feedback():
    """Submit user feedback."""
    data = request.get_json() or {}

    visitor_id = _resolve_visitor_id(data.get('visitor_id'), email=data.get('email'))
    category = data.get('category', 'general')
    rating = data.get('rating')
    message = data.get('message')

    try:
        feedback_id = db.submit_feedback(
            visitor_id=visitor_id,
            category=category,
            rating=rating,
            message=message
        )

        if visitor_id:
            db.log_analytics_event(
                event_type='feedback_submitted',
                visitor_id=visitor_id,
                event_data={'category': category, 'rating': rating}
            )

        return jsonify({'success': True, 'feedback_id': feedback_id})
    except Exception as e:
        logger.error(f"Failed to submit feedback: {e}")
        return jsonify({'success': False, 'message': str(e)}), 500


@app.route('/api/feedback/stats')
def get_feedback_stats():
    """Get feedback statistics."""
    try:
        stats = db.get_feedback_stats()
        return jsonify({'success': True, 'stats': stats})
    except Exception as e:
        logger.error(f"Failed to get feedback stats: {e}")
        return jsonify({'success': False, 'message': str(e)}), 500


# ============== Challenges ==============

@app.route('/api/challenges')
def get_challenges():
    """Get active challenges for visitor."""
    raw_visitor_id = request.args.get('visitor_id')
    visitor_id = _resolve_visitor_id(raw_visitor_id)

    try:
        if visitor_id:
            challenges = db.get_visitor_challenges(visitor_id)
        else:
            challenges = db.get_active_challenges()

        return jsonify({'success': True, 'challenges': challenges})
    except Exception as e:
        logger.error(f"Failed to get challenges: {e}")
        return jsonify({'success': False, 'message': str(e)}), 500


@app.route('/api/challenges/<int:challenge_id>/progress', methods=['POST'])
def update_challenge_progress(challenge_id):
    """Update challenge progress."""
    data = request.get_json() or {}
    visitor_id = _resolve_visitor_id(data.get('visitor_id'))
    progress = data.get('progress', 0)
    completed = data.get('completed', False)

    if not visitor_id:
        return jsonify({'success': False, 'message': 'Missing visitor_id'}), 400

    try:
        db.update_challenge_progress(visitor_id, challenge_id, progress, completed)
        return jsonify({'success': True})
    except Exception as e:
        logger.error(f"Failed to update challenge progress: {e}")
        return jsonify({'success': False, 'message': str(e)}), 500


# ============== Public Stats ==============

@app.route('/api/stats/public')
def get_public_stats():
    """Get public statistics for hero section."""
    try:
        visitor_stats = db.get_visitor_stats()
        return jsonify({
            'success': True,
            'stats': {
                'visitors': visitor_stats.get('total_visitors', 0),
                'sessions': visitor_stats.get('total_sessions', 0),
                'modelsTrainedToday': visitor_stats.get('visitors_today', 0)
            }
        })
    except Exception as e:
        logger.error(f"Failed to get public stats: {e}")
        return jsonify({'success': False, 'message': str(e)}), 500


# ============== Model Comparison ==============

@app.route('/api/models/compare', methods=['POST'])
def compare_models():
    """Compare multiple model checkpoints."""
    data = request.get_json()
    game_id = data.get('game_id')
    checkpoints = data.get('checkpoints', [])
    
    try:
        # Get checkpoint details
        comparison_data = []
        for checkpoint_name in checkpoints:
            checkpoints_list = model_manager.get_available_checkpoints(game_id)
            checkpoint = next((c for c in checkpoints_list if c['filename'] == checkpoint_name), None)
            if checkpoint:
                comparison_data.append(checkpoint)
        
        return jsonify({
            'success': True,
            'comparison': {
                'checkpoints': comparison_data,
                'game_id': game_id
            }
        })
    except Exception as e:
        logger.error(f"Failed to compare models: {e}")
        return jsonify({'success': False, 'message': str(e)}), 500


# ============== Share Session ==============

@app.route('/share/<int:session_id>')
def share_session(session_id):
    """Public share page for a training session."""
    # This would render a special share page
    # For now, redirect to main page
    return index()


# ============== Watch Mode ==============

@app.route('/api/watch/start', methods=['POST'])
def start_watch_mode():
    """Start watch mode with best model for a game."""
    global watch_mode_active, watch_mode_game
    
    data = request.get_json()
    game_id = data.get('game_id')
    
    if not game_id:
        return jsonify({'success': False, 'message': 'No game specified'}), 400
    
    try:
        # Find best model for this game
        checkpoints = model_manager.get_available_checkpoints(game_id)
        if not checkpoints:
            return jsonify({'success': False, 'message': 'No trained models available'}), 404
        
        # Get best checkpoint
        best_checkpoint = next((c for c in checkpoints if c.get('is_best')), checkpoints[0])
        
        watch_mode_active = True
        watch_mode_game = game_id
        
        return jsonify({
            'success': True,
            'game_id': game_id,
            'checkpoint': best_checkpoint['filename']
        })
    except Exception as e:
        logger.error(f"Failed to start watch mode: {e}")
        return jsonify({'success': False, 'message': str(e)}), 500


@app.route('/api/watch/stop', methods=['POST'])
def stop_watch_mode():
    """Stop watch mode."""
    global watch_mode_active
    watch_mode_active = False
    return jsonify({'success': True})

@app.route('/api/training/stop', methods=['POST'])
def api_stop_training():
    """Stop current training via HTTP."""
    _stop_training()
    return jsonify({'success': True})


# ============== Queue Management ==============

@app.route('/api/queue/status')
def get_queue_status():
    """Get training queue status."""
    return jsonify({
        'success': True,
        'active_sessions': active_training_sessions,
        'max_sessions': MAX_CONCURRENT_TRAINING,
        'queue_length': len(training_queue),
        'can_start': active_training_sessions < MAX_CONCURRENT_TRAINING
    })


# ============== WebSocket Events ==============

def resolve_game_id(raw_game_id: str) -> str:
    """Resolve game IDs from short names or display names to ALE ids."""
    if not raw_game_id:
        return raw_game_id

    if raw_game_id in game_envs.games:
        return raw_game_id

    alt_id = raw_game_id.replace('_', '/')
    if alt_id in game_envs.games:
        return alt_id

    normalized = ''.join(ch.lower() for ch in raw_game_id if ch.isalnum())
    for game_info in game_envs.games.values():
        if game_info.name.lower() == raw_game_id.lower():
            return game_info.id
        if game_info.display_name.lower() == raw_game_id.lower():
            return game_info.id
        if ''.join(ch.lower() for ch in game_info.name if ch.isalnum()) == normalized:
            return game_info.id
        if ''.join(ch.lower() for ch in game_info.display_name if ch.isalnum()) == normalized:
            return game_info.id

    return raw_game_id


def get_viz_settings(game_id: str) -> dict:
    """Return visualization settings for a given game."""
    settings = GAME_VIZ_SETTINGS.get(game_id, DEFAULT_VIZ_SETTINGS)
    return {
        'target_fps': int(settings.get('target_fps', DEFAULT_VIZ_SETTINGS['target_fps'])),
        'frame_skip': int(settings.get('frame_skip', DEFAULT_VIZ_SETTINGS['frame_skip'])),
    }


@socketio.on('connect')
def handle_connect(auth=None):
    """Handle new client connection."""
    logger.info("Client connected")





def serialize_pretrained_model(model_info: dict | None) -> dict | None:
    if not model_info:
        return None
    return {
        'id': model_info.get('id'),
        'source': model_info.get('source'),
        'algorithm': model_info.get('algorithm'),
        'game': model_info.get('game'),
        'game_id': model_info.get('game_id'),
        'seed': model_info.get('seed'),
        'step': model_info.get('step'),
        'level': model_info.get('level'),
        'filename': model_info.get('filename'),
        'format': model_info.get('format'),
        'framework': model_info.get('framework'),
        'episode': model_info.get('episode'),
        'reward': model_info.get('reward'),
        'timestamp': model_info.get('timestamp'),
    }
def load_pretrained_policy(model_info: dict, num_actions: int, device):
    """Create a policy instance for a pretrained model entry."""
    source = (model_info.get('source') or '').lower()
    algorithm = model_info.get('algorithm') or ''
    model_path = Path(model_info.get('path') or '')
    if not model_path.is_absolute():
        model_path = (pretrained_manager.base_dir / model_path).resolve()

    if source == 'bitdefender':
        num_atoms = 51 if 'C51' in algorithm else 1
        return BitdefenderPolicy(model_path, num_actions=num_actions, num_atoms=num_atoms, device=device)
    if source in {'local', 'rc_model'}:
        return LocalCheckpointPolicy(model_path, num_actions=num_actions, device=device)
    if source == 'sb3':
        return SB3Policy(model_path, algorithm=algorithm, device=device)
    if source == 'pfrl':
        return PFRLPolicy(model_path, algorithm=algorithm, num_actions=num_actions, device=device)

    raise ValueError(f"Unsupported pretrained source: {source}")



@socketio.on('get_init')
def handle_get_init():
    """Send init payload after the client confirms it is connected."""
    emit('init', _build_init_payload())


@socketio.on('disconnect')
def handle_disconnect():
    """Handle client disconnect."""
    logger.info("Client disconnected")
    
    # Stop training on disconnect as requested
    if is_training:
        logger.info("Stopping training due to client disconnect")
        _stop_training()


@socketio.on('start_training')
def handle_start_training(data):
    """Start training for a game."""
    global is_training, current_game, current_session_id
    global streamer, training_thread, rainbow_agent, frame_stack
    global current_run_mode, current_pretrained_model, pretrained_policy, current_num_actions
    global active_training_sessions, viz_frame_skip, viz_target_fps, training_level
    
    logger.info(f"Received start_training event with data: {data}")
    
    if is_training:
        emit('error', {'message': 'Training already in progress'})
        return
    
    # Check queue capacity
    if active_training_sessions >= MAX_CONCURRENT_TRAINING:
        queue_position = len(training_queue) + 1
        training_queue.append(data)
        emit('queued', {
            'position': queue_position,
            'message': f'Server is busy. You are #{queue_position} in queue.'
        })
        return
    
    raw_game_id = data.get('game')
    load_checkpoint = data.get('loadCheckpoint')  # Optional checkpoint to load
    resume_from_saved = data.get('resumeFromSaved')
    run_mode = data.get('runMode') or data.get('mode') or ('pretrained' if data.get('pretrainedId') else 'train')
    pretrained_id = data.get('pretrainedId') or data.get('pretrained_id') or data.get('pretrainedModelId')
    run_mode = str(run_mode).strip().lower()
    if run_mode not in {'train', 'pretrained'}:
        run_mode = 'train'

    if isinstance(load_checkpoint, str) and load_checkpoint.strip() == "":
        load_checkpoint = None

    if isinstance(resume_from_saved, str):
        resume_from_saved = resume_from_saved.strip().lower() in {"1", "true", "yes", "on"}

    requested_level = data.get('trainingLevel') or data.get('training_level')
    training_level = _resolve_training_level(requested_level)
    
    if not raw_game_id:
        emit('error', {'message': 'No game specified'})
        return
    
    game_id = resolve_game_id(raw_game_id)
    if game_id != raw_game_id:
        logger.info(f"Resolved game id '{raw_game_id}' -> '{game_id}'")

    if game_id not in game_envs.games:
        emit('error', {'message': f'Unknown game: {raw_game_id}'})
        return

    if run_mode == 'train' and model_manager.has_checkpoints(game_id):
        resume_from_saved = True
        load_checkpoint = None

    viz_settings = get_viz_settings(game_id)
    viz_frame_skip = viz_settings['frame_skip']
    viz_target_fps = viz_settings['target_fps']
    logger.info(
        f"Starting training for game_id: {game_id} | viz_fps={viz_target_fps} | frame_skip={viz_frame_skip}"
    )
    
    try:
        # Create environment
        env = game_envs.create_environment(game_id)
        
        # Get action space size
        num_actions = env.action_space.n
        current_num_actions = num_actions
        
        # Create frame stacker
        frame_stack = FrameStack(num_frames=4, frame_size=(84, 84))
        
        device = get_device()
        
        if run_mode == 'pretrained':
            model_info = _resolve_pretrained_model(game_id, training_level, pretrained_id)
            if not model_info:
                emit('error', {'message': 'Pre-trained model not found for this game'})
                return
            if model_info.get('game_id') and model_info.get('game_id') != game_id:
                emit('error', {'message': 'Pre-trained model does not match selected game'})
                return
            current_run_mode = 'pretrained'
            current_pretrained_model = model_info
            pretrained_policy = load_pretrained_policy(model_info, num_actions, device)
            rainbow_agent = None
        else:
            current_run_mode = 'train'
            current_pretrained_model = None
            pretrained_policy = None
        
            # Create Rainbow agent
            level_config = TRAINING_LEVELS.get(training_level, TRAINING_LEVELS['medium'])
            buffer_size = _env_int('RL_BUFFER_SIZE', level_config['buffer_size'])
            min_buffer_size = _env_int('RL_MIN_BUFFER_SIZE', level_config['min_buffer_size'])
            batch_size = _env_int('RL_BATCH_SIZE', level_config['batch_size'])
            n_step = _env_int('RL_N_STEP', level_config['n_step'])
            store_uint8 = _env_bool('RL_STORE_UINT8', level_config.get('store_uint8', False))
            if min_buffer_size > buffer_size:
                logger.warning(
                    'RL_MIN_BUFFER_SIZE %s exceeds RL_BUFFER_SIZE %s; clamping.',
                    min_buffer_size,
                    buffer_size,
                )
                min_buffer_size = buffer_size
            logger.info(
                'Training config: level=%s buffer_size=%s min_buffer_size=%s batch_size=%s n_step=%s store_uint8=%s',
                training_level,
                buffer_size,
                min_buffer_size,
                batch_size,
                n_step,
                store_uint8,
            )
            rainbow_agent = RainbowAgent(
                state_shape=(4, 84, 84),
                num_actions=num_actions,
                device=device,
                buffer_size=buffer_size,
                min_buffer_size=min_buffer_size,
                batch_size=batch_size,
                n_step=n_step,
                store_uint8=store_uint8
            )
        
            # Decide whether to resume from saved weights
            if resume_from_saved is None:
                resume_from_saved = model_manager.has_checkpoints(game_id)
        
            # Load checkpoint if specified or if resume is enabled
            if resume_from_saved:
                checkpoint_to_load = load_checkpoint or model_manager.get_latest_checkpoint_name(game_id)
                try:
                    if checkpoint_to_load:
                        model_manager.load_checkpoint(rainbow_agent, game_id, checkpoint_to_load)
                        logger.info(f'Loaded checkpoint: {checkpoint_to_load}')
                        emit('log', {'message': f'Loaded checkpoint: {checkpoint_to_load}', 'type': 'success'})
                    else:
                        best_path = model_manager.get_best_model_path(game_id)
                        if best_path:
                            model_manager.load_checkpoint(rainbow_agent, game_id, None)
                            logger.info('Loaded checkpoint: best_model.pt')
                            emit('log', {'message': 'Loaded checkpoint: best_model.pt', 'type': 'success'})
                        else:
                            emit('log', {'message': 'Starting fresh (no checkpoints found)', 'type': 'info'})
                except Exception as e:
                    logger.warning(f'Failed to load checkpoint: {e}')
                    emit('log', {'message': 'Starting fresh (checkpoint load failed)', 'type': 'warning'})
        # Create streamer
        streamer = FrameStreamer(env, socketio, target_fps=viz_target_fps)
        
        if run_mode == 'pretrained':
            hyperparameters = {
                'mode': 'pretrained',
                'model_id': current_pretrained_model.get('id') if current_pretrained_model else None,
                'source': current_pretrained_model.get('source') if current_pretrained_model else None,
                'algorithm': current_pretrained_model.get('algorithm') if current_pretrained_model else None,
                'step': current_pretrained_model.get('step') if current_pretrained_model else None,
            }
        else:
            hyperparameters = rainbow_agent.get_hyperparameters()
        
        # Create database session
        current_session_id = db.create_session(
            game_id=game_id,
            device=str(device),
            hyperparameters=hyperparameters,
        )
        
        current_game = game_id
        _start_training_activity(game_id)
        training_stop_event.clear()
        is_training = True
        active_training_sessions += 1
        
        # Notify client
        emit('training_started', {
            'game': game_id,
            'sessionId': current_session_id,
            'device': str(device),
            'trainingLevel': training_level,
            'vizFrameSkip': viz_frame_skip,
            'vizTargetFps': viz_target_fps,
            'runMode': run_mode,
            'pretrainedModel': current_pretrained_model if run_mode == 'pretrained' else None,
        })
        socketio.emit('status', {'isTraining': True, 'game': game_id, 'runMode': run_mode})
        
        # Start training in background thread
        thread_target = run_pretrained_loop if run_mode == 'pretrained' else run_training_loop
        training_thread = threading.Thread(target=thread_target, daemon=True)
        training_thread.start()
        
    except Exception as e:
        logger.error(f"Failed to start training: {e}")
        import traceback
        traceback.print_exc()
        emit('error', {'message': str(e)})


def _start_training_activity(game_id: str):
    """Initialize training activity tracking for leaderboard stats."""
    global last_training_save_time
    if not game_id:
        return
    with training_time_lock:
        last_training_save_time = time.time()
    db.record_training_activity(game_id, sessions=1)


def _record_training_time(game_id: str, now: float | None = None):
    """Record elapsed training time at the configured autosave interval."""
    global last_training_save_time
    if not game_id:
        return
    if now is None:
        now = time.time()
    with training_time_lock:
        if last_training_save_time is None:
            last_training_save_time = now
            return
        elapsed = now - last_training_save_time
        if elapsed < AUTOSAVE_INTERVAL_SECONDS:
            return
        last_training_save_time = now
    seconds = int(elapsed)
    if seconds <= 0:
        return
    db.record_training_activity(game_id, duration_seconds=seconds)


def _finalize_training_activity(game_id: str, episodes: int = 0, steps: int = 0):
    """Flush remaining training time and counters when training stops."""
    global last_training_save_time
    if not game_id:
        return
    now = time.time()
    with training_time_lock:
        if last_training_save_time is None:
            elapsed_seconds = 0
        else:
            elapsed_seconds = int(now - last_training_save_time)
            last_training_save_time = None
    db.record_training_activity(
        game_id=game_id,
        duration_seconds=max(elapsed_seconds, 0),
        episodes=episodes,
        steps=steps
    )


def _stop_training():
    """Shared stop logic for socket and HTTP."""
    global is_training

    logger.info("Stopping training")
    _finalize_training_activity(current_game)
    is_training = False
    training_stop_event.set()

    socketio.emit('training_stopped', {})
    socketio.emit('status', {'isTraining': False, 'game': None})


@socketio.on('stop_training')
def handle_stop_training(data=None):
    """Stop current training."""
    _stop_training()


@socketio.on('save_model')
def handle_save_model():
    """Manually save current model."""
    if not rainbow_agent or not current_game:
        emit('error', {'message': 'No active training session'})
        return
    
    try:
        # Get current stats
        stats = rainbow_agent.get_stats()
        episode = stats.get('episode_count', 0)
        
        # Get best reward from session
        session_stats = db.get_reward_stats(current_session_id) if current_session_id else {}
        best_reward = session_stats.get('best_reward', 0)
        
        # Save checkpoint
        path = model_manager.save_checkpoint(
            rainbow_agent,
            current_game,
            episode,
            best_reward,
            is_best=True,
            metadata={'session_id': current_session_id}
        )
        
        emit('model_saved', {'path': path, 'episode': episode})
        emit('log', {'message': f'Model saved at episode {episode}', 'type': 'success'})
        
    except Exception as e:
        logger.error(f"Failed to save model: {e}")
        emit('error', {'message': f'Save failed: {str(e)}'})


@socketio.on('load_model')
def handle_load_model(data):
    """Load a saved model."""
    game_id = data.get('game_id')
    checkpoint = data.get('checkpoint')
    
    if not rainbow_agent:
        emit('error', {'message': 'Start training first'})
        return
    
    try:
        model_manager.load_checkpoint(rainbow_agent, game_id, checkpoint)
        emit('model_loaded', {'checkpoint': checkpoint})
        emit('log', {'message': f'Loaded: {checkpoint}', 'type': 'success'})
    except Exception as e:
        emit('error', {'message': str(e)})


@socketio.on('set_training_level')
def handle_set_training_level(data):
    """Update training level selection and swap pretrained policy if running."""
    global training_level, pretrained_policy, current_pretrained_model
    global current_run_mode, current_game, current_num_actions

    requested_level = None
    if isinstance(data, dict):
        requested_level = data.get('trainingLevel') or data.get('training_level')
    training_level = _resolve_training_level(requested_level)

    if not is_training or current_run_mode != 'pretrained' or not current_game:
        socketio.emit('log', {'message': f'Training level set to {training_level}.', 'type': 'info'})
        return

    model_info = _resolve_pretrained_level(current_game, training_level)
    if not model_info:
        emit('error', {'message': 'No pre-trained model available for this level'})
        return

    if current_pretrained_model and model_info.get('id') == current_pretrained_model.get('id'):
        return

    num_actions = current_num_actions
    if not num_actions:
        game_info = game_envs.get_game_info(current_game)
        num_actions = game_info.action_space_size if game_info else 0
    if not num_actions:
        emit('error', {'message': 'Unable to switch model (unknown action space)'})
        return

    try:
        pretrained_policy = load_pretrained_policy(model_info, num_actions, get_device())
        current_pretrained_model = model_info
        socketio.emit('log', {'message': f'Switched to {training_level} snapshot.', 'type': 'success'})
    except Exception as exc:
        logger.error(f"Failed to switch pretrained model: {exc}")
        emit('error', {'message': str(exc)})


@socketio.on('set_training_speed')
def handle_set_training_speed(data):
    """Set training speed."""
    global training_speed
    speed = data.get('speed', '1x')
    
    if speed in ['1x', '2x', '4x']:
        training_speed = speed
        logger.info(f"Training speed set to: {speed}")
        socketio.emit('speed_changed', {'trainingSpeed': speed})


@socketio.on('set_viz_speed')
def handle_set_viz_speed(data):
    """Set visualization frame skip."""
    global viz_frame_skip, viz_target_fps
    skip = data.get('frameSkip', 1)
    target_fps = data.get('targetFps')
    
    if isinstance(skip, int) and 1 <= skip <= 100:
        viz_frame_skip = skip
        logger.info(f"Visualization frame skip set to: {skip}")
        socketio.emit('speed_changed', {'vizFrameSkip': skip})

    if isinstance(target_fps, (int, float)) and 5 <= target_fps <= 60:
        viz_target_fps = int(target_fps)
        if streamer:
            streamer.set_target_fps(viz_target_fps)
        logger.info(f"Visualization target FPS set to: {viz_target_fps}")
        socketio.emit('speed_changed', {'vizTargetFps': viz_target_fps})


@socketio.on('delete_checkpoint')
def handle_delete_checkpoint(data):
    """Delete a saved checkpoint."""
    game_id = data.get('game_id')
    checkpoint_name = data.get('checkpoint')
    
    if not game_id or not checkpoint_name:
        emit('error', {'message': 'Missing game_id or checkpoint name'})
        return
    
    try:
        success = model_manager.delete_checkpoint(game_id, checkpoint_name)
        if success:
            logger.info(f"Deleted checkpoint: {checkpoint_name} for {game_id}")
            # Send back updated checkpoint list
            checkpoints = model_manager.get_available_checkpoints(game_id)
            emit('checkpoint_deleted', {
                'success': True,
                'checkpoint': checkpoint_name,
                'checkpoints': checkpoints
            })
            emit('log', {'message': f'Deleted: {checkpoint_name}', 'type': 'success'})
        else:
            emit('error', {'message': f'Checkpoint not found: {checkpoint_name}'})
    except Exception as e:
        logger.error(f"Failed to delete checkpoint: {e}")
        emit('error', {'message': f'Delete failed: {str(e)}'})


@socketio.on('get_history')
def handle_get_history(data):
    """Get training history for charts."""
    session_id = data.get('sessionId', current_session_id)
    
    if not session_id:
        emit('history_data', {'episodes': [], 'steps': []})
        return
    
    episodes = db.get_recent_episodes(session_id, n=200)
    steps = db.get_step_metrics(session_id, limit=500)
    
    # Get action distribution
    action_dist = db.get_action_distribution(session_id)
    
    # Get reward distribution
    reward_bins, reward_counts = db.get_reward_distribution(session_id)
    
    emit('history_data', {
        'episodes': episodes,
        'steps': steps,
        'actionDistribution': action_dist,
        'rewardDistribution': {
            'bins': reward_bins,
            'counts': reward_counts
        }
    })


# ============== Training Loop ==============

def get_fire_action(game_id: str):
    """Return the FIRE action index if available for the game."""
    game_info = game_envs.get_game_info(game_id)
    if not game_info or not game_info.action_names:
        return None
    try:
        return game_info.action_names.index('FIRE')
    except ValueError:
        return None


def run_training_loop():
    """Main training loop running in background thread."""
    global is_training, active_training_sessions
    global streamer, rainbow_agent, frame_stack, current_game, current_session_id

    local_streamer = streamer
    local_agent = rainbow_agent
    local_frame_stack = frame_stack
    local_game = current_game
    local_session_id = current_session_id

    if not local_streamer or not local_agent or not local_frame_stack:
        return

    episode = local_agent.episode_count
    total_steps = local_agent.step_count
    best_episode_reward = float('-inf')
    session_episode_count = 0
    session_step_count = 0

    # Metrics for emission
    episode_start_time = time.time()
    frame_counter = 0
    step_sample_rate = 100  # Log step metrics every N steps

    # Time-based autosave tracking
    last_autosave_time = time.time()

    logger.info("Training loop started")

    try:
        while is_training and local_streamer and local_agent and not training_stop_event.is_set():
            episode += 1
            local_agent.episode_count = episode

            # Reset environment
            obs, _ = local_streamer.env.reset()
            state = local_frame_stack.reset(obs)
            fire_action = get_fire_action(local_game)
            if fire_action is not None:
                try:
                    obs, _, terminated, truncated, _ = local_streamer.env.step(fire_action)
                    if terminated or truncated:
                        obs, _ = local_streamer.env.reset()
                    state = local_frame_stack.reset(obs)
                except Exception as exc:
                    logger.debug(f"Auto-fire failed for {local_game}: {exc}")

            episode_reward = 0
            step = 0
            done = False
            episode_start_time = time.time()
            losses = []
            q_values = []

            while is_training and local_streamer and not done and not training_stop_event.is_set():
                step += 1
                total_steps += 1
                session_step_count += 1
                frame_counter += 1

                # Select action using Rainbow agent
                action = local_agent.select_action(state, training=True)

                # Step environment
                next_obs, reward, terminated, truncated, info = local_streamer.env.step(action)
                done = terminated or truncated
                episode_reward += reward

                # Process next state
                next_state = local_frame_stack.push(next_obs)

                # Store transition
                local_agent.push_transition(state, action, reward, next_state, done)

                # Learn
                loss = local_agent.learn()
                if loss is not None:
                    losses.append(loss)

                # Track Q-values
                q_values.append(local_agent.last_q_value)

                # Log step metrics (sampled)
                if step % step_sample_rate == 0 and local_session_id:
                    db.log_step_metrics(
                        local_session_id,
                        episode,
                        total_steps,
                        loss=loss,
                        action=action,
                        reward=reward
                    )

                # Emit frame (respecting frame skip for visualization)
                if (
                    local_streamer
                    and is_training
                    and not training_stop_event.is_set()
                    and frame_counter % viz_frame_skip == 0
                ):
                    local_streamer.emit_frame(
                        episode=episode,
                        step=step,
                        reward=episode_reward,
                        epsilon=local_agent.current_epsilon,
                        loss=loss,
                        q_value=local_agent.last_q_value
                    )

                # Speed control
                if training_speed == "1x":
                    time.sleep(0.033)  # ~30 FPS (normal)
                elif training_speed == "2x":
                    time.sleep(0.016)  # ~60 FPS (2x speed)
                elif training_speed == "4x":
                    if frame_counter % 5 == 0:
                        time.sleep(0.001)  # Minimal delay (4x speed)

                # Time-based autosave (every 90 seconds)
                current_time = time.time()
                _record_training_time(local_game, now=current_time)
                if current_time - last_autosave_time >= AUTOSAVE_INTERVAL_SECONDS:
                    if local_agent and local_game:
                        try:
                            path = model_manager.save_checkpoint(
                                local_agent,
                                local_game,
                                episode,
                                episode_reward
                            )
                            last_autosave_time = current_time
                            logger.info(f"Time-based autosave at episode {episode}, step {step}")
                            if path:
                                socketio.emit('model_saved', {'path': path, 'episode': episode})
                            socketio.emit('log', {
                                'message': f'Autosaved at episode {episode}',
                                'type': 'info'
                            })
                        except Exception as e:
                            logger.error(f"Autosave failed: {e}")

                state = next_state

            # Update epsilon at episode end for exploration decay
            if local_agent:
                local_agent.update_epsilon()

            # Episode ended
            episode_duration = int((time.time() - episode_start_time) * 1000)
            session_episode_count += 1
            avg_loss = sum(losses) / len(losses) if losses else 0
            avg_q = sum(q_values) / len(q_values) if q_values else 0
            max_q = max(q_values) if q_values else 0

            # Log to database
            if local_session_id:
                db.log_episode(
                    local_session_id,
                    episode,
                    episode_reward,
                    step,
                    loss=avg_loss,
                    q_value_mean=avg_q,
                    q_value_max=max_q,
                    duration_ms=episode_duration
                )

            # Check for best and auto-save
            if episode_reward > best_episode_reward:
                best_episode_reward = episode_reward

                # Auto-save best model
                if model_manager.should_auto_save(episode):
                    model_manager.save_checkpoint(
                        local_agent,
                        local_game,
                        episode,
                        episode_reward,
                        is_best=True
                    )
            elif model_manager.should_auto_save(episode):
                # Regular auto-save
                model_manager.save_checkpoint(
                    local_agent,
                    local_game,
                    episode,
                    episode_reward
                )

            if is_training and not training_stop_event.is_set():
                epsilon = local_agent.current_epsilon if local_agent else 0
                logger.info(
                    f"Episode {episode}: reward={episode_reward:.1f}, steps={step}, "
                    f"loss={avg_loss:.4f}, ε={epsilon:.3f}"
                )
                action_dist = db.get_action_distribution(local_session_id) if local_session_id else {}
                socketio.emit('episode_end', {
                    'episode': episode,
                    'reward': episode_reward,
                    'steps': step,
                    'loss': round(avg_loss, 4),
                    'qValueMean': round(avg_q, 2),
                    'qValueMax': round(max_q, 2),
                    'duration': episode_duration,
                    'bestReward': best_episode_reward,
                    'epsilon': round(epsilon, 4),
                    'actionDistribution': action_dist
                })

    except Exception as e:
        logger.error(f"Training loop error: {e}")
        import traceback
        traceback.print_exc()
        socketio.emit('error', {'message': str(e)})

    finally:
        is_training = False
        training_stop_event.clear()
        _finalize_training_activity(
            local_game,
            episodes=session_episode_count,
            steps=session_step_count
        )

        if local_session_id and current_session_id == local_session_id:
            db.end_session(local_session_id)

        if active_training_sessions > 0:
            active_training_sessions -= 1

        if local_streamer:
            local_streamer.stop()

        if streamer is local_streamer:
            streamer = None
        if rainbow_agent is local_agent:
            rainbow_agent = None
        if frame_stack is local_frame_stack:
            frame_stack = None
        if current_game == local_game:
            current_game = None
        if current_session_id == local_session_id:
            current_session_id = None

        _trim_memory()

        # Process queue if there are waiting sessions
        if training_queue and active_training_sessions < MAX_CONCURRENT_TRAINING:
            next_request = training_queue.pop(0)
            socketio.emit('queue_ready', {'data': next_request})

        logger.info("Training loop ended")




def run_pretrained_loop():
    """Run an inference-only loop using a pre-trained policy."""
    global is_training, active_training_sessions
    global streamer, pretrained_policy, frame_stack, current_game, current_session_id
    global current_pretrained_model, current_run_mode

    local_streamer = streamer
    local_policy = pretrained_policy
    local_frame_stack = frame_stack
    local_game = current_game
    local_session_id = current_session_id

    if not local_streamer or not local_policy or not local_frame_stack:
        return

    episode = 0
    total_steps = 0
    best_episode_reward = float('-inf')
    session_episode_count = 0
    session_step_count = 0
    frame_counter = 0
    step_sample_rate = 100

    logger.info('Pretrained loop started')

    try:
        while is_training and local_streamer and local_policy and not training_stop_event.is_set():
            if pretrained_policy is not local_policy and pretrained_policy is not None:
                local_policy = pretrained_policy
            episode += 1

            obs, _ = local_streamer.env.reset()
            state = local_frame_stack.reset(obs)
            fire_action = get_fire_action(local_game)
            if fire_action is not None:
                try:
                    obs, _, terminated, truncated, _ = local_streamer.env.step(fire_action)
                    if terminated or truncated:
                        obs, _ = local_streamer.env.reset()
                    state = local_frame_stack.reset(obs)
                except Exception as exc:
                    logger.debug(f'Auto-fire failed for {local_game}: {exc}')

            episode_reward = 0
            step = 0
            done = False
            q_values = []
            episode_start_time = time.time()

            while is_training and local_streamer and not done and not training_stop_event.is_set():
                step += 1
                total_steps += 1
                session_step_count += 1
                frame_counter += 1

                if pretrained_policy is not local_policy and pretrained_policy is not None:
                    local_policy = pretrained_policy
                action = local_policy.select_action(state)
                next_obs, reward, terminated, truncated, _ = local_streamer.env.step(action)
                done = terminated or truncated
                episode_reward += reward
                state = local_frame_stack.push(next_obs)

                if hasattr(local_policy, 'last_q_value'):
                    q_values.append(local_policy.last_q_value)

                if step % step_sample_rate == 0 and local_session_id:
                    db.log_step_metrics(
                        local_session_id,
                        episode,
                        total_steps,
                        loss=None,
                        action=action,
                        reward=reward,
                    )

                if (
                    local_streamer
                    and is_training
                    and not training_stop_event.is_set()
                    and frame_counter % viz_frame_skip == 0
                ):
                    local_streamer.emit_frame(
                        episode=episode,
                        step=step,
                        reward=episode_reward,
                        epsilon=0.0,
                        loss=0.0,
                        q_value=getattr(local_policy, 'last_q_value', None),
                    )

                if training_speed == '1x':
                    time.sleep(0.033)
                elif training_speed == '2x':
                    time.sleep(0.016)
                elif training_speed == '4x':
                    if frame_counter % 5 == 0:
                        time.sleep(0.001)

                _record_training_time(local_game)

            episode_duration = int((time.time() - episode_start_time) * 1000)
            session_episode_count += 1
            avg_q = sum(q_values) / len(q_values) if q_values else 0
            max_q = max(q_values) if q_values else 0

            if local_session_id:
                db.log_episode(
                    local_session_id,
                    episode,
                    episode_reward,
                    step,
                    loss=None,
                    q_value_mean=avg_q,
                    q_value_max=max_q,
                    epsilon=0.0,
                    duration_ms=episode_duration,
                )

            if episode_reward > best_episode_reward:
                best_episode_reward = episode_reward

            if is_training and not training_stop_event.is_set():
                action_dist = db.get_action_distribution(local_session_id) if local_session_id else {}
                socketio.emit('episode_end', {
                    'episode': episode,
                    'reward': episode_reward,
                    'steps': step,
                    'loss': 0,
                    'qValueMean': round(avg_q, 2),
                    'qValueMax': round(max_q, 2),
                    'duration': episode_duration,
                    'bestReward': best_episode_reward,
                    'epsilon': 0,
                    'actionDistribution': action_dist,
                })

    except Exception as e:
        logger.error(f'Pretrained loop error: {e}')
        import traceback
        traceback.print_exc()
        socketio.emit('error', {'message': str(e)})

    finally:
        is_training = False
        training_stop_event.clear()
        _finalize_training_activity(
            local_game,
            episodes=session_episode_count,
            steps=session_step_count,
        )

        if local_session_id and current_session_id == local_session_id:
            db.end_session(local_session_id)

        if active_training_sessions > 0:
            active_training_sessions -= 1

        if local_streamer:
            local_streamer.stop()

        if streamer is local_streamer:
            streamer = None
        if pretrained_policy is local_policy:
            pretrained_policy = None
        if frame_stack is local_frame_stack:
            frame_stack = None
        if current_game == local_game:
            current_game = None
        if current_session_id == local_session_id:
            current_session_id = None
        if current_pretrained_model and current_run_mode == 'pretrained':
            current_pretrained_model = None
        current_run_mode = 'train'

        _trim_memory()

        if training_queue and active_training_sessions < MAX_CONCURRENT_TRAINING:
            next_request = training_queue.pop(0)
            socketio.emit('queue_ready', {'data': next_request})

        logger.info('Pretrained loop ended')

# ============== Main ==============

def run_server(host='0.0.0.0', port=5001, debug=False):
    """Run the server."""
    logger.info(f"Starting server on http://{host}:{port}")
    socketio.run(app, host=host, port=port, debug=debug, allow_unsafe_werkzeug=True)


if __name__ == '__main__':
    run_server(debug=False)
