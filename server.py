"""
Atari RL Training Server with Rainbow DQN.
Flask-SocketIO server for real-time game streaming and training.
"""

import logging
import os
import threading
import time
from flask import Flask, send_from_directory, jsonify, request, send_file
from flask_socketio import SocketIO, emit
from flask_cors import CORS

from game_environments import GameEnvironments
from frame_streamer import FrameStreamer
from rainbow_agent import RainbowAgent, FrameStack, get_device
from model_manager import ModelManager
from db_manager import TrainingDatabase
from config import AUTOSAVE_INTERVAL_SECONDS

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Frontend build location
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FRONTEND_BUILD_DIR = os.environ.get(
    "FRONTEND_BUILD_DIR",
    os.path.join(BASE_DIR, "workspace", "reinforcement-learning", "atari-v2", "frontend", "build"),
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
    ping_interval=25
)

# Global state
game_envs = GameEnvironments()
model_manager = ModelManager()
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

# Watch mode state
watch_mode_active = False
watch_mode_game = None
watch_mode_thread = None

# Speed control settings
training_speed = "1x"  # 1x, 2x, 4x
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
            "workspace/reinforcement-learning/atari-v2/frontend or set "
            "FRONTEND_BUILD_DIR to the build output."
        ),
    }), 404


def _send_frontend_index():
    response = send_from_directory(FRONTEND_BUILD_DIR, 'index.html', conditional=False)
    response.headers["Cache-Control"] = "no-store, max-age=0"
    response.headers["Pragma"] = "no-cache"
    response.headers["Expires"] = "0"
    return response


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
    
    # Send initial state
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
    
    # Get device info
    device = get_device()
    
    emit('init', {
        'games': games,
        'isTraining': is_training,
        'currentGame': current_game,
        'sessionId': current_session_id,
        'savedModels': model_manager.get_all_games(),
        'device': str(device),
        'trainingSpeed': training_speed,
        'vizFrameSkip': viz_frame_skip,
        'vizTargetFps': viz_target_fps
    })


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
    global active_training_sessions, viz_frame_skip, viz_target_fps
    
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

    if isinstance(load_checkpoint, str) and load_checkpoint.strip() == "":
        load_checkpoint = None

    if isinstance(resume_from_saved, str):
        resume_from_saved = resume_from_saved.strip().lower() in {"1", "true", "yes", "on"}
    
    if not raw_game_id:
        emit('error', {'message': 'No game specified'})
        return
    
    game_id = resolve_game_id(raw_game_id)
    if game_id != raw_game_id:
        logger.info(f"Resolved game id '{raw_game_id}' -> '{game_id}'")

    if game_id not in game_envs.games:
        emit('error', {'message': f'Unknown game: {raw_game_id}'})
        return

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
        
        # Create frame stacker
        frame_stack = FrameStack(num_frames=4, frame_size=(84, 84))
        
        # Create Rainbow agent
        device = get_device()
        buffer_size = _env_int("RL_BUFFER_SIZE", 100000)
        min_buffer_size = _env_int("RL_MIN_BUFFER_SIZE", 1000)
        batch_size = _env_int("RL_BATCH_SIZE", 32)
        n_step = _env_int("RL_N_STEP", 3)
        store_uint8 = _env_bool("RL_STORE_UINT8", False)
        if min_buffer_size > buffer_size:
            logger.warning(
                "RL_MIN_BUFFER_SIZE %s exceeds RL_BUFFER_SIZE %s; clamping.",
                min_buffer_size,
                buffer_size,
            )
            min_buffer_size = buffer_size
        logger.info(
            "Training config: buffer_size=%s min_buffer_size=%s batch_size=%s n_step=%s store_uint8=%s",
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
                    logger.info(f"Loaded checkpoint: {checkpoint_to_load}")
                    emit('log', {'message': f'Loaded checkpoint: {checkpoint_to_load}', 'type': 'success'})
                else:
                    best_path = model_manager.get_best_model_path(game_id)
                    if best_path:
                        model_manager.load_checkpoint(rainbow_agent, game_id, None)
                        logger.info("Loaded checkpoint: best_model.pt")
                        emit('log', {'message': 'Loaded checkpoint: best_model.pt', 'type': 'success'})
                    else:
                        emit('log', {'message': 'Starting fresh (no checkpoints found)', 'type': 'info'})
            except Exception as e:
                logger.warning(f"Failed to load checkpoint: {e}")
                emit('log', {'message': 'Starting fresh (checkpoint load failed)', 'type': 'warning'})
        
        # Create streamer
        streamer = FrameStreamer(env, socketio, target_fps=viz_target_fps)
        
        # Create database session
        current_session_id = db.create_session(
            game_id=game_id,
            device=str(device),
            hyperparameters=rainbow_agent.get_hyperparameters()
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
            'vizFrameSkip': viz_frame_skip,
            'vizTargetFps': viz_target_fps
        })
        socketio.emit('status', {'isTraining': True, 'game': game_id})
        
        # Start training in background thread
        training_thread = threading.Thread(target=run_training_loop, daemon=True)
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


# ============== Main ==============

def run_server(host='0.0.0.0', port=5001, debug=False):
    """Run the server."""
    logger.info(f"Starting server on http://{host}:{port}")
    socketio.run(app, host=host, port=port, debug=debug, allow_unsafe_werkzeug=True)


if __name__ == '__main__':
    run_server(debug=False)
