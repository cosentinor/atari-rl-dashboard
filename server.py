"""
Atari RL Training Server with Rainbow DQN.
Flask-SocketIO server for real-time game streaming and training.
"""

import logging
import threading
import time
from datetime import datetime
from flask import Flask, send_from_directory, jsonify, request
from flask_socketio import SocketIO, emit
from flask_cors import CORS

from game_environments import GameEnvironments
from frame_streamer import FrameStreamer
from rainbow_agent import RainbowAgent, FrameStack, get_device
from model_manager import ModelManager
from db_manager import TrainingDatabase
from config import AUTOSAVE_INTERVAL_SECONDS, AUTOSAVE_INTERVAL_EPISODES

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Flask app
app = Flask(__name__, static_folder='frontend')
CORS(app)
socketio = SocketIO(app, cors_allowed_origins="*", async_mode='threading')

# Global state
game_envs = GameEnvironments()
model_manager = ModelManager()
db = TrainingDatabase()

# Training state
streamer = None
training_thread = None
is_training = False
current_game = None
current_session_id = None
rainbow_agent = None
frame_stack = None

# Speed control settings
training_speed = "normal"  # normal, fast, turbo
viz_frame_skip = 1  # 1 = every frame, 2 = every 2nd, etc.


# ============== HTTP Routes ==============

@app.route('/')
def index():
    """Serve the React frontend."""
    return send_from_directory('frontend', 'index.html')


@app.route('/<path:filename>')
def serve_static(filename):
    """Serve static files from frontend folder."""
    return send_from_directory('frontend', filename)


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


@app.route('/api/leaderboard')
def get_leaderboard():
    """Get training leaderboard."""
    game_id = request.args.get('game_id')
    return jsonify({
        'success': True,
        'leaderboard': db.get_leaderboard(game_id)
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


# ============== WebSocket Events ==============

@socketio.on('connect')
def handle_connect():
    """Handle new client connection."""
    logger.info("Client connected")
    
    # Send initial state
    games = [{'id': g.id, 'name': g.display_name} for g in game_envs.games.values() if g.is_available]
    
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
        'vizFrameSkip': viz_frame_skip
    })


@socketio.on('disconnect')
def handle_disconnect():
    """Handle client disconnect."""
    global is_training, streamer
    logger.info("Client disconnected")
    
    # Stop training on disconnect as requested
    if is_training:
        logger.info("Stopping training due to client disconnect")
        is_training = False
        
        if current_session_id:
            db.end_session(current_session_id)
        
        if streamer:
            streamer.stop()
            streamer = None
        
        socketio.emit('training_stopped', {})
        socketio.emit('status', {'isTraining': False, 'game': None})


@socketio.on('start_training')
def handle_start_training(data):
    """Start training for a game."""
    global is_training, current_game, current_session_id
    global streamer, training_thread, rainbow_agent, frame_stack
    
    logger.info(f"Received start_training event with data: {data}")
    
    if is_training:
        emit('error', {'message': 'Training already in progress'})
        return
    
    game_id = data.get('game')
    load_checkpoint = data.get('loadCheckpoint')  # Optional checkpoint to load
    
    if not game_id:
        emit('error', {'message': 'No game specified'})
        return
    
    logger.info(f"Starting training for game_id: {game_id}")
    
    try:
        # Create environment
        env = game_envs.create_environment(game_id)
        
        # Get action space size
        num_actions = env.action_space.n
        
        # Create frame stacker
        frame_stack = FrameStack(num_frames=4, frame_size=(84, 84))
        
        # Create Rainbow agent
        device = get_device()
        rainbow_agent = RainbowAgent(
            state_shape=(4, 84, 84),
            num_actions=num_actions,
            device=device
        )
        
        # Load checkpoint if specified
        if load_checkpoint:
            try:
                checkpoint = model_manager.load_checkpoint(
                    rainbow_agent, game_id, load_checkpoint
                )
                logger.info(f"Loaded checkpoint: {load_checkpoint}")
                emit('log', {'message': f'Loaded checkpoint: {load_checkpoint}', 'type': 'success'})
            except Exception as e:
                logger.warning(f"Failed to load checkpoint: {e}")
                emit('log', {'message': f'Starting fresh (checkpoint load failed)', 'type': 'warning'})
        
        # Create streamer
        streamer = FrameStreamer(env, socketio)
        
        # Create database session
        current_session_id = db.create_session(
            game_id=game_id,
            device=str(device),
            hyperparameters=rainbow_agent.get_hyperparameters()
        )
        
        current_game = game_id
        is_training = True
        
        # Notify client
        emit('training_started', {
            'game': game_id,
            'sessionId': current_session_id,
            'device': str(device)
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


@socketio.on('stop_training')
def handle_stop_training():
    """Stop current training."""
    global is_training, streamer
    
    logger.info("Stopping training")
    is_training = False
    
    # End database session
    if current_session_id:
        db.end_session(current_session_id)
    
    if streamer:
        streamer.stop()
        streamer = None
    
    socketio.emit('training_stopped', {})
    socketio.emit('status', {'isTraining': False, 'game': None})


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
    speed = data.get('speed', 'normal')
    
    if speed in ['normal', 'fast', 'turbo']:
        training_speed = speed
        logger.info(f"Training speed set to: {speed}")
        socketio.emit('speed_changed', {'trainingSpeed': speed})


@socketio.on('set_viz_speed')
def handle_set_viz_speed(data):
    """Set visualization frame skip."""
    global viz_frame_skip
    skip = data.get('frameSkip', 1)
    
    if isinstance(skip, int) and 1 <= skip <= 100:
        viz_frame_skip = skip
        logger.info(f"Visualization frame skip set to: {skip}")
        socketio.emit('speed_changed', {'vizFrameSkip': skip})


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

def run_training_loop():
    """Main training loop running in background thread."""
    global is_training
    
    if not streamer or not rainbow_agent:
        return
    
    episode = rainbow_agent.episode_count
    total_steps = rainbow_agent.step_count
    best_episode_reward = float('-inf')
    
    # Metrics for emission
    episode_start_time = time.time()
    frame_counter = 0
    step_sample_rate = 100  # Log step metrics every N steps
    
    # Time-based autosave tracking
    last_autosave_time = time.time()
    
    logger.info("Training loop started")
    
    try:
        while is_training and streamer and rainbow_agent:
            episode += 1
            rainbow_agent.episode_count = episode
            
            # Reset environment
            obs, _ = streamer.env.reset()
            state = frame_stack.reset(obs)
            
            episode_reward = 0
            step = 0
            done = False
            episode_start_time = time.time()
            losses = []
            q_values = []
            
            while is_training and streamer and not done:
                step += 1
                total_steps += 1
                frame_counter += 1
                
                # Select action using Rainbow agent
                action = rainbow_agent.select_action(state, training=True)
                
                # Step environment
                next_obs, reward, terminated, truncated, info = streamer.env.step(action)
                done = terminated or truncated
                episode_reward += reward
                
                # Process next state
                next_state = frame_stack.push(next_obs)
                
                # Store transition
                rainbow_agent.push_transition(state, action, reward, next_state, done)
                
                # Learn
                loss = rainbow_agent.learn()
                if loss is not None:
                    losses.append(loss)
                
                # Track Q-values
                q_values.append(rainbow_agent.last_q_value)
                
                # Log step metrics (sampled)
                if step % step_sample_rate == 0 and current_session_id:
                    db.log_step_metrics(
                        current_session_id,
                        episode,
                        total_steps,
                        loss=loss,
                        action=action,
                        reward=reward
                    )
                
                # Emit frame (respecting frame skip for visualization)
                # Check streamer is still valid (race condition fix)
                if streamer and is_training and frame_counter % viz_frame_skip == 0:
                    streamer.emit_frame(
                        episode=episode,
                        step=step,
                        reward=episode_reward,
                        epsilon=rainbow_agent.current_epsilon,
                        loss=loss,
                        q_value=rainbow_agent.last_q_value
                    )
                
                # Speed control
                if training_speed == "normal":
                    time.sleep(0.033)  # ~30 FPS
                elif training_speed == "fast":
                    if frame_counter % 10 == 0:
                        time.sleep(0.001)  # Minimal delay
                # turbo = no delay
                
                # Time-based autosave (every 90 seconds)
                current_time = time.time()
                if current_time - last_autosave_time >= AUTOSAVE_INTERVAL_SECONDS:
                    if rainbow_agent and current_game:
                        try:
                            model_manager.save_checkpoint(
                                rainbow_agent,
                                current_game,
                                episode,
                                episode_reward
                            )
                            last_autosave_time = current_time
                            logger.info(f"Time-based autosave at episode {episode}, step {step}")
                            socketio.emit('log', {
                                'message': f'Autosaved at episode {episode}',
                                'type': 'info'
                            })
                        except Exception as e:
                            logger.error(f"Autosave failed: {e}")
                
                state = next_state
            
            # Update epsilon at episode end for exploration decay
            if rainbow_agent:
                rainbow_agent.update_epsilon()
            
            # Episode ended
            episode_duration = int((time.time() - episode_start_time) * 1000)
            avg_loss = sum(losses) / len(losses) if losses else 0
            avg_q = sum(q_values) / len(q_values) if q_values else 0
            max_q = max(q_values) if q_values else 0
            
            # Log to database
            if current_session_id:
                db.log_episode(
                    current_session_id,
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
                        rainbow_agent,
                        current_game,
                        episode,
                        episode_reward,
                        is_best=True
                    )
            elif model_manager.should_auto_save(episode):
                # Regular auto-save
                model_manager.save_checkpoint(
                    rainbow_agent,
                    current_game,
                    episode,
                    episode_reward
                )
            
            if is_training:
                epsilon = rainbow_agent.current_epsilon if rainbow_agent else 0
                logger.info(f"Episode {episode}: reward={episode_reward:.1f}, steps={step}, loss={avg_loss:.4f}, ε={epsilon:.3f}")
                socketio.emit('episode_end', {
                    'episode': episode,
                    'reward': episode_reward,
                    'steps': step,
                    'loss': round(avg_loss, 4),
                    'qValueMean': round(avg_q, 2),
                    'qValueMax': round(max_q, 2),
                    'duration': episode_duration,
                    'bestReward': best_episode_reward,
                    'epsilon': round(epsilon, 4)
                })
    
    except Exception as e:
        logger.error(f"Training loop error: {e}")
        import traceback
        traceback.print_exc()
        socketio.emit('error', {'message': str(e)})
    
    finally:
        is_training = False
        if streamer:
            streamer.stop()
        logger.info("Training loop ended")


# ============== Main ==============

def run_server(host='0.0.0.0', port=5001, debug=False):
    """Run the server."""
    logger.info(f"Starting server on http://{host}:{port}")
    socketio.run(app, host=host, port=port, debug=debug, allow_unsafe_werkzeug=True)


if __name__ == '__main__':
    run_server(debug=False)
