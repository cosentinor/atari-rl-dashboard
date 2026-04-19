# Multi-User Implementation Plan

## Problem
The original server.py used global variables for training state, making it impossible for multiple users to train simultaneously.

## Solution Architecture

### Phase 1: Session Infrastructure ✅
- [x] Create TrainingSession class
- [x] Add session management helpers (get_or_create_session, get_session, cleanup_session)
- [x] Update FrameStreamer to support rooms
- [x] Update connect/disconnect handlers

### Phase 2: Core Handler Refactor ✅
- [x] Refactor `handle_start_training` to be session-aware
- [x] Refactor `handle_stop_training`
- [x] Refactor `handle_save_model`
- [x] Refactor `handle_set_training_speed`
- [x] Refactor `handle_set_viz_speed`
- [x] Refactor `handle_set_training_level`
- [x] Refactor `handle_load_model`
- [x] Refactor `handle_get_init` and `handle_get_status`

### Phase 3: Training Loop Refactor ✅
- [x] Create `run_training_loop_session` that accepts session parameter
- [x] Create `run_pretrained_loop_session` that accepts session parameter
- [x] Update all emit() calls in loops to use room parameter

### Phase 4: Helper Functions ✅
- [x] Create `_start_training_activity_session`
- [x] Create `_record_training_time_session`
- [x] Create `_finalize_training_activity_session`
- [x] Create `_build_checkpoint_metadata_session`
- [x] Create `_stop_training_session`

### Phase 5: Testing
- [ ] Test single user (regression)
- [ ] Test multiple users simultaneously
- [ ] Test disconnect behavior
- [ ] Test resource limits (MAX_CONCURRENT_TRAINING)

## Key Changes

### TrainingSession Class
Each connected user gets their own `TrainingSession` object that holds:
- Socket ID and room name
- Training state (streamer, thread, agent, frame_stack)
- Game state (current_game, db_session_id)
- Speed/visualization settings
- Stop event for graceful termination

### Session Management
- Sessions stored in `training_sessions` dict, keyed by socket ID
- `MAX_CONCURRENT_TRAINING = 5` limits total active training sessions
- `SESSION_TIMEOUT = 3600` (1 hour) for idle session cleanup
- Sessions auto-cleanup on disconnect

### Socket.IO Rooms
- Each session has its own room (`session_{socket_sid}`)
- Frames, logs, and events are emitted only to the session's room
- No more global broadcasts for training events

## Migration Notes

### Legacy Code
The original global-state training loops (`run_training_loop`, `run_pretrained_loop`) are kept for backward compatibility but are no longer used. They can be removed in a future cleanup.

### Frontend Compatibility
The frontend should work without changes since:
- The same events are emitted (`frame`, `episode_end`, `training_started`, etc.)
- The init payload includes the same fields
- Each user sees only their own training session

## Configuration

```python
MAX_CONCURRENT_TRAINING = 5  # Max simultaneous training sessions
SESSION_TIMEOUT = 3600       # 1 hour - cleanup idle sessions
```

## Benefits
1. Multiple users can train different games simultaneously
2. Each user's training is isolated - stopping one doesn't affect others
3. Resource limits prevent server overload
4. Graceful handling of disconnects and timeouts
