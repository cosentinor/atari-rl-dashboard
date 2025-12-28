"""
Frame Streamer for Atari RL Training.
Handles efficient frame encoding and WebSocket streaming.
Supports frame skipping for performance optimization.
"""

import cv2
import numpy as np
import logging
import time
import base64
from typing import Optional
from collections import deque

logger = logging.getLogger(__name__)


class FrameStreamer:
    """
    Efficient frame streaming for real-time game visualization.
    
    Features:
    - JPEG encoding with configurable quality
    - Frame rate throttling
    - Frame skipping for bandwidth optimization
    - FPS tracking
    - Automatic resizing
    """
    
    def __init__(
        self,
        env,
        socketio,
        target_fps: int = 30,
        jpeg_quality: int = 85
    ):
        """
        Initialize the frame streamer.
        
        Args:
            env: Gymnasium environment
            socketio: Flask-SocketIO instance
            target_fps: Target frames per second
            jpeg_quality: JPEG compression quality (1-100)
        """
        self.env = env
        self.socketio = socketio
        self.target_fps = target_fps
        self.jpeg_quality = jpeg_quality
        self.frame_interval = 1.0 / target_fps
        
        self.last_frame_time = 0
        self.frame_count = 0
        self.total_frames = 0
        self.is_running = True
        
        # FPS tracking
        self.fps_window = deque(maxlen=30)
        self.last_fps_update = time.time()
        self.current_fps = 0
        
        # Frame skip tracking
        self.frames_skipped = 0
        
        # JPEG encoding params
        self.encode_params = [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality]
        
        logger.info(f"FrameStreamer initialized: {target_fps}fps, quality={jpeg_quality}")
    
    def emit_frame(
        self,
        episode: int,
        step: int,
        reward: float,
        epsilon: float,
        loss: Optional[float] = None,
        q_value: Optional[float] = None
    ):
        """
        Capture and emit a frame if enough time has passed.
        
        Args:
            episode: Current episode number
            step: Current step in episode
            reward: Current episode reward
            epsilon: Current exploration rate (0 for Rainbow)
            loss: Current training loss
            q_value: Current Q-value
        """
        if not self.is_running:
            return
        
        self.total_frames += 1
        
        # Throttle to target FPS
        current_time = time.time()
        if current_time - self.last_frame_time < self.frame_interval:
            self.frames_skipped += 1
            return
        
        # Update FPS tracking
        self.fps_window.append(current_time)
        if len(self.fps_window) >= 2:
            time_span = self.fps_window[-1] - self.fps_window[0]
            if time_span > 0:
                self.current_fps = (len(self.fps_window) - 1) / time_span
        
        self.last_frame_time = current_time
        
        try:
            # Get frame from environment
            frame = self.env.render()
            if frame is None:
                return
            
            # Encode frame
            frame_data = self.encode_frame(frame)
            if frame_data is None:
                return
            
            self.frame_count += 1
            
            # Build frame payload
            payload = {
                'data': frame_data,
                'episode': episode,
                'step': step,
                'reward': round(reward, 1),
                'epsilon': round(epsilon, 4) if epsilon else 0,
                'fps': round(self.current_fps, 1),
                'frameCount': self.frame_count
            }
            
            # Add optional metrics
            if loss is not None:
                payload['loss'] = round(loss, 6)
            if q_value is not None:
                payload['qValue'] = round(q_value, 2)
            
            # Emit frame data
            self.socketio.emit('frame', payload)
            
        except Exception as e:
            logger.error(f"Frame emission error: {e}")
    
    def encode_frame(self, frame: np.ndarray) -> Optional[str]:
        """
        Encode a frame to JPEG base64.
        
        Args:
            frame: RGB numpy array
            
        Returns:
            Base64 encoded JPEG string with data URL prefix
        """
        try:
            # Ensure frame is uint8
            if frame.dtype != np.uint8:
                if frame.max() <= 1.0:
                    frame = (frame * 255).astype(np.uint8)
                else:
                    frame = frame.astype(np.uint8)
            
            # Convert RGB to BGR for OpenCV
            if len(frame.shape) == 3 and frame.shape[2] == 3:
                frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            else:
                frame_bgr = frame
            
            # Encode to JPEG
            success, buffer = cv2.imencode('.jpg', frame_bgr, self.encode_params)
            if not success:
                return None
            
            # Convert to base64 data URL
            b64_data = base64.b64encode(buffer).decode('utf-8')
            return f"data:image/jpeg;base64,{b64_data}"
            
        except Exception as e:
            logger.error(f"Frame encoding error: {e}")
            return None
    
    def set_quality(self, quality: int):
        """Update JPEG quality."""
        self.jpeg_quality = max(1, min(100, quality))
        self.encode_params = [cv2.IMWRITE_JPEG_QUALITY, self.jpeg_quality]
        logger.info(f"JPEG quality set to {self.jpeg_quality}")
    
    def set_target_fps(self, fps: int):
        """Update target FPS."""
        self.target_fps = max(1, min(60, fps))
        self.frame_interval = 1.0 / self.target_fps
        logger.info(f"Target FPS set to {self.target_fps}")
    
    def stop(self):
        """Stop the streamer and clean up."""
        self.is_running = False
        if self.env:
            try:
                self.env.close()
            except Exception:
                pass
        logger.info(f"FrameStreamer stopped. Total frames: {self.frame_count}, Skipped: {self.frames_skipped}")
    
    def get_stats(self) -> dict:
        """Get streaming statistics."""
        return {
            'frame_count': self.frame_count,
            'total_frames': self.total_frames,
            'frames_skipped': self.frames_skipped,
            'target_fps': self.target_fps,
            'current_fps': round(self.current_fps, 1),
            'jpeg_quality': self.jpeg_quality,
            'is_running': self.is_running
        }
