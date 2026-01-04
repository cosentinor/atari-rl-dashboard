"""
Rainbow DQN Agent for Atari Games.
Implements all 6 Rainbow components:
1. Double DQN - Use online network for action selection
2. Dueling Network - Separate value and advantage streams
3. Prioritized Experience Replay - TD-error based sampling
4. Multi-step Learning - N-step returns (default n=3)
5. Distributional RL - C51 categorical distribution
6. Noisy Networks - Parametric noise for exploration
"""

import math
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from typing import Dict, List, Tuple, Optional, Any
from collections import deque
import logging

logger = logging.getLogger(__name__)


# ============== Device Detection ==============

def get_device() -> torch.device:
    """Auto-detect best available device (CUDA > MPS > CPU)."""
    if torch.cuda.is_available():
        device = torch.device("cuda")
        logger.info(f"Using CUDA: {torch.cuda.get_device_name(0)}")
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device = torch.device("mps")
        logger.info("Using Apple MPS (Metal Performance Shaders)")
    else:
        device = torch.device("cpu")
        logger.info("Using CPU")
    return device


# ============== Noisy Linear Layer ==============

class NoisyLinear(nn.Module):
    """
    Noisy Linear Layer for exploration.
    Uses factorized Gaussian noise for efficiency.
    """
    
    def __init__(self, in_features: int, out_features: int, std_init: float = 0.5):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.std_init = std_init
        
        # Learnable parameters
        self.weight_mu = nn.Parameter(torch.empty(out_features, in_features))
        self.weight_sigma = nn.Parameter(torch.empty(out_features, in_features))
        self.bias_mu = nn.Parameter(torch.empty(out_features))
        self.bias_sigma = nn.Parameter(torch.empty(out_features))
        
        # Factorized noise buffers
        self.register_buffer('weight_epsilon', torch.empty(out_features, in_features))
        self.register_buffer('bias_epsilon', torch.empty(out_features))
        
        self.reset_parameters()
        self.reset_noise()
    
    def reset_parameters(self):
        """Initialize parameters."""
        mu_range = 1 / math.sqrt(self.in_features)
        self.weight_mu.data.uniform_(-mu_range, mu_range)
        self.weight_sigma.data.fill_(self.std_init / math.sqrt(self.in_features))
        self.bias_mu.data.uniform_(-mu_range, mu_range)
        self.bias_sigma.data.fill_(self.std_init / math.sqrt(self.out_features))
    
    def _scale_noise(self, size: int) -> torch.Tensor:
        """Generate scaled noise using factorization."""
        x = torch.randn(size, device=self.weight_mu.device)
        return x.sign().mul_(x.abs().sqrt_())
    
    def reset_noise(self):
        """Sample new noise."""
        epsilon_in = self._scale_noise(self.in_features)
        epsilon_out = self._scale_noise(self.out_features)
        self.weight_epsilon.copy_(epsilon_out.outer(epsilon_in))
        self.bias_epsilon.copy_(epsilon_out)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with noisy weights."""
        if self.training:
            weight = self.weight_mu + self.weight_sigma * self.weight_epsilon
            bias = self.bias_mu + self.bias_sigma * self.bias_epsilon
        else:
            weight = self.weight_mu
            bias = self.bias_mu
        return F.linear(x, weight, bias)


# ============== Rainbow Network ==============

class RainbowNetwork(nn.Module):
    """
    Rainbow DQN Network.
    Combines Dueling architecture with Noisy layers and Distributional output.
    """
    
    def __init__(
        self,
        input_shape: Tuple[int, int, int],
        num_actions: int,
        num_atoms: int = 51,
        v_min: float = -10.0,
        v_max: float = 10.0,
        noisy_std: float = 0.5
    ):
        super().__init__()
        self.num_actions = num_actions
        self.num_atoms = num_atoms
        self.v_min = v_min
        self.v_max = v_max
        
        # Support for distributional RL
        self.register_buffer(
            'support',
            torch.linspace(v_min, v_max, num_atoms)
        )
        self.delta_z = (v_max - v_min) / (num_atoms - 1)
        
        # CNN feature extractor (Nature DQN architecture)
        self.conv = nn.Sequential(
            nn.Conv2d(input_shape[0], 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU()
        )
        
        # Calculate conv output size
        conv_out_size = self._get_conv_out_size(input_shape)
        
        # Dueling streams with Noisy layers
        # Value stream
        self.value_hidden = NoisyLinear(conv_out_size, 512, noisy_std)
        self.value_out = NoisyLinear(512, num_atoms, noisy_std)
        
        # Advantage stream
        self.advantage_hidden = NoisyLinear(conv_out_size, 512, noisy_std)
        self.advantage_out = NoisyLinear(512, num_actions * num_atoms, noisy_std)
    
    def _get_conv_out_size(self, shape: Tuple[int, int, int]) -> int:
        """Calculate the output size of conv layers."""
        with torch.no_grad():
            dummy = torch.zeros(1, *shape)
            return self.conv(dummy).view(1, -1).size(1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass returning Q-value distributions.
        
        Returns:
            Tensor of shape (batch, actions, atoms)
        """
        batch_size = x.size(0)
        
        # Feature extraction
        features = self.conv(x).view(batch_size, -1)
        
        # Value stream
        value = F.relu(self.value_hidden(features))
        value = self.value_out(value).view(batch_size, 1, self.num_atoms)
        
        # Advantage stream
        advantage = F.relu(self.advantage_hidden(features))
        advantage = self.advantage_out(advantage).view(batch_size, self.num_actions, self.num_atoms)
        
        # Combine streams (dueling)
        q_atoms = value + advantage - advantage.mean(dim=1, keepdim=True)
        
        # Apply softmax to get probability distributions
        q_dist = F.softmax(q_atoms, dim=2)
        
        return q_dist
    
    def get_q_values(self, x: torch.Tensor) -> torch.Tensor:
        """Get Q-values by computing expected value of distributions."""
        q_dist = self.forward(x)
        q_values = (q_dist * self.support).sum(dim=2)
        return q_values
    
    def reset_noise(self):
        """Reset noise in all noisy layers."""
        self.value_hidden.reset_noise()
        self.value_out.reset_noise()
        self.advantage_hidden.reset_noise()
        self.advantage_out.reset_noise()


# ============== Sum Tree for Prioritized Replay ==============

class SumTree:
    """
    Sum Tree data structure for efficient prioritized sampling.
    Allows O(log n) updates and sampling.
    """
    
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.tree = np.zeros(2 * capacity - 1)
        self.data = np.zeros(capacity, dtype=object)
        self.write_idx = 0
        self.n_entries = 0
    
    def _propagate(self, idx: int, change: float):
        """Propagate priority change up the tree."""
        parent = (idx - 1) // 2
        self.tree[parent] += change
        if parent != 0:
            self._propagate(parent, change)
    
    def _retrieve(self, idx: int, s: float) -> int:
        """Find sample index for given priority sum."""
        left = 2 * idx + 1
        right = left + 1
        
        if left >= len(self.tree):
            return idx
        
        if s <= self.tree[left]:
            return self._retrieve(left, s)
        else:
            return self._retrieve(right, s - self.tree[left])
    
    def total(self) -> float:
        """Get total priority sum."""
        return self.tree[0]
    
    def add(self, priority: float, data: Any):
        """Add new data with given priority."""
        idx = self.write_idx + self.capacity - 1
        self.data[self.write_idx] = data
        self.update(idx, priority)
        
        self.write_idx = (self.write_idx + 1) % self.capacity
        self.n_entries = min(self.n_entries + 1, self.capacity)
    
    def update(self, idx: int, priority: float):
        """Update priority at given index."""
        change = priority - self.tree[idx]
        self.tree[idx] = priority
        self._propagate(idx, change)
    
    def get(self, s: float) -> Tuple[int, float, Any]:
        """Get data for given priority sum."""
        idx = self._retrieve(0, s)
        data_idx = idx - self.capacity + 1
        return idx, self.tree[idx], self.data[data_idx]


# ============== Prioritized Replay Buffer ==============

class PrioritizedReplayBuffer:
    """
    Prioritized Experience Replay Buffer using Sum Tree.
    Supports n-step returns.
    """
    
    def __init__(
        self,
        capacity: int,
        alpha: float = 0.6,
        beta_start: float = 0.4,
        beta_frames: int = 100000,
        n_step: int = 3,
        gamma: float = 0.99,
        store_uint8: bool = False
    ):
        self.capacity = capacity
        self.alpha = alpha  # Prioritization exponent
        self.beta_start = beta_start
        self.beta_frames = beta_frames
        self.n_step = n_step
        self.gamma = gamma
        self.store_uint8 = store_uint8
        
        self.tree = SumTree(capacity)
        self.max_priority = 1.0
        self.frame_count = 0
        
        # N-step buffer
        self.n_step_buffer = deque(maxlen=n_step)

    def _maybe_quantize(self, state: np.ndarray) -> np.ndarray:
        """Optionally store states as uint8 to reduce memory usage."""
        if not self.store_uint8:
            return state
        if state.dtype == np.uint8:
            return state
        return np.clip(state * 255.0, 0, 255).astype(np.uint8)
    
    def _get_beta(self) -> float:
        """Anneal beta from beta_start to 1.0."""
        return min(1.0, self.beta_start + self.frame_count * (1.0 - self.beta_start) / self.beta_frames)
    
    def _compute_n_step_return(self) -> Tuple[np.ndarray, float, np.ndarray, bool]:
        """Compute n-step return from buffer."""
        # Get first and last transitions
        first = self.n_step_buffer[0]
        last = self.n_step_buffer[-1]
        
        # Compute n-step return
        n_step_return = 0.0
        for i, (_, _, reward, _, _) in enumerate(self.n_step_buffer):
            n_step_return += (self.gamma ** i) * reward
        
        return first[0], n_step_return, last[3], last[4]
    
    def push(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool
    ):
        """Add transition to buffer."""
        self.frame_count += 1
        
        # Add to n-step buffer
        self.n_step_buffer.append((state, action, reward, next_state, done))
        
        # Only add to main buffer when n-step buffer is full or episode ends
        if len(self.n_step_buffer) < self.n_step and not done:
            return
        
        # Compute n-step transition
        state_0, n_step_return, next_state_n, done_n = self._compute_n_step_return()
        action_0 = self.n_step_buffer[0][1]
        
        # Store with max priority
        state_0 = self._maybe_quantize(state_0)
        next_state_n = self._maybe_quantize(next_state_n)
        transition = (state_0, action_0, n_step_return, next_state_n, done_n)
        self.tree.add(self.max_priority ** self.alpha, transition)
        
        # Clear buffer if episode ended
        if done:
            self.n_step_buffer.clear()
    
    def sample(self, batch_size: int) -> Tuple[torch.Tensor, ...]:
        """Sample batch with priorities."""
        indices = []
        priorities = []
        samples = []
        
        segment = self.tree.total() / batch_size
        beta = self._get_beta()
        
        for i in range(batch_size):
            a = segment * i
            b = segment * (i + 1)
            s = random.uniform(a, b)
            
            idx, priority, data = self.tree.get(s)
            indices.append(idx)
            priorities.append(priority)
            samples.append(data)
        
        # Compute importance sampling weights
        probs = np.array(priorities) / self.tree.total()
        weights = (self.tree.n_entries * probs) ** (-beta)
        weights = weights / weights.max()
        
        # Unpack samples
        states = np.array([s[0] for s in samples])
        actions = np.array([s[1] for s in samples])
        rewards = np.array([s[2] for s in samples])
        next_states = np.array([s[3] for s in samples])
        dones = np.array([s[4] for s in samples])

        if self.store_uint8:
            states = states.astype(np.float32) / 255.0
            next_states = next_states.astype(np.float32) / 255.0
        
        return (
            torch.FloatTensor(states),
            torch.LongTensor(actions),
            torch.FloatTensor(rewards),
            torch.FloatTensor(next_states),
            torch.FloatTensor(dones),
            torch.FloatTensor(weights),
            indices
        )
    
    def update_priorities(self, indices: List[int], priorities: np.ndarray):
        """Update priorities after learning."""
        for idx, priority in zip(indices, priorities):
            # Clamp priority to avoid zero
            priority = max(priority, 1e-6)
            self.max_priority = max(self.max_priority, priority)
            self.tree.update(idx, priority ** self.alpha)
    
    def __len__(self) -> int:
        return self.tree.n_entries


# ============== Rainbow Agent ==============

class RainbowAgent:
    """
    Full Rainbow DQN Agent.
    Combines all 6 improvements over vanilla DQN.
    """
    
    def __init__(
        self,
        state_shape: Tuple[int, int, int],
        num_actions: int,
        device: Optional[torch.device] = None,
        # Hyperparameters
        lr: float = 6.25e-5,
        gamma: float = 0.99,
        batch_size: int = 32,
        buffer_size: int = 100000,
        min_buffer_size: int = 1000,
        target_update_freq: int = 1000,
        # Distributional
        num_atoms: int = 51,
        v_min: float = -10.0,
        v_max: float = 10.0,
        # Prioritized replay
        alpha: float = 0.5,
        beta_start: float = 0.4,
        beta_frames: int = 100000,
        # N-step
        n_step: int = 3,
        # Memory optimization
        store_uint8: bool = False,
        # Noisy nets - increased for better exploration
        noisy_std: float = 0.7,
        # Early exploration fallback
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.01,
        epsilon_decay_episodes: int = 50
    ):
        self.device = device or get_device()
        self.num_actions = num_actions
        self.gamma = gamma
        self.batch_size = batch_size
        self.min_buffer_size = min_buffer_size
        self.target_update_freq = target_update_freq
        self.n_step = n_step
        self.store_uint8 = store_uint8
        
        # Epsilon-greedy fallback for early exploration
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay_episodes = epsilon_decay_episodes
        self.current_epsilon = epsilon_start
        
        # Distributional parameters
        self.num_atoms = num_atoms
        self.v_min = v_min
        self.v_max = v_max
        self.delta_z = (v_max - v_min) / (num_atoms - 1)
        self.support = torch.linspace(v_min, v_max, num_atoms).to(self.device)
        
        # Networks
        self.online_net = RainbowNetwork(
            state_shape, num_actions, num_atoms, v_min, v_max, noisy_std
        ).to(self.device)
        
        self.target_net = RainbowNetwork(
            state_shape, num_actions, num_atoms, v_min, v_max, noisy_std
        ).to(self.device)
        self.target_net.load_state_dict(self.online_net.state_dict())
        self.target_net.eval()
        
        # Optimizer
        self.optimizer = optim.Adam(self.online_net.parameters(), lr=lr, eps=1.5e-4)
        
        # Replay buffer
        self.buffer = PrioritizedReplayBuffer(
            buffer_size, alpha, beta_start, beta_frames, n_step, gamma, store_uint8=store_uint8
        )
        
        # Training state
        self.step_count = 0
        self.episode_count = 0
        self.last_loss = 0.0
        self.last_q_value = 0.0
        
        logger.info(f"RainbowAgent initialized on {self.device}")
        logger.info(f"  State shape: {state_shape}, Actions: {num_actions}")
        logger.info(f"  Atoms: {num_atoms}, V_range: [{v_min}, {v_max}]")
        logger.info(f"  Epsilon: {epsilon_start} -> {epsilon_end} over {epsilon_decay_episodes} episodes")
    
    def select_action(self, state: np.ndarray, training: bool = True) -> int:
        """
        Select action using noisy network with epsilon-greedy fallback.
        Early episodes use epsilon-greedy for more aggressive exploration,
        which transitions to pure noisy network exploration as training progresses.
        """
        # Reset noise before each action for proper exploration
        if training:
            self.online_net.reset_noise()
        
        # Epsilon-greedy fallback for early exploration
        if training and random.random() < self.current_epsilon:
            return random.randrange(self.num_actions)
        
        with torch.no_grad():
            if training:
                self.online_net.train()
            else:
                self.online_net.eval()
            
            state_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            q_values = self.online_net.get_q_values(state_t)
            self.last_q_value = q_values.max().item()
            return q_values.argmax(dim=1).item()
    
    def update_epsilon(self):
        """Update epsilon for the current episode (call at episode end)."""
        if self.episode_count < self.epsilon_decay_episodes:
            # Linear decay
            decay_progress = self.episode_count / self.epsilon_decay_episodes
            self.current_epsilon = self.epsilon_start - (self.epsilon_start - self.epsilon_end) * decay_progress
        else:
            self.current_epsilon = self.epsilon_end
    
    def push_transition(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool
    ):
        """Add transition to replay buffer."""
        self.buffer.push(state, action, reward, next_state, done)
    
    def learn(self) -> Optional[float]:
        """
        Perform one learning step.
        Returns loss if learning happened, None otherwise.
        """
        if len(self.buffer) < self.min_buffer_size:
            return None
        
        self.step_count += 1
        
        # Sample from buffer
        states, actions, rewards, next_states, dones, weights, indices = self.buffer.sample(self.batch_size)
        
        # Move to device
        states = states.to(self.device)
        actions = actions.to(self.device)
        rewards = rewards.to(self.device)
        next_states = next_states.to(self.device)
        dones = dones.to(self.device)
        weights = weights.to(self.device)
        
        # Reset noise for this update
        self.online_net.reset_noise()
        self.target_net.reset_noise()
        
        # Compute loss and update
        loss, td_errors = self._compute_loss(states, actions, rewards, next_states, dones, weights)
        
        self.optimizer.zero_grad()
        loss.backward()
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(self.online_net.parameters(), 10.0)
        self.optimizer.step()
        
        # Update priorities
        self.buffer.update_priorities(indices, td_errors.cpu().numpy())
        
        # Update target network
        if self.step_count % self.target_update_freq == 0:
            self.target_net.load_state_dict(self.online_net.state_dict())
        
        self.last_loss = loss.item()
        return self.last_loss
    
    def _compute_loss(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        next_states: torch.Tensor,
        dones: torch.Tensor,
        weights: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute distributional loss with Double DQN action selection.
        """
        batch_size = states.size(0)
        
        # Get current distributions
        current_dist = self.online_net(states)
        current_dist = current_dist[range(batch_size), actions]  # (batch, atoms)
        
        # Double DQN: use online net for action selection
        with torch.no_grad():
            next_q_values = self.online_net.get_q_values(next_states)
            next_actions = next_q_values.argmax(dim=1)
            
            # Use target net for value estimation
            next_dist = self.target_net(next_states)
            next_dist = next_dist[range(batch_size), next_actions]  # (batch, atoms)
            
            # Compute projected distribution
            target_dist = self._project_distribution(rewards, dones, next_dist)
        
        # Cross-entropy loss
        log_current_dist = torch.log(current_dist + 1e-8)
        loss = -(target_dist * log_current_dist).sum(dim=1)
        
        # TD errors for priority updates
        td_errors = loss.detach().abs() + 1e-6
        
        # Apply importance sampling weights
        weighted_loss = (weights * loss).mean()
        
        return weighted_loss, td_errors
    
    def _project_distribution(
        self,
        rewards: torch.Tensor,
        dones: torch.Tensor,
        next_dist: torch.Tensor
    ) -> torch.Tensor:
        """
        Project target distribution onto support.
        Implements the categorical projection from C51 paper.
        """
        batch_size = rewards.size(0)
        
        # Compute Tz = r + γ^n * z (for n-step)
        gamma_n = self.gamma ** self.n_step
        rewards = rewards.unsqueeze(1)
        dones = dones.unsqueeze(1)
        
        Tz = rewards + (1 - dones) * gamma_n * self.support.unsqueeze(0)
        Tz = Tz.clamp(self.v_min, self.v_max)
        
        # Compute projection
        b = (Tz - self.v_min) / self.delta_z
        l = b.floor().long()
        u = b.ceil().long()
        
        # Clamp to valid indices
        l = l.clamp(0, self.num_atoms - 1)
        u = u.clamp(0, self.num_atoms - 1)
        
        # Distribute probability mass
        target_dist = torch.zeros_like(next_dist)
        
        offset = torch.arange(batch_size, device=self.device).unsqueeze(1) * self.num_atoms
        
        target_dist.view(-1).index_add_(
            0, (l + offset).view(-1),
            (next_dist * (u.float() - b)).view(-1)
        )
        target_dist.view(-1).index_add_(
            0, (u + offset).view(-1),
            (next_dist * (b - l.float())).view(-1)
        )
        
        return target_dist
    
    def reset_noise(self):
        """Reset noise in networks."""
        self.online_net.reset_noise()
        self.target_net.reset_noise()
    
    def get_hyperparameters(self) -> Dict[str, Any]:
        """Get current hyperparameters."""
        return {
            'gamma': self.gamma,
            'batch_size': self.batch_size,
            'buffer_size': self.buffer.capacity,
            'store_uint8': self.store_uint8,
            'target_update_freq': self.target_update_freq,
            'num_atoms': self.num_atoms,
            'v_min': self.v_min,
            'v_max': self.v_max,
            'n_step': self.n_step,
            'lr': self.optimizer.param_groups[0]['lr'],
            'epsilon_start': self.epsilon_start,
            'epsilon_end': self.epsilon_end,
            'epsilon_decay_episodes': self.epsilon_decay_episodes
        }
    
    def get_stats(self) -> Dict[str, Any]:
        """Get current training statistics."""
        return {
            'step_count': self.step_count,
            'episode_count': self.episode_count,
            'buffer_size': len(self.buffer),
            'last_loss': self.last_loss,
            'last_q_value': self.last_q_value,
            'epsilon': self.current_epsilon,
            'device': str(self.device)
        }
    
    def save(self, path: str, episode: int, total_reward: float, metadata: Optional[Dict] = None):
        """Save model checkpoint."""
        checkpoint = {
            'online_net': self.online_net.state_dict(),
            'target_net': self.target_net.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'step_count': self.step_count,
            'episode_count': episode,
            'total_reward': total_reward,
            'hyperparameters': self.get_hyperparameters(),
            'metadata': metadata or {}
        }
        torch.save(checkpoint, path)
        logger.info(f"Model saved to {path}")
    
    def load(self, path: str) -> Dict[str, Any]:
        """Load model checkpoint."""
        try:
            checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        except TypeError:
            # Older torch versions don't support weights_only.
            checkpoint = torch.load(path, map_location=self.device)
        
        self.online_net.load_state_dict(checkpoint['online_net'])
        self.target_net.load_state_dict(checkpoint['target_net'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.step_count = checkpoint.get('step_count', 0)
        self.episode_count = checkpoint.get('episode_count', 0)
        
        logger.info(f"Model loaded from {path}")
        return checkpoint


# ============== Frame Preprocessing ==============

class FrameStack:
    """
    Frame stacking for temporal information.
    Stacks last N frames as channels.
    """
    
    def __init__(self, num_frames: int = 4, frame_size: Tuple[int, int] = (84, 84)):
        self.num_frames = num_frames
        self.frame_size = frame_size
        self.frames = deque(maxlen=num_frames)
    
    def reset(self, initial_frame: np.ndarray) -> np.ndarray:
        """Reset with initial frame."""
        processed = self._preprocess(initial_frame)
        for _ in range(self.num_frames):
            self.frames.append(processed)
        return self.get_state()
    
    def push(self, frame: np.ndarray) -> np.ndarray:
        """Add new frame and return stacked state."""
        processed = self._preprocess(frame)
        self.frames.append(processed)
        return self.get_state()
    
    def get_state(self) -> np.ndarray:
        """Get current stacked state."""
        return np.array(self.frames)
    
    def _preprocess(self, frame: np.ndarray) -> np.ndarray:
        """Preprocess frame: grayscale, resize, normalize."""
        import cv2
        
        # Convert to grayscale if needed
        if len(frame.shape) == 3:
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        
        # Resize
        frame = cv2.resize(frame, self.frame_size, interpolation=cv2.INTER_AREA)
        
        # Normalize to [0, 1]
        frame = frame.astype(np.float32) / 255.0
        
        return frame
