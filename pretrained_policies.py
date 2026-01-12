"""
Utilities for loading and running pre-trained Atari policies.
Supports Bitdefender model zoo checkpoints plus SB3 and PFRL weights.
"""

from __future__ import annotations

import gzip
import io
import zipfile
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class AtariNet(nn.Module):
    """Bitdefender AtariNet architecture used in their model zoo."""

    def __init__(self, num_actions: int, num_atoms: int = 1, v_min: float = -10.0, v_max: float = 10.0):
        super().__init__()
        self.num_actions = num_actions
        self.num_atoms = num_atoms

        self.__features = nn.Sequential(
            nn.Conv2d(4, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
        )

        if num_atoms > 1:
            support = torch.linspace(v_min, v_max, num_atoms)
            # Match Bitdefender state dict key (_AtariNet__support).
            self.register_buffer("_AtariNet__support", support)

        output_dim = num_actions * num_atoms
        self.__head = nn.Sequential(
            nn.Linear(3136, 512),
            nn.ReLU(),
            nn.Linear(512, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.__features(x)
        x = x.view(x.size(0), -1)
        return self.__head(x)

    def get_q_values(self, x: torch.Tensor) -> torch.Tensor:
        logits = self.forward(x)
        if self.num_atoms <= 1:
            return logits
        logits = logits.view(-1, self.num_actions, self.num_atoms)
        probs = F.softmax(logits, dim=2)
        support = getattr(self, "_AtariNet__support")
        return (probs * support).sum(dim=2)


class BitdefenderPolicy:
    """Inference-only policy for Bitdefender AtariNet checkpoints."""

    def __init__(
        self,
        checkpoint_path: str | Path,
        num_actions: int,
        num_atoms: int = 1,
        device: Optional[torch.device] = None,
    ):
        self.checkpoint_path = Path(checkpoint_path)
        self.device = device or torch.device("cpu")
        self.model = AtariNet(num_actions=num_actions, num_atoms=num_atoms)
        self.model.to(self.device)
        self.model.eval()
        self.last_q_value = 0.0
        self.step = None
        self._load_checkpoint()

    def _load_checkpoint(self):
        if not self.checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {self.checkpoint_path}")

        with gzip.open(self.checkpoint_path, "rb") as handle:
            payload = torch.load(handle, map_location=self.device)

        state_dict = payload.get("estimator_state", payload)
        self.step = payload.get("step")
        self.model.load_state_dict(state_dict)

    def select_action(self, state) -> int:
        """Select greedy action for a single stacked state."""
        state_t = torch.as_tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            q_values = self.model.get_q_values(state_t)
        self.last_q_value = float(q_values.max().item())
        return int(q_values.argmax(dim=1).item())


class FactorizedNoisyLinear(nn.Module):
    """Minimal Factorized Noisy Linear compatible with PFRL checkpoints."""

    def __init__(self, in_features: int, out_features: int, sigma_scale: float = 0.4, bias: bool = True):
        super().__init__()
        self.mu = nn.Linear(in_features, out_features, bias=bias)
        self.sigma = nn.Linear(in_features, out_features, bias=bias)
        self.use_noise = False

    def set_deterministic(self, deterministic: bool = True) -> None:
        self.use_noise = not deterministic

    @staticmethod
    def _factorized_noise(size: int, device, dtype):
        noise = torch.normal(mean=0.0, std=1.0, size=(size,), device=device, dtype=dtype)
        return torch.sign(noise) * torch.sqrt(torch.abs(noise))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.use_noise:
            return F.linear(x, self.mu.weight, self.mu.bias)

        dtype = self.sigma.weight.dtype
        device = self.sigma.weight.device
        out_size, in_size = self.sigma.weight.shape

        eps_in = self._factorized_noise(in_size, device, dtype)
        eps_out = self._factorized_noise(out_size, device, dtype)
        weight = self.mu.weight + self.sigma.weight * torch.ger(eps_out, eps_in)
        if self.mu.bias is not None:
            bias = self.mu.bias + self.sigma.bias * eps_out
        else:
            bias = None
        return F.linear(x, weight, bias)


class PFRLAtariBody(nn.Module):
    """CNN body for PFRL DQN checkpoints."""

    def __init__(self):
        super().__init__()
        self.layers = nn.ModuleList(
            [
                nn.Conv2d(4, 32, 8, stride=4),
                nn.Conv2d(32, 64, 4, stride=2),
                nn.Conv2d(64, 64, 3, stride=1),
            ]
        )
        self.output = nn.Linear(3136, 512)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = x
        for layer in self.layers:
            h = F.relu(layer(h))
        h = h.view(h.size(0), -1)
        h = F.relu(self.output(h))
        return h


class PFRLRainbowNet(nn.Module):
    """Distributional dueling Q-network with noisy layers (PFRL Rainbow)."""

    def __init__(self, num_actions: int, n_atoms: int = 51, v_min: float = -10.0, v_max: float = 10.0):
        super().__init__()
        self.num_actions = num_actions
        self.n_atoms = n_atoms
        self.v_min = v_min
        self.v_max = v_max

        self.conv_layers = nn.ModuleList(
            [
                nn.Conv2d(4, 32, 8, stride=4),
                nn.Conv2d(32, 64, 4, stride=2),
                nn.Conv2d(64, 64, 3, stride=1),
            ]
        )
        self.main_stream = FactorizedNoisyLinear(3136, 1024)
        self.a_stream = FactorizedNoisyLinear(512, num_actions * n_atoms)
        self.v_stream = FactorizedNoisyLinear(512, n_atoms)
        self.register_buffer("support", torch.linspace(v_min, v_max, n_atoms))

    def set_deterministic(self, deterministic: bool = True) -> None:
        self.main_stream.set_deterministic(deterministic)
        self.a_stream.set_deterministic(deterministic)
        self.v_stream.set_deterministic(deterministic)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = x
        for layer in self.conv_layers:
            h = F.relu(layer(h))

        batch_size = x.size(0)
        h = h.view(batch_size, -1)
        h = F.relu(self.main_stream(h))
        h_a, h_v = torch.chunk(h, 2, dim=1)

        adv = self.a_stream(h_a).view(batch_size, self.num_actions, self.n_atoms)
        adv_mean = adv.mean(dim=1, keepdim=True)
        adv = adv - adv_mean

        val = self.v_stream(h_v).view(batch_size, 1, self.n_atoms)
        q_atoms = adv + val
        q_dist = F.softmax(q_atoms, dim=2)
        return q_dist

    def get_q_values(self, x: torch.Tensor) -> torch.Tensor:
        q_dist = self.forward(x)
        return (q_dist * self.support).sum(dim=2)


class SoftmaxCategoricalHead(nn.Module):
    def forward(self, logits):
        return torch.distributions.Categorical(logits=logits)


class Branched(nn.Module):
    """Minimal Branched module compatible with PFRL state dicts."""

    def __init__(self, *modules: nn.Module):
        super().__init__()
        self.child_modules = nn.ModuleList(modules)

    def forward(self, x):
        return tuple(module(x) for module in self.child_modules)


class PFRLA3CNet(nn.Module):
    """A3C model layout used in PFRL reproduction scripts."""

    def __init__(self, num_actions: int):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(4, 16, 8, stride=4),
            nn.ReLU(),
            nn.Conv2d(16, 32, 4, stride=2),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(2592, 256),
            nn.ReLU(),
            Branched(
                nn.Sequential(
                    nn.Linear(256, num_actions),
                    SoftmaxCategoricalHead(),
                ),
                nn.Linear(256, 1),
            ),
        )

    def forward(self, x):
        return self.model(x)


class SB3Policy:
    """Inference-only policy for SB3 RL Zoo checkpoints."""

    def __init__(self, checkpoint_path: str | Path, algorithm: str, device: Optional[torch.device] = None):
        self.checkpoint_path = str(checkpoint_path)
        self.algorithm = (algorithm or "").lower()
        self.device = device or torch.device("cpu")
        self.last_q_value = None

        try:
            from stable_baselines3 import DQN, A2C, PPO
        except Exception as exc:
            raise ImportError("stable-baselines3 is required for SB3 pretrained models") from exc

        algo_map = {
            "dqn": DQN,
            "a2c": A2C,
            "ppo": PPO,
        }

        if self.algorithm == "qrdqn":
            try:
                from sb3_contrib import QRDQN
            except Exception as exc:
                raise ImportError("sb3-contrib is required for QRDQN pretrained models") from exc
            algo_cls = QRDQN
        else:
            algo_cls = algo_map.get(self.algorithm)

        if not algo_cls:
            raise ValueError(f"Unsupported SB3 algorithm: {self.algorithm}")

        self.model = algo_cls.load(self.checkpoint_path, device=self.device)
        self.obs_shape = getattr(self.model.observation_space, "shape", None)

    def _prepare_obs(self, state: np.ndarray) -> np.ndarray:
        obs = state
        if obs.dtype != np.uint8:
            obs = (obs * 255.0).clip(0, 255).astype(np.uint8)
        if self.obs_shape and len(self.obs_shape) == 3:
            if self.obs_shape[0] == 4 and obs.shape == (4, 84, 84):
                return obs
            if self.obs_shape[-1] == 4 and obs.shape == (4, 84, 84):
                return np.transpose(obs, (1, 2, 0))
        return obs

    def select_action(self, state) -> int:
        obs = self._prepare_obs(state)
        action, _ = self.model.predict(obs, deterministic=True)
        try:
            obs_t, _ = self.model.policy.obs_to_tensor(obs)
            if hasattr(self.model, "q_net"):
                with torch.no_grad():
                    q_values = self.model.q_net(obs_t)
                self.last_q_value = float(q_values.max().item())
            else:
                self.last_q_value = None
        except Exception:
            self.last_q_value = None
        return int(action)


class PFRLPolicy:
    """Inference-only policy for PFRL pretrained models."""

    def __init__(
        self,
        checkpoint_path: str | Path,
        algorithm: str,
        num_actions: int,
        device: Optional[torch.device] = None,
    ):
        self.checkpoint_path = Path(checkpoint_path)
        self.algorithm = (algorithm or "").upper()
        self.device = device or torch.device("cpu")
        self.last_q_value = None

        state_dict = self._load_state_dict(self.checkpoint_path)

        if self.algorithm == "DQN":
            body = PFRLAtariBody()
            model = nn.Sequential(body, nn.Linear(512, num_actions))
        elif self.algorithm == "RAINBOW":
            model = PFRLRainbowNet(num_actions)
            model.set_deterministic(True)
        elif self.algorithm == "A3C":
            model = PFRLA3CNet(num_actions)
        else:
            raise ValueError(f"Unsupported PFRL algorithm: {self.algorithm}")

        model.load_state_dict(state_dict)
        model.to(self.device)
        model.eval()
        self.model = model

    @staticmethod
    def _load_state_dict(zip_path: Path) -> dict:
        if not zip_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {zip_path}")
        with zipfile.ZipFile(zip_path, "r") as zf:
            model_entries = [name for name in zf.namelist() if name.endswith("model.pt")]
            if not model_entries:
                raise FileNotFoundError("model.pt not found in PFRL zip")
            data = zf.read(model_entries[0])
        return torch.load(io.BytesIO(data), map_location="cpu")

    def select_action(self, state) -> int:
        state_t = torch.as_tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)

        if self.algorithm == "A3C":
            dist, _ = self.model(state_t)
            if hasattr(dist, "probs"):
                probs = dist.probs
                action = probs.argmax(dim=1).item()
            else:
                logits = dist.logits
                action = logits.argmax(dim=1).item()
            return int(action)

        if self.algorithm == "RAINBOW":
            with torch.no_grad():
                q_values = self.model.get_q_values(state_t)
        else:
            with torch.no_grad():
                q_values = self.model(state_t)
        self.last_q_value = float(q_values.max().item())
        return int(q_values.argmax(dim=1).item())
