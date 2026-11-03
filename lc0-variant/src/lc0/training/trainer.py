"""Supervised step of the Lc0 RL loop: train on self-play (policy, WDL, MLH)."""

from __future__ import annotations

from collections import deque
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from lc0.config import EngineConfig
from lc0.neural.backend import NetworkBackend
from lc0.selfplay.game import TrainingExample


class ReplayBuffer:
    def __init__(self, capacity: int) -> None:
        self._items: deque[TrainingExample] = deque(maxlen=capacity)

    def extend(self, examples: list[TrainingExample]) -> None:
        self._items.extend(examples)

    def __len__(self) -> int:
        return len(self._items)

    def sample(self, batch_size: int) -> list[TrainingExample]:
        size = min(batch_size, len(self._items))
        idx = np.random.choice(len(self._items), size=size, replace=False)
        return [self._items[int(i)] for i in idx]


class Trainer:
    def __init__(self, backend: NetworkBackend, cfg: EngineConfig) -> None:
        self.backend = backend
        self.cfg = cfg
        self.buffer = ReplayBuffer(cfg.train.replay_size)
        self.optimizer = torch.optim.AdamW(
            backend.model.parameters(),
            lr=cfg.train.learning_rate,
            weight_decay=cfg.train.weight_decay,
        )
        self.step_count = 0

    def add_games(self, examples: list[TrainingExample]) -> None:
        self.buffer.extend(examples)

    def train_steps(self, steps: int | None = None) -> dict[str, float]:
        steps = steps or self.cfg.train.train_steps_per_iteration
        if len(self.buffer) < 8:
            return {"loss": 0.0, "policy": 0.0, "value": 0.0, "mlh": 0.0, "steps": 0}
        self.backend.train_mode()
        totals = {"loss": 0.0, "policy": 0.0, "value": 0.0, "mlh": 0.0}
        for _ in range(steps):
            batch = self.buffer.sample(self.cfg.train.batch_size)
            metrics = self._step(batch)
            for key, value in metrics.items():
                totals[key] += value
            self.step_count += 1
        self.backend.eval_mode()
        n = max(steps, 1)
        return {key: value / n for key, value in totals.items()} | {"steps": float(steps)}

    def _step(self, batch: list[TrainingExample]) -> dict[str, float]:
        planes = torch.from_numpy(np.stack([ex.planes for ex in batch])).to(self.backend.device)
        policy_t = torch.from_numpy(np.stack([ex.policy for ex in batch])).to(self.backend.device)
        wdl_t = torch.from_numpy(np.stack([ex.wdl for ex in batch])).to(self.backend.device)
        mlh_t = torch.tensor([ex.moves_left for ex in batch], dtype=torch.float32, device=self.backend.device)

        policy_logits, value, moves_left = self.backend.model(planes)
        log_policy = F.log_softmax(policy_logits, dim=-1)
        policy_loss = -(policy_t * log_policy).sum(dim=-1).mean()
        if self.backend.cfg.value_head == "wdl":
            value_loss = F.cross_entropy(value, wdl_t)
        else:
            z = wdl_t[:, 0] - wdl_t[:, 2]
            value_loss = F.mse_loss(value.squeeze(-1), z)
        mlh_loss = F.mse_loss(moves_left, mlh_t)
        loss = (
            self.cfg.train.policy_weight * policy_loss
            + self.cfg.train.value_weight * value_loss
            + self.cfg.train.mlh_weight * mlh_loss
        )
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(self.backend.model.parameters(), 10.0)
        self.optimizer.step()
        return {
            "loss": float(loss.item()),
            "policy": float(policy_loss.item()),
            "value": float(value_loss.item()),
            "mlh": float(mlh_loss.item()),
        }

    def save(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "model": self.backend.model.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "step": self.step_count,
                "config": self.cfg.to_dict(),
            },
            path,
        )
