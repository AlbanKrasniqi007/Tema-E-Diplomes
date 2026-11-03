"""Batched PyTorch backend with an NNCache, analogous to lc0 neural backends."""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass

import numpy as np
import torch
from torch import nn

from lc0.chess.encoder import NUM_INPUT_PLANES
from lc0.chess.policy_index import POLICY_SIZE
from lc0.config import NetworkConfig
from lc0.neural.network import Lc0SEResNet


@dataclass
class NNOutput:
    policy: np.ndarray  # (1858,) probabilities after legal masking is applied by search
    logits: np.ndarray
    w: float
    d: float
    l: float
    q: float
    m: float


class NetworkBackend:
    def __init__(
        self,
        cfg: NetworkConfig | None = None,
        device: str = "cuda",
        cache_size: int = 200_000,
        weights_path: str | None = None,
    ) -> None:
        self.cfg = cfg or NetworkConfig()
        if device == "cuda" and not torch.cuda.is_available():
            device = "cpu"
        self.device = torch.device(device)
        self.model = Lc0SEResNet(self.cfg).to(self.device)
        self.weights_path: str | None = None
        self.weights_error: str | None = None
        if weights_path:
            try:
                self.load(weights_path)
            except Exception as exc:
                if not self.weights_error:
                    self.weights_error = str(exc)
        self.model.eval()
        self.cache_size = cache_size
        self._cache: OrderedDict[str, NNOutput] = OrderedDict()

    def load(self, path: str) -> None:
        path = path.strip().strip('"')
        lower = path.lower()
        if lower.endswith((".pb.gz", ".pb", ".onnx")):
            self.weights_error = (
                f"rejected official Lc0 net {path} — this replica only loads PyTorch .pt checkpoints"
            )
            raise ValueError(self.weights_error)
        state = torch.load(path, map_location=self.device, weights_only=True)
        if isinstance(state, dict) and "model" in state:
            self.model.load_state_dict(state["model"])
        else:
            self.model.load_state_dict(state)
        self.weights_path = path
        self.weights_error = None
        self._cache.clear()
        self.model.eval()

    def save(self, path: str) -> None:
        torch.save({"model": self.model.state_dict(), "config": self.cfg.__dict__}, path)

    def train_mode(self) -> nn.Module:
        self.model.train()
        return self.model

    def eval_mode(self) -> None:
        self.model.eval()

    def evaluate_batch(
        self,
        planes: np.ndarray,
        policy_temp: float = 1.61,
    ) -> list[NNOutput]:
        """Evaluate ``(N, 112, 8, 8)`` planes. Does not use the cache."""
        tensor = torch.from_numpy(planes).to(self.device)
        if tensor.ndim == 3:
            tensor = tensor.unsqueeze(0)
        with torch.no_grad():
            logits, value, moves_left = self.model(tensor)
            if self.cfg.value_head == "wdl":
                wdl = torch.softmax(value, dim=-1)
            else:
                q = value.squeeze(-1)
                w = (1 + q) / 2
                l = (1 - q) / 2
                d = torch.zeros_like(q)
                wdl = torch.stack([w, d, l], dim=-1)
            temp = max(policy_temp, 1e-3)
            policy = torch.softmax(logits / temp, dim=-1)
        logits_np = logits.cpu().numpy()
        policy_np = policy.cpu().numpy()
        wdl_np = wdl.cpu().numpy()
        mlh_np = moves_left.cpu().numpy()
        outputs: list[NNOutput] = []
        for i in range(logits_np.shape[0]):
            w, d, l = (float(wdl_np[i, 0]), float(wdl_np[i, 1]), float(wdl_np[i, 2]))
            outputs.append(
                NNOutput(
                    policy=policy_np[i],
                    logits=logits_np[i],
                    w=w,
                    d=d,
                    l=l,
                    q=w - l,
                    m=float(mlh_np[i]),
                )
            )
        return outputs

    def evaluate_cached(
        self,
        key: str,
        planes: np.ndarray,
        policy_temp: float = 1.61,
    ) -> NNOutput:
        cached = self._cache.get(key)
        if cached is not None:
            self._cache.move_to_end(key)
            return cached
        result = self.evaluate_batch(planes[None, ...], policy_temp=policy_temp)[0]
        self._cache[key] = result
        if len(self._cache) > self.cache_size:
            self._cache.popitem(last=False)
        return result

    @property
    def dummy_output(self) -> NNOutput:
        return NNOutput(
            policy=np.ones(POLICY_SIZE, dtype=np.float32) / POLICY_SIZE,
            logits=np.zeros(POLICY_SIZE, dtype=np.float32),
            w=0.0,
            d=1.0,
            l=0.0,
            q=0.0,
            m=0.0,
        )


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
