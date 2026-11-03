"""Two color-specialist networks behind one evaluation / UCI backend."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch

from lc0.chess.policy_index import POLICY_SIZE
from lc0.config import NetworkConfig
from lc0.neural.backend import NNOutput, NetworkBackend


def _model_state_from_blob(blob: Any) -> Any:
    """Accept a raw state_dict or a ``{model: ...}`` (or trainer) blob."""
    if isinstance(blob, dict) and "model" in blob:
        return blob["model"]
    return blob


def _stm_white_from_key_or_planes(key: str, planes: np.ndarray) -> bool:
    parts = key.split()
    if len(parts) >= 2 and parts[1] in ("w", "b"):
        return parts[1] == "w"
    # Encoder plane 108: 0 = White to move, 1 = Black to move.
    return float(np.asarray(planes)[108].reshape(-1)[0]) < 0.5


class DualNetworkBackend:
    """Routes leaf evals by side-to-move onto White and Black specialists.

    UCI and self-play keep a single ``PUCTSearch``; this wrapper makes White
    nodes use the White net and Black nodes use the Black net, so the two
    agents play each other inside one process.
    """

    def __init__(
        self,
        cfg: NetworkConfig | None = None,
        device: str = "cuda",
        cache_size: int = 200_000,
        weights_path: str | None = None,
        white: Any = None,
        black: Any = None,
    ) -> None:
        self.cfg = cfg or NetworkConfig()
        created_pair = white is None and black is None
        self.white = white or NetworkBackend(self.cfg, device=device, cache_size=cache_size)
        self.black = black or NetworkBackend(self.cfg, device=device, cache_size=cache_size)
        self.weights_path: str | None = None
        self.weights_error: str | None = None
        self.cloned_from_single = created_pair
        self.white_optimizer_state: Any = None
        self.black_optimizer_state: Any = None
        self.white_step = 0
        self.black_step = 0
        self.meta_games_played = 0
        self.meta_iteration = 0
        if created_pair and hasattr(self.white, "model") and hasattr(self.black, "model"):
            # Same random init, then they diverge in co-play training.
            self.black.model.load_state_dict(self.white.model.state_dict())
        self.device = getattr(self.white, "device", torch.device("cpu"))
        if weights_path:
            try:
                self.load(weights_path)
            except Exception as exc:
                if not self.weights_error:
                    self.weights_error = str(exc)
        self.eval_mode()

    def eval_mode(self) -> None:
        if hasattr(self.white, "eval_mode"):
            self.white.eval_mode()
        if hasattr(self.black, "eval_mode"):
            self.black.eval_mode()

    def load(self, path: str) -> None:
        path = path.strip().strip('"')
        lower = path.lower()
        if lower.endswith((".pb.gz", ".pb", ".onnx")):
            self.weights_error = (
                f"rejected official Lc0 net {path} — this replica only loads PyTorch .pt checkpoints"
            )
            raise ValueError(self.weights_error)
        try:
            state = torch.load(path, map_location=self.device, weights_only=True)
        except Exception:
            state = torch.load(path, map_location=self.device, weights_only=False)
        self.white_optimizer_state = None
        self.black_optimizer_state = None
        self.white_step = 0
        self.black_step = 0
        self.meta_games_played = 0
        self.meta_iteration = 0
        if isinstance(state, dict):
            self.meta_games_played = int(state.get("games_played", 0) or 0)
            self.meta_iteration = int(state.get("iteration", 0) or 0)
        if isinstance(state, dict) and "white" in state and "black" in state:
            self._apply_blob(self.white, state["white"])
            self._apply_blob(self.black, state["black"])
            self.cloned_from_single = False
            self._stash_optimizer(state["white"], state["black"])
        else:
            # Old single-net checkpoint (Trainer.save or NetworkBackend.save).
            blob = _model_state_from_blob(state) if isinstance(state, dict) else state
            self._apply_blob(self.white, blob)
            self._apply_blob(self.black, blob)
            self.cloned_from_single = True
        self.weights_path = path
        if hasattr(self.white, "weights_path"):
            self.white.weights_path = path
        if hasattr(self.black, "weights_path"):
            self.black.weights_path = path
        self.weights_error = None
        self.eval_mode()

    def save(self, path: str, extra: dict[str, Any] | None = None) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        white_blob: dict[str, Any] = {"model": self.white.model.state_dict()}
        black_blob: dict[str, Any] = {"model": self.black.model.state_dict()}
        payload: dict[str, Any] = {
            "kind": "dual",
            "white": white_blob,
            "black": black_blob,
            "config": dict(self.cfg.__dict__) if hasattr(self.cfg, "__dict__") else self.cfg,
        }
        if extra:
            if isinstance(extra.get("white"), dict):
                white_blob.update(extra["white"])
            if isinstance(extra.get("black"), dict):
                black_blob.update(extra["black"])
            for key, value in extra.items():
                if key not in {"white", "black"}:
                    payload[key] = value
        torch.save(payload, path)

    def evaluate_batch(
        self,
        planes: np.ndarray,
        policy_temp: float = 1.61,
    ) -> list[NNOutput]:
        """Split a ``(N, 112, 8, 8)`` batch by plane 108; preserve input order."""
        tensor = np.asarray(planes)
        if tensor.ndim == 3:
            tensor = tensor[None, ...]
        n = int(tensor.shape[0])
        if n == 0:
            return []
        # Plane 108 is filled with 0 (White) or 1 (Black).
        stm_black = tensor[:, 108, 0, 0] >= 0.5
        outputs: list[NNOutput | None] = [None] * n
        white_idx = np.flatnonzero(~stm_black)
        black_idx = np.flatnonzero(stm_black)
        if white_idx.size:
            white_out = self.white.evaluate_batch(tensor[white_idx], policy_temp=policy_temp)
            for i, out in zip(white_idx, white_out, strict=True):
                outputs[int(i)] = out
        if black_idx.size:
            black_out = self.black.evaluate_batch(tensor[black_idx], policy_temp=policy_temp)
            for i, out in zip(black_idx, black_out, strict=True):
                outputs[int(i)] = out
        return outputs  # type: ignore[return-value]

    def evaluate_cached(
        self,
        key: str,
        planes: np.ndarray,
        policy_temp: float = 1.61,
    ) -> NNOutput:
        backend = self.white if _stm_white_from_key_or_planes(key, planes) else self.black
        return backend.evaluate_cached(key, planes, policy_temp=policy_temp)

    @property
    def dummy_output(self) -> NNOutput:
        if hasattr(self.white, "dummy_output"):
            return self.white.dummy_output
        return NNOutput(
            policy=np.ones(POLICY_SIZE, dtype=np.float32) / POLICY_SIZE,
            logits=np.zeros(POLICY_SIZE, dtype=np.float32),
            w=0.0,
            d=1.0,
            l=0.0,
            q=0.0,
            m=0.0,
        )

    def restore_optimizers(self, white_trainer: Any, black_trainer: Any) -> None:
        """Reload AdamW state from a dual checkpoint, if present."""
        if self.white_optimizer_state is not None:
            white_trainer.optimizer.load_state_dict(self.white_optimizer_state)
            white_trainer.step_count = int(self.white_step)
        if self.black_optimizer_state is not None:
            black_trainer.optimizer.load_state_dict(self.black_optimizer_state)
            black_trainer.step_count = int(self.black_step)

    def _apply_blob(self, backend: NetworkBackend, blob: Any) -> None:
        state_dict = _model_state_from_blob(blob)
        backend.model.load_state_dict(state_dict)
        backend._cache.clear()
        backend.model.eval()
        backend.weights_error = None

    def _stash_optimizer(self, white_blob: Any, black_blob: Any) -> None:
        if isinstance(white_blob, dict) and "optimizer" in white_blob:
            self.white_optimizer_state = white_blob["optimizer"]
            self.white_step = int(white_blob.get("step", 0) or 0)
        if isinstance(black_blob, dict) and "optimizer" in black_blob:
            self.black_optimizer_state = black_blob["optimizer"]
            self.black_step = int(black_blob.get("step", 0) or 0)
