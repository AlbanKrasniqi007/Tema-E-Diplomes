"""YAML-backed engine configuration."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import yaml

from lc0.chess.opening import BERLIN_UCI, make_opening_board


@dataclass
class NetworkConfig:
    """NETWORK_SE_WITH_HEADFORMAT replica, scaled for local training."""

    blocks: int = 6
    filters: int = 64
    se_channels: int = 16
    policy_head: str = "convolution"  # convolution | classical
    value_head: str = "wdl"  # wdl | classical
    mlh_channels: int = 8
    mlh_fc: int = 128
    policy_conv_size: int = 32


@dataclass
class SearchConfig:
    cpuct: float = 1.745
    cpuct_base: float = 38739.0
    cpuct_factor: float = 3.894
    fpu_strategy: str = "reduction"
    fpu_value: float = 0.330
    policy_softmax_temp: float = 1.61
    minibatch_size: int = 32
    max_collision_events: int = 32
    dirichlet_alpha: float = 0.3
    dirichlet_epsilon: float = 0.25
    two_fold_draws: bool = True


@dataclass
class TrainConfig:
    visits: int = 160
    temperature: float = 1.0
    temp_decay_moves: int = 15
    max_game_ply: int = 512
    batch_size: int = 256
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    mlh_weight: float = 0.05
    policy_weight: float = 1.0
    value_weight: float = 1.0
    games_per_iteration: int = 64
    train_steps_per_iteration: int = 200
    replay_size: int = 50_000
    checkpoint_every: int = 1
    opening_name: str = "Berlin Defence"
    opening_fen: str = "r1bqkb1r/pppp1ppp/2n2n2/1B2p3/4P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 4 4"
    opening_pgn: str = "1. e4 e5 2. Nf3 Nc6 3. Bb5 Nf6"
    opening_moves: list[str] | None = None


@dataclass
class EngineConfig:
    device: str = "cuda"
    network: NetworkConfig = field(default_factory=NetworkConfig)
    search: SearchConfig = field(default_factory=SearchConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    weights_path: str = ""
    seed: int = 0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _merge(dc_cls, data: dict[str, Any] | None):
    if not data:
        return dc_cls()
    field_names = {f.name for f in dc_cls.__dataclass_fields__.values()}
    merged = {k: v for k, v in data.items() if k in field_names}
    obj = dc_cls(**merged)
    return obj


def load_config(path: str | Path | None = None) -> EngineConfig:
    if path is None:
        cfg = EngineConfig()
        cfg.train.opening_moves = list(BERLIN_UCI)
        return cfg
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    cfg = EngineConfig(
        device=raw.get("device", "cuda"),
        weights_path=raw.get("weights_path", ""),
        seed=int(raw.get("seed", 0)),
        network=_merge(NetworkConfig, raw.get("network")),
        search=_merge(SearchConfig, raw.get("search")),
        train=_merge(TrainConfig, raw.get("train")),
    )
    if not cfg.train.opening_moves:
        cfg.train.opening_moves = list(BERLIN_UCI)
    return cfg


def opening_board(cfg: EngineConfig | TrainConfig | None = None):
    train = cfg.train if isinstance(cfg, EngineConfig) else cfg
    if train is None:
        return make_opening_board(moves=BERLIN_UCI)
    return make_opening_board(fen=train.opening_fen, moves=train.opening_moves or BERLIN_UCI)
