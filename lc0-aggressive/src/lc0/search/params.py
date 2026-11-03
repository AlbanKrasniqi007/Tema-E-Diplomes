"""Classic search constants from lc0 ``search/classic/params.cc``."""

from __future__ import annotations

from dataclasses import dataclass

from lc0.config import SearchConfig


@dataclass
class SearchParams:
    cpuct: float = 1.745
    cpuct_base: float = 38739.0
    cpuct_factor: float = 3.894
    fpu_absolute: bool = False
    fpu_value: float = 0.330
    policy_softmax_temp: float = 1.61
    minibatch_size: int = 32
    max_collision_events: int = 32
    dirichlet_alpha: float = 0.3
    dirichlet_epsilon: float = 0.0
    two_fold_draws: bool = True
    q_mode: str = "asymmetric"  # wl | drawscore | winonly | asymmetric
    fpu_mode: str = "negation"  # negation | parent

    @classmethod
    def from_config(cls, cfg: SearchConfig, training: bool = False) -> "SearchParams":
        q_mode = getattr(cfg, "q_mode", None) or "asymmetric"
        fpu_mode = getattr(cfg, "fpu_mode", None) or "negation"
        return cls(
            cpuct=cfg.cpuct,
            cpuct_base=cfg.cpuct_base,
            cpuct_factor=cfg.cpuct_factor,
            fpu_absolute=cfg.fpu_strategy == "absolute",
            fpu_value=cfg.fpu_value,
            policy_softmax_temp=cfg.policy_softmax_temp,
            minibatch_size=cfg.minibatch_size,
            max_collision_events=cfg.max_collision_events,
            dirichlet_alpha=cfg.dirichlet_alpha,
            dirichlet_epsilon=cfg.dirichlet_epsilon if training else 0.0,
            two_fold_draws=cfg.two_fold_draws,
            q_mode=q_mode,
            fpu_mode=fpu_mode,
        )

    def compute_cpuct(self, parent_n: int) -> float:
        if self.cpuct_factor == 0.0:
            return self.cpuct
        return self.cpuct + self.cpuct_factor * float(
            __import__("math").log((parent_n + self.cpuct_base) / self.cpuct_base)
        )
