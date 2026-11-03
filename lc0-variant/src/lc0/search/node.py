"""MCTS tree nodes and edges for classic PUCT search."""

from __future__ import annotations

from dataclasses import dataclass, field

import chess

from lc0.value import chess_q, search_q


@dataclass
class Edge:
    move: chess.Move
    prior: float
    n: int = 0
    n_in_flight: int = 0
    w: float = 0.0
    d: float = 0.0
    l: float = 0.0
    m_sum: float = 0.0
    child: Node | None = None
    stm_white: bool = True
    q_mode: str = "drawscore"

    @property
    def n_total(self) -> int:
        return self.n + self.n_in_flight

    @property
    def q(self) -> float:
        """Match utility for PUCT (White draw-seeking, Black win-seeking)."""
        if self.n == 0:
            return 0.0
        n = float(self.n)
        return search_q(self.w / n, self.d / n, self.l / n, self.stm_white, self.q_mode)

    @property
    def chess_q(self) -> float:
        """Chess W−L for UCI display."""
        if self.n == 0:
            return 0.0
        n = float(self.n)
        return chess_q(self.w / n, self.d / n, self.l / n)

    @property
    def m(self) -> float:
        if self.n == 0:
            return 0.0
        return self.m_sum / self.n

    def u(self, u_coeff: float) -> float:
        return u_coeff * self.prior / (1 + self.n_total)


@dataclass
class Node:
    n: int = 0
    w: float = 0.0
    d: float = 0.0
    l: float = 0.0
    m_sum: float = 0.0
    terminal: bool = False
    stm_white: bool = True
    q_mode: str = "drawscore"
    edges: list[Edge] = field(default_factory=list)

    @property
    def q(self) -> float:
        if self.n == 0:
            return 0.0
        n = float(self.n)
        return search_q(self.w / n, self.d / n, self.l / n, self.stm_white, self.q_mode)

    @property
    def chess_q(self) -> float:
        if self.n == 0:
            return 0.0
        n = float(self.n)
        return chess_q(self.w / n, self.d / n, self.l / n)

    @property
    def visited_policy(self) -> float:
        return float(sum(edge.prior for edge in self.edges if edge.n_total > 0))

    def is_expanded(self) -> bool:
        return bool(self.edges) or self.terminal

    def best_edge(self) -> Edge:
        return max(self.edges, key=lambda e: (e.n, e.q, e.prior))
