"""Self-play game generation for the AlphaZero / Lc0 training loop."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import chess
import numpy as np

from lc0.chess.encoder import encode_position
from lc0.chess.opening import history_from_board, make_berlin_board
from lc0.neural.backend import NetworkBackend
from lc0.search.params import SearchParams
from lc0.search.puct import PUCTSearch, SearchResult


@dataclass
class TrainingExample:
    planes: np.ndarray
    policy: np.ndarray
    wdl: np.ndarray
    moves_left: float
    ply: int


class SelfPlayGame:
    def __init__(
        self,
        backend: NetworkBackend,
        params: SearchParams,
        visits: int = 160,
        temperature: float = 1.0,
        temp_decay_moves: int = 15,
        max_ply: int = 512,
        start_factory: Callable[[], chess.Board] | None = None,
    ) -> None:
        self.search = PUCTSearch(backend, params)
        self.visits = visits
        self.temperature = temperature
        self.temp_decay_moves = temp_decay_moves
        self.max_ply = max_ply
        self.start_factory = start_factory or make_berlin_board

    def play(self, start: chess.Board | None = None) -> tuple[list[TrainingExample], chess.Board]:
        board = start.copy(stack=True) if start is not None else self.start_factory()
        history = history_from_board(board)
        snapshots: list[tuple[np.ndarray, np.ndarray, int, chess.Color]] = []
        ply = 0
        while not board.is_game_over(claim_draw=True) and ply < self.max_ply:
            tau = self.temperature if ply < self.temp_decay_moves else 0.0
            result: SearchResult = self.search.search(
                board,
                history=history,
                visits=self.visits,
                add_noise=True,
                temperature=tau,
            )
            if result.move not in board.legal_moves:
                break
            planes = encode_position(board, history)
            snapshots.append((planes, result.policy_target, ply, board.turn))
            board.push(result.move)
            history.append(board.copy(stack=False))
            ply += 1

        wdl_white = _outcome_wdl(board)
        examples: list[TrainingExample] = []
        final_ply = ply
        for planes, policy, example_ply, stm in snapshots:
            wdl = wdl_white if stm == chess.WHITE else wdl_white[::-1]
            examples.append(
                TrainingExample(
                    planes=planes,
                    policy=policy,
                    wdl=wdl,
                    moves_left=float(max(final_ply - example_ply, 0)),
                    ply=example_ply,
                )
            )
        return examples, board


def _outcome_wdl(board: chess.Board) -> np.ndarray:
    """WDL from White's perspective."""
    outcome = board.outcome(claim_draw=True)
    if outcome is None or outcome.winner is None:
        return np.array([0.0, 1.0, 0.0], dtype=np.float32)
    if outcome.winner == chess.WHITE:
        return np.array([1.0, 0.0, 0.0], dtype=np.float32)
    return np.array([0.0, 0.0, 1.0], dtype=np.float32)
