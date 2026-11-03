"""Chess WDL vs PUCT scalar Q.

The value head still predicts true chess WDL (win / draw / loss from STM).
Search can remap that WDL to a scalar Q:

  wl          Q = W − L                 (classic Lc0)
  drawscore   Q = (W − L) + ds · D      (Lc0 Armageddon / DrawScore)
  winonly     Q = W                     (same ranking as 2W−1, range [0, 1])
  asymmetric  White Q = W − L, Black Q = W   (H3 default)

Black Q = W is the same move ranking as DrawScore Black (2W − 1), but the
value sits nearer zero so classic FPU (−Q − 0.330√P) stays well-behaved.
"""

from __future__ import annotations

import chess
import numpy as np


def draw_score(stm_is_white: bool) -> float:
    return 1.0 if stm_is_white else -1.0


def utility_q(w: float, d: float, l: float, stm_is_white: bool) -> float:
    return (w - l) + draw_score(stm_is_white) * d


def chess_q(w: float, d: float, l: float) -> float:
    return w - l


def winonly_q(w: float, d: float, l: float) -> float:
    """Win probability only: draw and loss both score 0."""
    return w


def search_q(w: float, d: float, l: float, stm_is_white: bool, q_mode: str) -> float:
    if q_mode == "drawscore":
        return utility_q(w, d, l, stm_is_white)
    if q_mode == "winonly":
        return winonly_q(w, d, l)
    if q_mode == "asymmetric":
        return chess_q(w, d, l) if stm_is_white else winonly_q(w, d, l)
    return chess_q(w, d, l)


def stm_wdl(wdl_white: np.ndarray, stm: chess.Color) -> np.ndarray:
    wdl = wdl_white if stm == chess.WHITE else wdl_white[::-1]
    return np.asarray(wdl, dtype=np.float32)


def training_wdl(wdl_white: np.ndarray, stm: chess.Color) -> np.ndarray:
    """Value-head target: true outcome from STM. Draws stay draws."""
    return stm_wdl(wdl_white, stm)
