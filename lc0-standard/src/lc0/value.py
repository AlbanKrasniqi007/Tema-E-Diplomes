"""Chess WDL vs PUCT scalar Q.

Classic search uses Q = W − L. The thesis ablation can switch the same
network to Lc0 DrawScore Q = (W − L) + ds · D without retraining.
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


def search_q(w: float, d: float, l: float, stm_is_white: bool, q_mode: str) -> float:
    if q_mode == "drawscore":
        return utility_q(w, d, l, stm_is_white)
    return chess_q(w, d, l)


def stm_wdl(wdl_white: np.ndarray, stm: chess.Color) -> np.ndarray:
    wdl = wdl_white if stm == chess.WHITE else wdl_white[::-1]
    return np.asarray(wdl, dtype=np.float32)
