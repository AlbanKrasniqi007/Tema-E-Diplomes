"""Chess WDL (what happened) vs match utility (what each color wants).

The value head predicts the real game outcome from the side-to-move:
win, draw, or loss. That is the AlphaZero / Lc0 training target
(Silver et al. 2018; Lc0 WDL head, 2019).

Match goals are applied only when converting WDL → Q for PUCT, using
Lc0's DrawScore formula Q = (W − L) + ds · D:

  White ds = +1  →  Q = 1 − 2L   (draw as good as a win)
  Black ds = −1  →  Q = 2W − 1   (draw as bad as a loss)

See Lc0 v0.30 ``--draw-score`` / Armageddon: the net still predicts
chess WDL; search changes how much a draw is worth.
"""

from __future__ import annotations

import chess
import numpy as np


def draw_score(stm_is_white: bool) -> float:
    """Lc0 DrawScore from the side-to-move: White seeks draws, Black avoids them."""
    return 1.0 if stm_is_white else -1.0


def utility_q(w: float, d: float, l: float, stm_is_white: bool) -> float:
    """PUCT scalar: chess WDL scored with this color's match goal."""
    return (w - l) + draw_score(stm_is_white) * d


def chess_q(w: float, d: float, l: float) -> float:
    """Standard Lc0 WL = W − L (draw contributes 0)."""
    return w - l


def search_q(w: float, d: float, l: float, stm_is_white: bool, q_mode: str) -> float:
    """PUCT scalar: ``wl`` is W−L; ``drawscore`` is Lc0 Armageddon utility."""
    if q_mode == "drawscore":
        return utility_q(w, d, l, stm_is_white)
    return chess_q(w, d, l)


def stm_wdl(wdl_white: np.ndarray, stm: chess.Color) -> np.ndarray:
    """Game result from the side-to-move. Draws stay draws."""
    wdl = wdl_white if stm == chess.WHITE else wdl_white[::-1]
    return np.asarray(wdl, dtype=np.float32)


def training_wdl(wdl_white: np.ndarray, stm: chess.Color) -> np.ndarray:
    """Value-head target: true outcome from STM, not collapsed utility."""
    return stm_wdl(wdl_white, stm)
