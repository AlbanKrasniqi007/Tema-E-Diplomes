"""INPUT_CLASSICAL_112_PLANE encoder.

Matches lc0 ``EncodePositionForNN`` for the legacy 112-plane format:

* 8 history positions × 13 planes (6 ours, 6 theirs, 1 repetition)
* 4 castling planes (us/them, queenside/kingside) filled with 1 if the right exists
* side-to-move plane (all ones if Black is to move in the unflipped game)
* rule-50 plane filled with the half-move clock
* unused movecount plane (zeros)
* all-ones plane to help the net find the board edge

The board is always encoded from the side-to-move view (vertical flip when
Black is to move), as in lc0's ``ChessBoard`` representation.
"""

from __future__ import annotations

import chess
import numpy as np

NUM_INPUT_PLANES = 112
PLANES_PER_BOARD = 13
HISTORY_LENGTH = 8
AUX_PLANE_BASE = HISTORY_LENGTH * PLANES_PER_BOARD  # 104

_PIECE_TO_PLANE = {
    chess.PAWN: 0,
    chess.KNIGHT: 1,
    chess.BISHOP: 2,
    chess.ROOK: 3,
    chess.QUEEN: 4,
    chess.KING: 5,
}


def _stm_square(square: int, we_are_black: bool) -> int:
    return chess.square_mirror(square) if we_are_black else square


def _set_square(plane: np.ndarray, square: int, we_are_black: bool, value: float = 1.0) -> None:
    mapped = _stm_square(square, we_are_black)
    rank = chess.square_rank(mapped)
    file = chess.square_file(mapped)
    plane[rank, file] = value


def encode_position(
    board: chess.Board,
    history: list[chess.Board] | None = None,
    history_fill: str = "fen_only",
) -> np.ndarray:
    """Return a ``(112, 8, 8)`` float32 tensor for ``board``.

    ``history`` should be previous positions *including* ``board`` as the last
    element. If omitted, only the current position is used.
    """
    planes = np.zeros((NUM_INPUT_PLANES, 8, 8), dtype=np.float32)
    we_are_black = board.turn == chess.BLACK
    ours_color = board.turn

    if history is None:
        chain = [board]
    else:
        chain = list(history)
        if not chain or chain[-1].fen() != board.fen():
            chain.append(board)

    positions = list(reversed(chain[-HISTORY_LENGTH:]))

    for i, position in enumerate(positions):
        base = i * PLANES_PER_BOARD
        for square, piece in position.piece_map().items():
            offset = _PIECE_TO_PLANE[piece.piece_type]
            channel = offset if piece.color == ours_color else offset + 6
            _set_square(planes[base + channel], square, we_are_black)
        if position.is_repetition(2):
            planes[base + 12].fill(1.0)

    us_oo = board.has_kingside_castling_rights(ours_color)
    us_ooo = board.has_queenside_castling_rights(ours_color)
    them = not ours_color
    them_oo = board.has_kingside_castling_rights(them)
    them_ooo = board.has_queenside_castling_rights(them)
    if us_ooo:
        planes[AUX_PLANE_BASE + 0].fill(1.0)
    if us_oo:
        planes[AUX_PLANE_BASE + 1].fill(1.0)
    if them_ooo:
        planes[AUX_PLANE_BASE + 2].fill(1.0)
    if them_oo:
        planes[AUX_PLANE_BASE + 3].fill(1.0)
    if we_are_black:
        planes[AUX_PLANE_BASE + 4].fill(1.0)
    planes[AUX_PLANE_BASE + 5].fill(float(board.halfmove_clock))
    planes[AUX_PLANE_BASE + 7].fill(1.0)
    return planes
