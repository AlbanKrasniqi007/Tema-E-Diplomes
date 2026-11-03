"""Map chess moves to the official Lc0 1858-index policy vector.

Knight promotions are stored without a suffix (same as a non-promoting
queen-like step). Queen/rook/bishop promotions use ``q``/``r``/``b``.
When Black is to move, squares are vertically mirrored so the index is
always from the side-to-move view, matching lc0's board flip.

The AlphaZero-style convolutional policy head emits 80×8×8 logits which
are gathered into the 1858-vector using the mapping from
``lczero-training/tf/lc0_az_policy_map.py``.
"""

from __future__ import annotations

import chess
import numpy as np

from lc0.chess.policy_index import POLICY_INDEX, POLICY_SIZE

MOVE_TO_INDEX: dict[str, int] = {uci: i for i, uci in enumerate(POLICY_INDEX)}

_COLUMNS = "abcdefgh"
_ROWS = "12345678"
_PROMO_PIECES = "rbq"  # knight promotions share the unsuffixed queen-move slot
_QUEEN_DIRS = {
    "N": (0, 1),
    "NE": (1, 1),
    "E": (1, 0),
    "SE": (1, -1),
    "S": (0, -1),
    "SW": (-1, -1),
    "W": (-1, 0),
    "NW": (-1, 1),
}
_KNIGHT_DIRS = {
    "N": (1, 2),
    "NE": (2, 1),
    "E": (2, -1),
    "SE": (1, -2),
    "S": (-1, -2),
    "SW": (-2, -1),
    "W": (-2, 1),
    "NW": (-1, 2),
}


def _valid(file: int, rank: int) -> bool:
    return 0 <= file <= 7 and 0 <= rank <= 7


def _square_name(file: int, rank: int) -> str:
    return _COLUMNS[file] + _ROWS[rank]


def _apply_dir(start: str, vector: tuple[int, int], steps: int) -> str | None:
    file = _COLUMNS.index(start[0]) + vector[0] * steps
    rank = _ROWS.index(start[1]) + vector[1] * steps
    if not _valid(file, rank):
        return None
    return _square_name(file, rank)


def build_az_flat_moves() -> list[str]:
    """Enumerate the 80×8×8 AlphaZero policy planes (file-inner, rank-outer)."""
    moves: list[str] = []
    for direction in ("N", "NE", "E", "SE", "S", "SW", "W", "NW"):
        for steps in range(1, 8):
            for rank in _ROWS:
                for file in _COLUMNS:
                    start = file + rank
                    end = _apply_dir(start, _QUEEN_DIRS[direction], steps)
                    moves.append("illegal" if end is None else start + end)
    for direction in ("N", "NE", "E", "SE", "S", "SW", "W", "NW"):
        for rank in _ROWS:
            for file in _COLUMNS:
                start = file + rank
                end = _apply_dir(start, _KNIGHT_DIRS[direction], 1)
                moves.append("illegal" if end is None else start + end)
    for direction in ("NW", "N", "NE"):
        for promo in _PROMO_PIECES:
            for rank in _ROWS:
                for file in _COLUMNS:
                    if rank != "7":
                        moves.append("illegal")
                        continue
                    start = file + rank
                    end = _apply_dir(start, _QUEEN_DIRS[direction], 1)
                    moves.append("illegal" if end is None else start + end + promo)
    return moves


def build_gather_indices() -> np.ndarray:
    """For each of the 1858 Lc0 slots, the matching flat 80×8×8 index."""
    gather = np.full((POLICY_SIZE,), -1, dtype=np.int64)
    for az_i, move in enumerate(build_az_flat_moves()):
        if move == "illegal":
            continue
        gather[MOVE_TO_INDEX[move]] = az_i
    if np.any(gather < 0):
        missing = int(np.sum(gather < 0))
        raise RuntimeError(f"AZ policy gather is incomplete ({missing} missing slots)")
    return gather


AZ_GATHER_INDICES = build_gather_indices()


def _uci_key(move: chess.Move) -> str:
    origin = chess.square_name(move.from_square)
    dest = chess.square_name(move.to_square)
    if move.promotion is None or move.promotion == chess.KNIGHT:
        return origin + dest
    suffix = {chess.QUEEN: "q", chess.ROOK: "r", chess.BISHOP: "b"}[move.promotion]
    return origin + dest + suffix


def stm_move(move: chess.Move, we_are_black: bool) -> chess.Move:
    if not we_are_black:
        return move
    return chess.Move(
        chess.square_mirror(move.from_square),
        chess.square_mirror(move.to_square),
        promotion=move.promotion,
    )


def move_to_policy_index(move: chess.Move, we_are_black: bool) -> int:
    key = _uci_key(stm_move(move, we_are_black))
    try:
        return MOVE_TO_INDEX[key]
    except KeyError as exc:
        raise KeyError(f"Move {move.uci()} (stm key {key}) is not in the 1858-index") from exc


def policy_index_to_move(index: int, we_are_black: bool) -> chess.Move:
    uci = POLICY_INDEX[index]
    promo = None
    if len(uci) == 5:
        promo = {"q": chess.QUEEN, "r": chess.ROOK, "b": chess.BISHOP, "n": chess.KNIGHT}[uci[4]]
        uci = uci[:4]
    move = chess.Move.from_uci(uci)
    if promo is not None:
        move.promotion = promo
    if we_are_black:
        move = chess.Move(
            chess.square_mirror(move.from_square),
            chess.square_mirror(move.to_square),
            promotion=move.promotion,
        )
    return move
