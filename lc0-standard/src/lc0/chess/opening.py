"""Thesis opening: Berlin Defence, trained like Lc0 odds specialist nets.

Every self-play training game starts from this tabiya (White to move after
``1. e4 e5 2. Nf3 Nc6 3. Bb5 Nf6``), the same idea as LeelaKnightOdds /
LeelaQueenOdds / LeelaRookOdds being trained only from their restricted
starting position rather than from the standard initial array.
"""

from __future__ import annotations

import chess

BERLIN_NAME = "Berlin Defence"
BERLIN_ECO = "C65"
BERLIN_PGN = "1. e4 e5 2. Nf3 Nc6 3. Bb5 Nf6"
BERLIN_FEN = "r1bqkb1r/pppp1ppp/2n2n2/1B2p3/4P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 4 4"
BERLIN_UCI = ("e2e4", "e7e5", "g1f3", "b8c6", "f1b5", "g8f6")


def make_berlin_board() -> chess.Board:
    """Board at the Berlin tabiya, with the six opening plies on the stack."""
    board = chess.Board()
    for uci in BERLIN_UCI:
        board.push_uci(uci)
    expected = chess.Board(BERLIN_FEN)
    if board.board_fen() != expected.board_fen() or board.turn != expected.turn:
        raise RuntimeError(f"Berlin setup mismatch: {board.fen()} vs {BERLIN_FEN}")
    return board


def make_opening_board(fen: str = BERLIN_FEN, moves: list[str] | tuple[str, ...] | None = None) -> chess.Board:
    """Start position for specialist training. Prefers playing ``moves`` so history exists."""
    if moves:
        board = chess.Board()
        for uci in moves:
            board.push_uci(uci)
        return board
    return chess.Board(fen)


def history_from_board(board: chess.Board) -> list[chess.Board]:
    """Full position list from game start through ``board`` (for the 112-plane encoder)."""
    rewind = board.copy(stack=True)
    positions = [rewind.copy(stack=False)]
    while rewind.move_stack:
        rewind.pop()
        positions.append(rewind.copy(stack=False))
    positions.reverse()
    return positions
