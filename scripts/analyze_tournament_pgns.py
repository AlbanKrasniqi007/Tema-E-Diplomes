#!/usr/bin/env python3
"""Summarise the Berlin evaluation matches from PGN headers and moves."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import chess
import chess.pgn

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "tournament" / "pgn_analysis.json"
FEN_C67 = "r1bk3r/ppp2ppp/2p5/4Pn2/8/8/PPP2PPP/RNB2RK1 w - - 0 9"

MATCHES = [
    ("stdW_varB", ROOT / "tournament" / "berlin_std_white_vs_var_black_g3440" / "games"),
    ("stdW_stdB", ROOT / "tournament" / "berlin_std_white_vs_std_black_g3440" / "games"),
    ("varW_stdB", ROOT / "tournament" / "berlin_var_white_vs_std_black_g3440" / "games"),
    ("aggW_stdB", ROOT / "tournament" / "berlin_aggressive_white_vs_std_black_g3440" / "games"),
]


def analyse_dir(games_dir: Path) -> dict:
    files = sorted(games_dir.glob("game_*.pgn"))
    terms: Counter[str] = Counter()
    fourth: Counter[str] = Counter()
    plies: list[int] = []
    qxd8 = 0
    c67 = 0
    queens_off_by_20 = 0
    results: Counter[str] = Counter()
    for path in files:
        with path.open(encoding="utf-8") as fh:
            game = chess.pgn.read_game(fh)
        if game is None:
            continue
        terms[game.headers.get("Termination", "UNKNOWN")] += 1
        results[game.headers.get("Result", "*")] += 1
        ply = int(game.headers.get("PlyCount") or 0)
        plies.append(ply)
        board = game.board()
        sans: list[str] = []
        saw_qxd8 = False
        saw_c67 = False
        queens_off = False
        for i, move in enumerate(game.mainline_moves()):
            san = board.san(move)
            sans.append(san)
            if i == 6:  # first move after Berlin (White's 4th)
                fourth[san] += 1
            if "Qxd8" in san:
                saw_qxd8 = True
            board.push(move)
            if board.board_fen() == FEN_C67.split()[0] and board.turn == chess.WHITE:
                saw_c67 = True
            if i < 40 and board.pieces(chess.QUEEN, chess.WHITE) == chess.SquareSet() and board.pieces(chess.QUEEN, chess.BLACK) == chess.SquareSet():
                queens_off = True
        if saw_qxd8:
            qxd8 += 1
        if saw_c67:
            c67 += 1
        if queens_off:
            queens_off_by_20 += 1
    n = len(files)
    plies_sorted = sorted(plies)
    return {
        "n": n,
        "termination": dict(terms),
        "result": dict(results),
        "white_move_4": dict(fourth.most_common(12)),
        "qxd8_games": qxd8,
        "c67_exact_fen": c67,
        "both_queens_off": queens_off_by_20,
        "mean_ply": round(sum(plies) / n, 1) if n else 0,
        "median_ply": plies_sorted[n // 2] if n else 0,
        "min_ply": min(plies) if plies else 0,
        "max_ply": max(plies) if plies else 0,
    }


def main() -> None:
    payload = {key: analyse_dir(path) for key, path in MATCHES}
    OUT.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
