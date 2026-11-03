from __future__ import annotations

import sys
from pathlib import Path

import chess

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from lc0.config import load_config
from lc0.neural.backend import NetworkBackend, count_parameters
from lc0.search.params import SearchParams
from lc0.search.puct import PUCTSearch


def main() -> None:
    cfg = load_config(ROOT / "configs" / "default.yaml")
    backend = NetworkBackend(cfg.network, device=cfg.device)
    print("device", backend.device)
    print("params", f"{count_parameters(backend.model):,}")
    search = PUCTSearch(backend, SearchParams.from_config(cfg.search))
    board = chess.Board()
    result = search.search(board, visits=16, temperature=0.0)
    print("bestmove", result.move.uci(), "nodes", result.nodes, "q", round(result.q, 4), "time", round(result.elapsed, 3))
    assert result.move in board.legal_moves


if __name__ == "__main__":
    main()
