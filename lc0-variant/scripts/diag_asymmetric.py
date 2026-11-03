"""Print chess WDL vs match-utility Q and FPU / visit concentration at the Berlin."""

from __future__ import annotations

import sys
from pathlib import Path

import chess
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from lc0.chess.encoder import encode_position
from lc0.chess.opening import history_from_board, make_berlin_board
from lc0.config import load_config
from lc0.neural.backend import NetworkBackend
from lc0.search.node import Node
from lc0.search.params import SearchParams
from lc0.search.puct import PUCTSearch
from lc0.value import training_wdl, utility_q


def _wdl3(w: float, d: float, l: float) -> str:
    return f"W={w:.3f} D={d:.3f} L={l:.3f}"


def print_training_table() -> None:
    outcomes = {
        "1-0": np.array([1.0, 0.0, 0.0], dtype=np.float32),
        "draw": np.array([0.0, 1.0, 0.0], dtype=np.float32),
        "0-1": np.array([0.0, 0.0, 1.0], dtype=np.float32),
    }
    print("training_wdl  (true game result from STM — draws stay draws)")
    print(f"{'result':<8}{'White STM':<28}{'Black STM'}")
    for name, wdl in outcomes.items():
        w = training_wdl(wdl, chess.WHITE)
        b = training_wdl(wdl, chess.BLACK)
        print(
            f"{name:<8}"
            f"{_wdl3(float(w[0]), float(w[1]), float(w[2])):<28}"
            f"{_wdl3(float(b[0]), float(b[1]), float(b[2]))}"
        )


def diagnose(search: PUCTSearch, backend: NetworkBackend, board: chess.Board, label: str) -> None:
    history = history_from_board(board)
    raw = backend.evaluate_cached(
        board.fen(), encode_position(board, history), search.params.policy_softmax_temp
    )
    stm_white = board.turn == chess.WHITE
    uq = utility_q(raw.w, raw.d, raw.l, stm_white)
    node = Node()
    search._evaluate_and_expand(node, board, history, add_noise=False)
    node.n = max(node.n, 1)
    fpu = search._fpu(node)
    color = "White" if stm_white else "Black"
    print(f"\n=== {label}  ({color} to move) ===")
    print(f"raw NN     {_wdl3(raw.w, raw.d, raw.l)}  chess_Q={raw.q:+.4f}")
    print(f"utility Q  {uq:+.4f}")
    print(f"node.q     {node.q:+.4f}  chess_q={node.chess_q:+.4f}  FPU={fpu:+.4f}")

    result = search.search(board, history=history, visits=32, add_noise=False, temperature=0.0)
    edges = sorted(result.root.edges, key=lambda e: (e.n, e.q, e.prior), reverse=True)
    print(f"{'move':<8}{'N':>5}{'P':>9}{'utilQ':>10}{'chessQ':>10}{'D':>8}")
    for edge in edges[:8]:
        d = (edge.d / edge.n) if edge.n else 0.0
        print(
            f"{edge.move.uci():<8}{edge.n:5d}{edge.prior:9.4f}"
            f"{edge.q:+10.4f}{edge.chess_q:+10.4f}{d:8.3f}"
        )
    visits = [e.n for e in edges]
    ones = sum(1 for n in visits if n == 1)
    print(f"concentration  maxN={max(visits) if visits else 0}  N==1={ones}/{len(visits)}")


def main() -> None:
    print_training_table()
    ckpt = ROOT / "checkpoints" / "lc0-variant-iter0050.pt"
    if not ckpt.is_file():
        print(f"\nskip NN diag: missing {ckpt}")
        return
    device = "cuda" if torch.cuda.is_available() else "cpu"
    cfg = load_config(ROOT / "configs" / "default.yaml")
    backend = NetworkBackend(cfg.network, device=device, weights_path=str(ckpt))
    params = SearchParams.from_config(cfg.search, training=False)
    params.minibatch_size = 8
    search = PUCTSearch(backend, params)
    print(f"\ncheckpoint {ckpt.name}  device {backend.device}")

    berlin = make_berlin_board()
    diagnose(search, backend, berlin, "Berlin")
    black = berlin.copy(stack=True)
    black.push_uci("d2d4")
    diagnose(search, backend, black, "Berlin + d2d4")


if __name__ == "__main__":
    main()
