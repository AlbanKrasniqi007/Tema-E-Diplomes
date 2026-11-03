#!/usr/bin/env python3
"""Generate self-play training games and optionally save them as .npz."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from lc0.config import load_config, opening_board
from lc0.neural.backend import NetworkBackend
from lc0.search.params import SearchParams
from lc0.selfplay.game import SelfPlayGame

ENGINE_ID = "lc0-standard"


def main() -> None:
    print("=" * 64, flush=True)
    print(f"  ENGINE: {ENGINE_ID}", flush=True)
    print("=" * 64, flush=True)
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(ROOT / "configs" / "default.yaml"))
    parser.add_argument("--games", type=int, default=1)
    parser.add_argument("--visits", type=int, default=None)
    parser.add_argument("--weights", default="")
    parser.add_argument("--out", default="")
    args = parser.parse_args()

    cfg = load_config(args.config)
    backend = NetworkBackend(cfg.network, device=cfg.device, weights_path=args.weights or cfg.weights_path or None)
    params = SearchParams.from_config(cfg.search, training=True)
    visits = args.visits or cfg.train.visits
    worker = SelfPlayGame(
        backend,
        params,
        visits=visits,
        temperature=cfg.train.temperature,
        temp_decay_moves=cfg.train.temp_decay_moves,
        max_ply=cfg.train.max_game_ply,
        start_factory=lambda: opening_board(cfg),
    )
    print(f"opening={cfg.train.opening_name} fen={opening_board(cfg).fen()}")
    all_examples = []
    for i in range(args.games):
        examples, board = worker.play()
        outcome = board.outcome(claim_draw=True)
        print(f"game {i + 1}: {len(examples)} positions, result={outcome.result() if outcome else '*'}")
        all_examples.extend(examples)
    if args.out:
        np.savez_compressed(
            args.out,
            planes=np.stack([e.planes for e in all_examples]),
            policy=np.stack([e.policy for e in all_examples]),
            wdl=np.stack([e.wdl for e in all_examples]),
            moves_left=np.array([e.moves_left for e in all_examples], dtype=np.float32),
        )
        print("wrote", args.out)


if __name__ == "__main__":
    main()
