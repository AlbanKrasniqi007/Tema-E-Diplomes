#!/usr/bin/env python3
"""AlphaZero / Lc0 RL loop: self-play, then train, then checkpoint."""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from lc0.config import load_config, opening_board
from lc0.neural.backend import NetworkBackend, count_parameters
from lc0.search.params import SearchParams
from lc0.selfplay.game import SelfPlayGame
from lc0.training.trainer import Trainer

ENGINE_ID = "lc0-winonly"


def _seed_everything(seed: int) -> None:
    if seed <= 0:
        return
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _write_progress(out_dir: Path, payload: dict) -> None:
    (out_dir / "progress.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _restore_trainer(trainer: Trainer, path: Path) -> tuple[int, int]:
    try:
        blob = torch.load(path, map_location="cpu", weights_only=False)
    except Exception:
        return 0, 0
    if not isinstance(blob, dict):
        return 0, 0
    if "optimizer" in blob:
        trainer.optimizer.load_state_dict(blob["optimizer"])
    if "step" in blob:
        trainer.step_count = int(blob["step"])
    games_played = int(blob.get("games_played", 0) or 0)
    iteration = int(blob.get("iteration", 0) or 0)
    return games_played, iteration


def main() -> None:
    print("=" * 64, flush=True)
    print(f"  ENGINE: {ENGINE_ID}", flush=True)
    print("  ROLE:   H3 (one net; White Q=W-L; Black Q=W; classic FPU)", flush=True)
    print("=" * 64, flush=True)
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(ROOT / "configs" / "default.yaml"))
    parser.add_argument("--iterations", type=int, default=None)
    parser.add_argument("--target-games", type=int, default=None)
    parser.add_argument("--games", type=int, default=None)
    parser.add_argument("--visits", type=int, default=None)
    parser.add_argument("--weights", default="")
    parser.add_argument("--out-dir", default=str(ROOT / "checkpoints"))
    parser.add_argument(
        "--from-scratch",
        action="store_true",
        help="Ignore yaml/WeightsFile and any latest.pt in --out-dir; start a random net.",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    _seed_everything(cfg.seed)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    latest_path = out_dir / "lc0-winonly-latest.pt"
    games_n = args.games or cfg.train.games_per_iteration
    visits = args.visits or cfg.train.visits
    target_games = args.target_games
    if target_games is None:
        iters = args.iterations if args.iterations is not None else 1
        target_games = iters * games_n
    checkpoint_every = max(1, int(cfg.train.checkpoint_every))

    resume_path = None
    if not args.from_scratch:
        if args.weights:
            resume_path = Path(args.weights)
        elif latest_path.is_file():
            resume_path = latest_path

    backend = NetworkBackend(
        cfg.network,
        device=cfg.device,
        weights_path=str(resume_path) if resume_path else None,
    )
    print(f"device={backend.device} params={count_parameters(backend.model):,}", flush=True)
    trainer = Trainer(backend, cfg)
    games_played = 0
    iteration = 0
    if resume_path:
        games_played, iteration = _restore_trainer(trainer, resume_path)
        print(
            f"weights={resume_path} games_played={games_played} iteration={iteration} "
            f"from_scratch={args.from_scratch}",
            flush=True,
        )
    else:
        print("weights=NONE (random untrained net) from_scratch=True", flush=True)

    params = SearchParams.from_config(cfg.search, training=True)
    start_board = opening_board(cfg)
    print(
        f"opening={cfg.train.opening_name} fen={start_board.fen()} "
        f"pgn={cfg.train.opening_pgn}",
        flush=True,
    )
    print(
        f"value=true WDL search=asymmetric (W: W-L, B: W) fpu=negation visits={visits} games_per_iter={games_n} "
        f"target_games={target_games} checkpoint_every={checkpoint_every} iter(s)",
        flush=True,
    )
    worker = SelfPlayGame(
        backend,
        params,
        visits=visits,
        temperature=cfg.train.temperature,
        temp_decay_moves=cfg.train.temp_decay_moves,
        max_ply=cfg.train.max_game_ply,
        start_factory=lambda: opening_board(cfg),
    )
    started = time.perf_counter()

    if games_played >= target_games:
        print(f"{ENGINE_ID} | already at {games_played}/{target_games} games — nothing to do", flush=True)
        return

    while games_played < target_games:
        iteration += 1
        collected = 0
        games_this_iter = 0
        for g in range(games_n):
            if games_played >= target_games:
                break
            examples, board = worker.play()
            trainer.add_games(examples)
            collected += len(examples)
            games_played += 1
            games_this_iter += 1
            outcome = board.outcome(claim_draw=True)
            elapsed = time.perf_counter() - started
            rate = games_played / elapsed * 3600 if elapsed > 1 else 0.0
            remaining = max(target_games - games_played, 0)
            eta_h = remaining / rate if rate > 0 else 0.0
            print(
                f"{ENGINE_ID} | iter {iteration} game {g + 1}/{games_n} | "
                f"total {games_played}/{target_games} | "
                f"{len(examples)} pos, {outcome.result() if outcome else '*'} | "
                f"{rate:.1f} games/h eta={eta_h:.1f}h",
                flush=True,
            )
        if games_this_iter == 0:
            break
        metrics = trainer.train_steps()
        extra = {
            "games_played": games_played,
            "iteration": iteration,
            "visits": visits,
            "target_games": target_games,
        }
        trainer.save(latest_path, extra=extra)
        numbered = out_dir / f"lc0-winonly-games{games_played:06d}.pt"
        save_numbered = iteration % checkpoint_every == 0 or games_played >= target_games
        if save_numbered:
            trainer.save(numbered, extra=extra)
            saved_msg = str(numbered)
        else:
            saved_msg = f"{latest_path} (latest only)"
        _write_progress(
            out_dir,
            {
                "games_played": games_played,
                "iteration": iteration,
                "target_games": target_games,
                "visits": visits,
                "latest": str(latest_path),
                "checkpoint": saved_msg,
            },
        )
        print(
            f"{ENGINE_ID} | iter {iteration} | games={games_played}/{target_games} "
            f"pos={collected} | loss={metrics['loss']:.4f} policy={metrics['policy']:.4f} "
            f"value={metrics['value']:.4f} mlh={metrics['mlh']:.4f} | saved {saved_msg}",
            flush=True,
        )

    print(f"{ENGINE_ID} | DONE {games_played}/{target_games} games at {visits} visits", flush=True)


if __name__ == "__main__":
    main()
