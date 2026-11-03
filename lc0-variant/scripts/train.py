#!/usr/bin/env python3
"""AlphaZero / Lc0 RL loop: co-play two specialists, then train each on its color."""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

import chess
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from lc0.config import load_config, opening_board
from lc0.neural.backend import count_parameters
from lc0.neural.dual import DualNetworkBackend
from lc0.search.params import SearchParams
from lc0.selfplay.game import SelfPlayGame
from lc0.training.trainer import Trainer

ENGINE_ID = "lc0-variant"

_EMPTY_METRICS = {"loss": 0.0, "policy": 0.0, "value": 0.0, "mlh": 0.0, "steps": 0.0}


def _maybe_train(trainer: Trainer, batch_size: int) -> dict[str, float]:
    """Skip a color when its replay cannot fill a batch (avoids a useless/crashy step)."""
    if len(trainer.buffer) < batch_size:
        return dict(_EMPTY_METRICS)
    return trainer.train_steps()


def _seed_everything(seed: int) -> None:
    if seed <= 0:
        return
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _write_progress(out_dir: Path, payload: dict) -> None:
    path = out_dir / "progress.json"
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _save_dual(
    dual: DualNetworkBackend,
    white_trainer: Trainer,
    black_trainer: Trainer,
    cfg,
    path: Path,
    *,
    games_played: int,
    iteration: int,
    visits: int,
    target_games: int,
) -> None:
    dual.save(
        str(path),
        extra={
            "white": {
                "optimizer": white_trainer.optimizer.state_dict(),
                "step": white_trainer.step_count,
            },
            "black": {
                "optimizer": black_trainer.optimizer.state_dict(),
                "step": black_trainer.step_count,
            },
            "config": cfg.to_dict(),
            "games_played": games_played,
            "iteration": iteration,
            "visits": visits,
            "target_games": target_games,
        },
    )


def main() -> None:
    print("=" * 64, flush=True)
    print(f"  ENGINE: {ENGINE_ID}", flush=True)
    print("  ROLE:   dual co-evolution (two specialists, one process)", flush=True)
    print("  AGENTS: White draw-seeking | Black win-seeking (Berlin)", flush=True)
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
        help="Ignore yaml/WeightsFile and any latest.pt in --out-dir; start random twins.",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    _seed_everything(cfg.seed)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    latest_path = out_dir / "lc0-variant-latest.pt"
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

    dual = DualNetworkBackend(
        cfg.network,
        device=cfg.device,
        weights_path=str(resume_path) if resume_path else None,
    )
    games_played = int(getattr(dual, "meta_games_played", 0) or 0) if resume_path else 0
    iteration = int(getattr(dual, "meta_iteration", 0) or 0) if resume_path else 0
    print(
        f"device={dual.device} agents=white-specialist,black-specialist "
        f"params_each={count_parameters(dual.white.model):,} "
        f"cloned_from_single={dual.cloned_from_single} from_scratch={args.from_scratch}",
        flush=True,
    )
    if dual.weights_path:
        print(f"weights={dual.weights_path} games_played={games_played} iteration={iteration}", flush=True)
    else:
        print("weights=NONE (random twin specialists)", flush=True)
    params = SearchParams.from_config(cfg.search, training=True)
    start_board = opening_board(cfg)
    print(
        f"opening={cfg.train.opening_name} fen={start_board.fen()} "
        f"pgn={cfg.train.opening_pgn}",
        flush=True,
    )
    print(
        f"value_goals=white:win=draw black:draw=loss visits={visits} "
        f"games_per_iter={games_n} target_games={target_games} "
        f"checkpoint_every={checkpoint_every} iter(s)",
        flush=True,
    )
    worker = SelfPlayGame(
        dual,
        params,
        visits=visits,
        temperature=cfg.train.temperature,
        temp_decay_moves=cfg.train.temp_decay_moves,
        max_ply=cfg.train.max_game_ply,
        start_factory=lambda: opening_board(cfg),
    )
    white_trainer = Trainer(dual.white, cfg)
    black_trainer = Trainer(dual.black, cfg)
    dual.restore_optimizers(white_trainer, black_trainer)
    batch_size = cfg.train.batch_size
    started = time.perf_counter()

    if games_played >= target_games:
        print(f"{ENGINE_ID} | already at {games_played}/{target_games} games — nothing to do", flush=True)
        return

    while games_played < target_games:
        iteration += 1
        white_collected = 0
        black_collected = 0
        games_this_iter = 0
        for g in range(games_n):
            if games_played >= target_games:
                break
            examples, board = worker.play()
            white_ex = [e for e in examples if e.stm == chess.WHITE]
            black_ex = [e for e in examples if e.stm == chess.BLACK]
            white_trainer.add_games(white_ex)
            black_trainer.add_games(black_ex)
            white_collected += len(white_ex)
            black_collected += len(black_ex)
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
                f"{len(examples)} pos (W={len(white_ex)} B={len(black_ex)}), "
                f"{outcome.result() if outcome else '*'} | "
                f"{rate:.1f} games/h eta={eta_h:.1f}h",
                flush=True,
            )
        if games_this_iter == 0:
            break
        white_metrics = _maybe_train(white_trainer, batch_size)
        black_metrics = _maybe_train(black_trainer, batch_size)
        if len(white_trainer.buffer) < batch_size:
            print(
                f"{ENGINE_ID} | iter {iteration} skip white train: "
                f"replay {len(white_trainer.buffer)} < batch_size {batch_size}",
                flush=True,
            )
        if len(black_trainer.buffer) < batch_size:
            print(
                f"{ENGINE_ID} | iter {iteration} skip black train: "
                f"replay {len(black_trainer.buffer)} < batch_size {batch_size}",
                flush=True,
            )
        save_numbered = iteration % checkpoint_every == 0 or games_played >= target_games
        extra_kwargs = dict(
            games_played=games_played,
            iteration=iteration,
            visits=visits,
            target_games=target_games,
        )
        _save_dual(dual, white_trainer, black_trainer, cfg, latest_path, **extra_kwargs)
        numbered = out_dir / f"lc0-variant-games{games_played:06d}.pt"
        if save_numbered:
            _save_dual(dual, white_trainer, black_trainer, cfg, numbered, **extra_kwargs)
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
            f"{ENGINE_ID} | iter {iteration} agents=white-specialist,black-specialist | "
            f"games={games_played}/{target_games} "
            f"white_pos={white_collected} black_pos={black_collected} | "
            f"white_loss={white_metrics['loss']:.4f} black_loss={black_metrics['loss']:.4f} "
            f"white_policy={white_metrics['policy']:.4f} black_policy={black_metrics['policy']:.4f} "
            f"white_value={white_metrics['value']:.4f} black_value={black_metrics['value']:.4f} "
            f"| saved {saved_msg}",
            flush=True,
        )

    print(f"{ENGINE_ID} | DONE {games_played}/{target_games} games at {visits} visits", flush=True)


if __name__ == "__main__":
    main()
