#!/usr/bin/env python3
"""Keep train.py running until progress.json reaches the target game count."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _progress(out_dir: Path) -> int:
    path = out_dir / "progress.json"
    if not path.is_file():
        return 0
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return int(data.get("games_played", 0) or 0)
    except Exception:
        return 0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--target-games", type=int, default=10000)
    parser.add_argument("--games", type=int, default=8)
    parser.add_argument("--visits", type=int, default=120)
    parser.add_argument("--out-dir", default=str(ROOT / "checkpoints" / "run10k"))
    parser.add_argument("--from-scratch", action="store_true")
    args = parser.parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    train_py = ROOT / "scripts" / "train.py"
    first = True
    while True:
        played = _progress(out_dir)
        if played >= args.target_games:
            print(f"run_to_target: already {played}/{args.target_games} — stopping", flush=True)
            return
        cmd = [
            sys.executable,
            "-u",
            str(train_py),
            "--target-games",
            str(args.target_games),
            "--games",
            str(args.games),
            "--visits",
            str(args.visits),
            "--out-dir",
            str(out_dir),
        ]
        if first and args.from_scratch and played == 0:
            cmd.append("--from-scratch")
        first = False
        print(f"run_to_target: launching ({played}/{args.target_games}) {' '.join(cmd)}", flush=True)
        result = subprocess.run(cmd, cwd=str(ROOT))
        played = _progress(out_dir)
        if played >= args.target_games:
            print(f"run_to_target: DONE {played}/{args.target_games}", flush=True)
            return
        print(
            f"run_to_target: train.py exited {result.returncode} at {played}/{args.target_games}; "
            "restarting in 20s",
            flush=True,
        )
        time.sleep(20)


if __name__ == "__main__":
    main()
