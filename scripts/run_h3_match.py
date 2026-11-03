#!/usr/bin/env python3
"""H3 Berlin match: lc0-standard White vs lc0-winonly Black.

Equal freeze 3440, 120 visits, N=100, 8 temperature plies then greedy.
Compares Black score with the existing H1 pairings vs the same White.
Does not reopen H1 or use later unequal checkpoints.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "berlin_tournament.py"


def main() -> None:
    cmd = [
        sys.executable,
        "-u",
        str(SCRIPT),
        "--games",
        "100",
        "--visits",
        "120",
        "--checkpoint-games",
        "3440",
        "--white-engine",
        "standard",
        "--black-engine",
        "winonly",
        "--white-q",
        "wl",
        "--black-q",
        "asymmetric",
        "--white-fpu",
        "negation",
        "--black-fpu",
        "negation",
        "--out-dir",
        str(ROOT / "tournament" / "berlin_std_white_vs_winonly_black_g3440"),
    ]
    proc = subprocess.run(cmd, cwd=str(ROOT))
    if proc.returncode != 0:
        raise SystemExit(f"H3 match failed exit {proc.returncode}")
    print("H3 MATCH DONE", flush=True)


if __name__ == "__main__":
    main()
