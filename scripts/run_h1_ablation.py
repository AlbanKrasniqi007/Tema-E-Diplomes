#!/usr/bin/env python3
"""Search-only 2x2 ablation for the Berlin H1 claim.

White is always lc0-standard with classic Q and classic FPU (same as H1).
Black cells, FPU held at classic negation:

  std + wl          already played (control, Black 52/100)
  std + drawscore   DrawScore on the identical net
  dual + wl         dual specialists, chess Q
  dual + drawscore  dual + DrawScore, but now with classic FPU
                    (the original H1 treatment used parent-relative FPU)

No retraining. Snapshot 3440, 120 visits, 100 games.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "berlin_tournament.py"

MATCHES = [
    {
        "name": "A DrawScore-only (standard net)",
        "args": [
            "--white-engine", "standard",
            "--black-engine", "standard",
            "--white-q", "wl",
            "--black-q", "drawscore",
            "--white-fpu", "negation",
            "--black-fpu", "negation",
            "--out-dir", str(ROOT / "tournament" / "ablation_stdW_stdB_drawscore_negfpu_g3440"),
        ],
    },
    {
        "name": "B dual-only (classic Q)",
        "args": [
            "--white-engine", "standard",
            "--black-engine", "variant",
            "--white-q", "wl",
            "--black-q", "wl",
            "--white-fpu", "negation",
            "--black-fpu", "negation",
            "--out-dir", str(ROOT / "tournament" / "ablation_stdW_varB_wl_negfpu_g3440"),
        ],
    },
    {
        "name": "C dual + DrawScore (classic FPU)",
        "args": [
            "--white-engine", "standard",
            "--black-engine", "variant",
            "--white-q", "wl",
            "--black-q", "drawscore",
            "--white-fpu", "negation",
            "--black-fpu", "negation",
            "--out-dir", str(ROOT / "tournament" / "ablation_stdW_varB_drawscore_negfpu_g3440"),
        ],
    },
]


def main() -> None:
    common = [
        sys.executable,
        "-u",
        str(SCRIPT),
        "--games",
        "100",
        "--visits",
        "120",
        "--checkpoint-games",
        "3440",
    ]
    for match in MATCHES:
        print("=" * 72, flush=True)
        print(f"ABLATION {match['name']}", flush=True)
        print("=" * 72, flush=True)
        cmd = common + match["args"]
        proc = subprocess.run(cmd, cwd=str(ROOT))
        if proc.returncode != 0:
            raise SystemExit(f"ablation failed: {match['name']} exit {proc.returncode}")
    print("ALL ABLATION MATCHES DONE", flush=True)


if __name__ == "__main__":
    main()
