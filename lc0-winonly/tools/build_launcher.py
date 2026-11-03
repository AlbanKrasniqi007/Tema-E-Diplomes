"""Compile Lc0Berlin.exe so Nibbler can browse and load a real program."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CSC = Path(r"C:\Windows\Microsoft.NET\Framework64\v4.0.30319\csc.exe")
CS = ROOT / "tools" / "NibblerLauncher.cs"


def main() -> None:
    out_name = sys.argv[1] if len(sys.argv) > 1 else "Lc0Berlin-WinOnly.exe"
    out = ROOT / out_name
    if not CSC.exists():
        raise SystemExit(f"csc.exe not found at {CSC}")
    python = Path(sys.executable).resolve()
    (ROOT / "python.path").write_text(str(python), encoding="utf-8")
    cmd = [
        str(CSC),
        "/nologo",
        "/optimize+",
        "/target:exe",
        f"/out:{out}",
        str(CS),
    ]
    print("compiling", out)
    subprocess.check_call(cmd)
    print("python.path", python)
    print("wrote", out)


if __name__ == "__main__":
    main()
