# Nibbler

## Engine program

`Lc0Berlin-Aggressive.exe` is a Windows UCI launcher. Nibbler’s file picker
lists `.exe` files, so choose that — not the `.py` or `.cmd` helpers.

Path (from this folder):

```
Lc0Berlin-Aggressive.exe
```

Rebuild (from this folder):

```bat
python tools\build_launcher.py Lc0Berlin-Aggressive.exe
```

UCI `id name` is still `lc0-winonly (Lc0 Berlin H3)` (cloned from H3).

## Play / analyze / self-play from the Berlin

1. **File → Open PGN…** → `openings\berlin.pgn`
2. You should see White to move after `3…Nf6`.
3. Analyze, play vs engine, or run Nibbler self-play from that position.

Point **WeightsFile** to the diploma freeze
`checkpoints\run10k\lc0-aggressive-games003440.pt`.
