# Nibbler

## Engine program

`Lc0Berlin.exe` is a Windows UCI program. Nibbler’s file picker lists `.exe`
files, so choose that — not the `.py` or `.cmd` helpers.

Path (from this folder):

```
Lc0Berlin.exe
```

Variant engine: `../lc0-variant/Lc0Berlin-Variant.exe`

Rebuild (from the engine folder):

```bat
python tools\build_launcher.py
```

## Play / analyze / self-play from the Berlin

Training always starts from the Berlin tabiya. Nibbler owns the visible board,
so for play and self-play open the supplied PGN first:

1. **File → Open PGN…** → `openings\berlin.pgn`
2. You should see White to move after `3…Nf6`.
3. Analyze, play vs engine, or run Nibbler self-play from that position.

You can also paste the FEN from `openings\berlin.fen`.

Point **WeightsFile** to the diploma freeze
`checkpoints\run10k\lc0-standard-games003440.pt`.
