# Nibbler

## Engine program

`Lc0Berlin-Variant.exe` is a Windows UCI program. Nibbler’s file picker lists
`.exe` files, so choose that — not the `.py` or `.cmd` helpers.

Path (from this folder):

```
Lc0Berlin-Variant.exe
```

Control engine: `../lc0-standard/Lc0Berlin.exe`

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

Analysis uses the same dual specialists as training: **White is draw-seeking**,
**Black is win-seeking** in search (`Q`). The UCI `wdl` line is **chess**
win/draw/loss from the side to move, so Nibbler expected score should stay
near 50% on a drawish Berlin instead of jumping ~90% → ~4% after one ply.

Nibbler treats this program as Lc0 (`id name` contains `Lc0`). It already
sends **MultiPV 500** on its own. Do **not** set MultiPV in the Engine menu —
that only prints the blue “MultiPV should be 500 for this engine” warning
and does nothing.

To see candidate moves: **Analysis → Go** (or infinite). Leave MultiPV alone.

Point **WeightsFile** to the diploma freeze
`checkpoints\run10k\lc0-variant-games003440.pt`.
