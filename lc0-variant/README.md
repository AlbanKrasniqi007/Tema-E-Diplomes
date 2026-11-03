# Lc0-variant

Experimental engine for the thesis. Same Berlin Defence start as
`lc0-standard`, but this is **two agents co-evolving in one process**, not a
single net with remapped Q.

- **White specialist:** plays only White, draw-seeking in search (win = draw),
  and trains only on White positions.
- **Black specialist:** plays only Black from the Berlin, win-seeking in search
  (draw = loss), and trains only on Black positions.
- They share games. One UCI executable; `DualNetworkBackend` routes each leaf
  by side-to-move. `lc0-standard` remains the symmetric one-net control.

FPU stays parent-relative (\(Q - fpu\sqrt{P}\)). Search uses Lc0 DrawScore
for each color’s `Q`; the value head is trained on real win/draw/loss.
Details: [VARIANT.md](VARIANT.md).

Diploma freeze: `checkpoints/run10k/lc0-variant-games003440.pt`

`DualNetworkBackend.load` still clones an old single-net `.pt` into both
specialists if you point it at one. The diploma checkpoint is already dual.

## Nibbler

`Lc0Berlin-Variant.exe` in this folder.

Then **File → Open PGN** → `openings\berlin.pgn` for play, analysis, or
self-play from the Berlin tabiya.

If you play Black, you face the White specialist; if you play White, you face
the Black specialist.

Point **WeightsFile** at `checkpoints\run10k\lc0-variant-games003440.pt`.

## Train

```bat
cd lc0-variant
python -m pytest tests -q
python -u scripts\run_to_target.py --target-games 10000 --visits 120 --games 8 --out-dir checkpoints\run10k
```

The diploma freeze remains **3440** even if the loop continues toward 10 000.
Later checkpoints must not enter H1.
