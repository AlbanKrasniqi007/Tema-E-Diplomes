# Lc0-standard

Trainable replica of the **Leela Chess Zero** engine as documented in the
[developer section](https://lczero.org/dev/) and the AlphaZero papers.

This is the **control engine** for the thesis: Lc0 classic search + SE-ResNet,
specialized on the **Berlin Defence** the same way LeelaKnightOdds /
LeelaQueenOdds / LeelaRookOdds are trained from a restricted start.

The experimental fork lives in `../lc0-variant`.

Diploma freeze: `checkpoints/run10k/lc0-standard-games003440.pt`

## Berlin Defence start

Every training and engine self-play game begins after:

`1. e4 e5 2. Nf3 Nc6 3. Bb5 Nf6`

FEN: `r1bqkb1r/pppp1ppp/2n2n2/1B2p3/4P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 4 4`

The six opening plies are forced (not learned). The net only trains on play
from that tabiya onward, with those moves kept in the 112-plane history.

Files: `openings/berlin.pgn`, `openings/berlin.fen`.

## Load in Nibbler (play, analyze, self-play)

Nibbler looks for a **program** (`.exe`). Use `Lc0Berlin.exe` in this folder.

1. Nibbler → **Engine → Choose Engine…**
2. Browse to `Lc0Berlin.exe` and open it.
3. After it appears, you can pick it again from **Engine → Known Engine**.

Then:

| Mode | What to do |
| --- | --- |
| **Analyze** | Open any position, or `openings/berlin.pgn`, and start analysis. |
| **Play** | **File → Open PGN** → `openings/berlin.pgn` (board is the Berlin tabiya), then play against the engine. |
| **Self-play** | Same: open `berlin.pgn` first, then Nibbler’s self-play from the current board. |

The UCI name is `lc0-standard (Lc0 Berlin)`. Point **WeightsFile** at
`checkpoints\run10k\lc0-standard-games003440.pt`.

## Train

`configs/default.yaml` lists `train.visits: 160`. Diploma training and matches
used **120** visits.

```bat
cd lc0-standard
python -m pytest tests -q
python -u scripts\run_to_target.py --target-games 10000 --visits 120 --games 8 --out-dir checkpoints\run10k
```

Default net on this PC: **6×64 SE** (RTX 4060 8 GB). Paper-scale: `configs/paper.yaml`.
The diploma freeze remains **3440** even if the loop continues toward 10 000.
