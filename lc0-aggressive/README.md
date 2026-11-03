# Lc0-aggressive (H4)

Clone of `lc0-winonly`, used as **White** in the H4 diploma match. One 6×64
SE-ResNet (not the dual-net variant). The value head still trains on **true
chess WDL**.

`configs/default.yaml` is still `q_mode: asymmetric` (White \(Q=W-L\), Black
\(Q=W\)), classic FPU. Unit tests assert that. The retained H4 match
**overrides** search: PGN headers `WhiteQ=winonly` (\(Q=W\)), `BlackQ=wl`.

Do **not** mix these weights into H1.

The Windows program is `Lc0Berlin-Aggressive.exe`. The UCI `id name` is still
`lc0-winonly (Lc0 Berlin H3)` because this folder was cloned from H3.

Diploma freeze: `checkpoints/run10k/lc0-aggressive-games003440.pt`

Every training game starts from the Berlin tabiya
`1. e4 e5 2. Nf3 Nc6 3. Bb5 Nf6`.

## Nibbler

`Lc0Berlin-Aggressive.exe` in this folder.

Open `openings/berlin.pgn` first. Point **WeightsFile** at
`checkpoints\run10k\lc0-aggressive-games003440.pt`.

## Train

```bat
cd lc0-aggressive
python -m pytest tests -q
python -u scripts\run_to_target.py --target-games 10000 --visits 120 --games 8 --out-dir checkpoints\run10k
```

The diploma freeze remains **3440** even if the loop continues toward 10 000.
Later checkpoints must not enter H1.
