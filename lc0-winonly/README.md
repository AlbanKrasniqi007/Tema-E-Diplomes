# Lc0-winonly (H3)

Third thesis engine: a clone of `lc0-standard` with **one** 6×64 SE-ResNet
(not the dual-net variant). The value head still trains on **true chess WDL**.
Only the PUCT scalar changes:

| Side  | Search Q | FPU |
| --- | --- | --- |
| White | \(Q = W - L\) | classic \( -Q - 0.330\sqrt{P} \) |
| Black | \(Q = W\) (win +1, draw 0, loss 0) | same classic FPU |

Black \(Q = W\) ranks moves the same way as DrawScore Black (\(2W-1\)), but
the values stay near zero so classic FPU is not mixed with parent-relative FPU
or a dual net. That packaged combination is `lc0-variant` and belongs to H1;
do not put these weights into H1.

Every training game starts from the Berlin tabiya
`1. e4 e5 2. Nf3 Nc6 3. Bb5 Nf6`.

Diploma freeze: `checkpoints/run10k/lc0-winonly-games003440.pt`

## Nibbler

`Lc0Berlin-WinOnly.exe` in this folder.

Open `openings/berlin.pgn` first. Point **WeightsFile** at
`checkpoints\run10k\lc0-winonly-games003440.pt`.

## Train

```bat
cd lc0-winonly
python -m pytest tests -q
python -u scripts\run_to_target.py --target-games 10000 --visits 120 --games 8 --out-dir checkpoints\run10k --from-scratch
```

Equal-budget freeze for the H3 match vs standard is **3440 games**, same
protocol as H1 (120 visits, N=100). Later checkpoints must not enter H1.
