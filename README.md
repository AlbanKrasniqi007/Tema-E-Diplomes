# Tema e Diplomës — Berlin-specialist Lc0 engines

Bachelor thesis: **Strukturat asimetrike të shpërblimit në inteligjencën artificiale në shah: mësimi përforcues për Mbrojtjen e Berlinit**.

Student Alban Krasniqi, UNI–Universum International College. Mentor: Prof. Ass. Dr. Rrezart Prebreza.

All engines are **UCI programs** Nibbler can load as `.exe`. Training and matches start from
`1. e4 e5 2. Nf3 Nc6 3. Bb5 Nf6` (ECO C65). Freeze for every diploma comparison:
**games003440.pt**, **120 visits**, **N = 100**, fixed colours.

**Full technical record, including every retained score:** [docs/PROJECT.md](docs/PROJECT.md)

| Program | Role |
| --- | --- |
| [`lc0-standard/Lc0Berlin.exe`](lc0-standard/) | H1 control: one net, \(Q=W-L\), classic FPU |
| [`lc0-variant/Lc0Berlin-Variant.exe`](lc0-variant/) | H1 packaged variant: dual nets, DrawScore Q, parent FPU |
| [`lc0-winonly/Lc0Berlin-WinOnly.exe`](lc0-winonly/) | H3: one net, White \(Q=W-L\), Black \(Q=W\) — **not mixed into H1** |
| [`lc0-aggressive/Lc0Berlin-Aggressive.exe`](lc0-aggressive/) | H4 match: as White with `WhiteQ=winonly` (\(Q=W\)). `default.yaml` is still `asymmetric` |

Headline diploma result: **H1 rejected as harm**. Variant Black 29/100 vs standard Black 52/100
(\(\Delta s=-0.23\), 95% CI \([-0.332,-0.128]\)). As White vs the same standard Black: standard 48,
variant 35.5, aggressive 26.

## Layout

```
lc0-standard/   lc0-variant/   lc0-winonly/   lc0-aggressive/
scripts/        berlin_tournament.py, run_h1_ablation.py, run_h3_match.py, analyze_tournament_pgns.py
tournament/     PGN + results.json for every retained match
docs/PROJECT.md
```

Engine folders ignore `checkpoints/` and `*.pt`. The four freeze files
`*/checkpoints/run10k/*-games003440.pt` are in git. Intermediate training
snapshots are not. Faculty Word files are not in this repository.

## Nibbler

**Engine → Choose Engine…** → pick the `.exe` in that engine folder, then
**File → Open PGN…** → `openings/berlin.pgn`. Point **WeightsFile** at
`checkpoints\run10k\*-games003440.pt` in the same folder.

## Tests

```bat
python -m pytest lc0-standard\tests -q
python -m pytest lc0-variant\tests -q
python -m pytest lc0-winonly\tests -q
python -m pytest lc0-aggressive\tests -q
```

Official Lc0 references used for the replica:

- [Developer docs](https://lczero.org/dev/)
- [AlphaZero primer](https://lczero.org/dev/lc0/search/alphazero/)
- [Neural network topology](https://lczero.org/dev/old/nn/)
