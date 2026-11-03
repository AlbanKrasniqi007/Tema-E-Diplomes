# Project documentation — Berlin asymmetric self-play

Student: **Alban Krasniqi**  
Programme: Bachelor, Computer Science, UNI–Universum International College  
Mentor: **Prof. Ass. Dr. Rrezart Prebreza**  
Thesis title: *Strukturat asimetrike të shpërblimit në inteligjencën artificiale në shah: mësimi përforcues për Mbrojtjen e Berlinit*  
(*Asymmetric Reward Structures in Artificial Intelligence in Chess: Reinforcement Learning for the Berlin Defence*)

This file is the technical record of the repository: what was built, how it was frozen, which matches count, and which numbers are the diploma results. It does **not** replace the Word thesis. The thesis is the faculty document; this is the lab notebook.

---

## 1. What this project is

Four Python UCI engines, each a **trainable replica** of documented Leela Chess Zero *classic* search + `NETWORK_SE_WITH_HEADFORMAT` (not a rebuild of C++ `lc0.exe`). All self-play and all evaluation games start from the same Berlin Defence tabiya after `1.e4 e5 2.Nf3 Nc6 3.Bb5 Nf6`.

FEN \(s_0\) (ECO **C65**):

```
r1bqkb1r/pppp1ppp/2n2n2/1B2p3/4P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 4 4
```

The six opening plies are forced. The network only learns play from that position onward, with those moves kept in the 112-plane history.

The scientific claim is narrow. **H1** asked whether a packaged asymmetric search objective (dual nets + Lc0 DrawScore + parent-relative FPU), trained from this FEN with an equal game budget, would raise Black’s FIDE score against the same standard White. It did not. The classical \(Q = W - L\) control scored more.

Official Lc0 references used while writing the replica:

- <https://lczero.org/dev/>
- <https://lczero.org/dev/lc0/search/alphazero/>
- <https://lczero.org/dev/old/nn/>

---

## 2. Repository layout

```
Tema-E-Diplomes/
├── README.md
├── docs/PROJECT.md        this file
├── lc0-standard/          H1 control engine
├── lc0-variant/           H1 packaged dual-net + DrawScore engine
├── lc0-winonly/           H3 one-net, asymmetric Q (not in the H1 corpus)
├── lc0-aggressive/        H4 checkpoint; match uses WhiteQ=winonly
├── scripts/               tournaments, ablation, H3 match, PGN stats
├── tournament/            PGN + results.json for every match
├── Diploma versions/      submitted thesis and proposal files
├── Punim_Diplome_Veteluajtja_Asimetrike_Berlin.docx
└── Tema_Diplomes_Alban final.docx
```

Root `scripts/` holds only the four match/analysis programs listed below. There is no `thesis_figures/` folder; the figures live inside the Word files.

Each engine folder follows the same shape:

| Path | Role |
| --- | --- |
| `src/lc0/chess/` | Board, FEN, 1858-move policy map, Berlin opening |
| `src/lc0/search/` | Classic PUCT / MCTS |
| `src/lc0/neural/` | SE-ResNet backend (`dual.py` only in the variant) |
| `src/lc0/selfplay/` | Self-play games |
| `src/lc0/training/` | Policy CE + WDL CE + moves-left MSE |
| `src/lc0/uci/` | UCI loop |
| `src/lc0/value.py` | `search_q` (WDL → PUCT scalar) |
| `configs/default.yaml` | 6×64 SE, Berlin FEN (diploma eval: 120 visits) |
| `scripts/train.py` | One training cycle |
| `scripts/run_to_target.py` | Loop until a game-count target |
| `scripts/uci.py` / `*.exe` | GUI / Cute Chess entry |
| `tests/test_engine.py` | Unit tests |
| `checkpoints/run10k/` | Training run; freeze file `*-games003440.pt` |
| `openings/berlin.pgn` | Tabia for Nibbler |

Root helper scripts:

| Script | Role |
| --- | --- |
| `scripts/berlin_tournament.py` | Fixed-colour N-game match, PGN + `results.json` |
| `scripts/run_h1_ablation.py` | Search-only 2×2 (cells A–C) |
| `scripts/run_h3_match.py` | Standard White vs winonly Black |
| `scripts/analyze_tournament_pgns.py` | Termination, 4th move, ply, C67 / Qxd8 |

---

## 3. Engines and search Q

Every line uses the same body: **6×64 SE-ResNet**, 16 SE channels, 112 input planes, policy 1858, WDL value head, moves-left head. Default training/eval: **120 MCTS visits**, AdamW \(10^{-3}\), batch 128, 8 games per iteration, RTX **4060 8 GB**.

PUCT constants (documented Lc0 classic): \(c_{\mathrm{init}}=1.745\), \(c_{\mathrm{base}}=38739\), \(c_{\mathrm{factor}}=3.894\). FPU reduction \(0.330\sqrt{P}\).

The **value head always learns true chess WDL**. Asymmetry is applied only when WDL is converted to the PUCT scalar \(Q\).

| Engine | Nets | Native search Q | Native FPU | Diploma role |
| --- | --- | --- | --- | --- |
| `lc0-standard` | 1 | \(Q=W-L\) both colours | classic \(-Q-0.330\sqrt{P}\) | H1 control |
| `lc0-variant` | 2 specialists (`DualNetworkBackend`) | DrawScore: White \(1-2L\), Black \(2W-1\) | parent-relative \(Q-0.330\sqrt{P}\) | H1 treatment |
| `lc0-winonly` | 1 | White \(W-L\), Black \(Q=W\) (`q_mode=asymmetric`) | classic | H3; **not** mixed into H1 |
| `lc0-aggressive` | 1 | Tournament native Q is `winonly` (\(Q=W\)); `configs/default.yaml` is still `q_mode: asymmetric` | classic | H4 as **White** with `WhiteQ=winonly` |

DrawScore (Lc0): \(Q=(W-L)+d_s D\) with \(d_s=+1\) for White and \(d_s=-1\) for Black.

Win-only: \(Q=W\). A draw and a loss are both \(0\).

**Honesty about H4.** `lc0-aggressive/configs/default.yaml` still says `q_mode: asymmetric` (same lineage as H3). Unit tests assert that. The retained diploma match **overrides** search: PGN headers `WhiteQ=winonly`, `BlackQ=wl`. Do not write “both colours used \(Q=W\) during self-play” unless a future training run actually changes the yaml.

UCI binaries (Nibbler: **Engine → Choose Engine…**). Paths are relative to the repo root:

- `lc0-standard/Lc0Berlin.exe` — UCI name `lc0-standard (Lc0 Berlin)`
- `lc0-variant/Lc0Berlin-Variant.exe` — UCI name `lc0-variant (Lc0 Berlin dual co-evolution)`
- `lc0-winonly/Lc0Berlin-WinOnly.exe` — UCI name `lc0-winonly (Lc0 Berlin H3)`
- `lc0-aggressive/Lc0Berlin-Aggressive.exe` — Windows file name is Aggressive; the UCI `id name` is still `lc0-winonly (Lc0 Berlin H3)` because this folder was cloned from H3

Open each engine’s `openings/berlin.pgn` first. Point **WeightsFile** at that engine’s `checkpoints/run10k/*-games003440.pt`.

---

## 4. Frozen evaluation protocol

Locked **before** looking at H1 scores, and not changed afterwards:

| Knob | Frozen value |
| --- | --- |
| Start | C65 FEN above |
| Checkpoint | `games003440.pt` (equal budget, not the 10 000 script ceiling) |
| Visits | 120 / move, training and matches |
| Match length | \(N=100\) |
| Colours | **Fixed** (no colour swap inside a match) |
| Temperature | \(\tau=1\) for **8 plies after Berlin**, then greedy |
| Contempt | off |
| Book | only the six forced plies |
| Hardware | NVIDIA RTX 4060 8 GB |
| Decision | Tabela 7 in the thesis: 95% CI of \(\Delta s\) entirely \(>0\) supports H1; entirely \(<0\) rejects H1 as harm |

Checkpoint files used in every diploma match:

```
lc0-standard/checkpoints/run10k/lc0-standard-games003440.pt
lc0-variant/checkpoints/run10k/lc0-variant-games003440.pt
lc0-aggressive/checkpoints/run10k/lc0-aggressive-games003440.pt
lc0-winonly/checkpoints/run10k/lc0-winonly-games003440.pt   # H3 only
```

The `run_to_target.py` ceiling of 10 000 games is a loop cap. It is **not** the evaluation \(N\) and was **not** used as the freeze.

`lc0-standard/configs/default.yaml` still lists `train.visits: 160`. Diploma training and matches used **120** via `--visits 120`. The other three engines already have `visits: 120` in yaml.

Git: each engine `.gitignore` drops `checkpoints/` and `*.pt`. Only the four freeze files above are tracked. Intermediate snapshots stay on the training machine.

Each PGN carries `White`, `Black`, `WhiteWeights`, `BlackWeights`, `TimeControl` `120 nodes/move`, `Seed`, `BerlinProof`, `WhiteQ`, `BlackQ`, `WhiteFPU`, `BlackFPU`. `scripts/berlin_tournament.py` rejects a game if the first six UCI moves are not `e2e4 e7e5 g1f3 b8c6 f1b5 g8f6`.

---

## 5. Hypotheses

| Id | Statement | Status in the diploma |
| --- | --- | --- |
| **H0** | Packaged asymmetry does not raise Black’s score: \(s_{\mathrm{var}}\le s_{\mathrm{std}}\) (both as Black vs the same standard White) | Not rejected; H1 is |
| **H1** | \(s_{\mathrm{var}}>s_{\mathrm{std}}\) as Black vs standard White | **Rejected as harm** |
| **H2** | Variant Black produces a higher decisive-game rate than the standard–standard control | **Held**, secondary, no frozen table like H1 |
| **H3** | One-net win-only Black (\(Q=W\)) vs standard White | Match exists on disk; **not** part of the H1 corpus |
| **H4** | Aggressive as White with `WhiteQ=winonly` vs the same standard Black used in the variant–standard auxiliary match | **Run and reported**; labelled, not a reopening of H1 |

H1 metric: game score \(s=(W+0.5D)/N\) of the tested side. \(\Delta s\) is the difference of two independent sample means (\(N=100\) vs \(N=100\)). Normal 95% CI uses the empirical per-game variance. Games after eight temperature plies are only partly independent, so that SE is a **lower bound** on uncertainty.

---

## 6. Diploma results (cite these)

All rows: freeze 3440, 120 visits, \(N=100\), fixed colours, \(\tau=1\) for 8 plies then greedy. W/D/L are from the **named side**.

### 6.1 H1 — Black vs the same standard White

Directory: `tournament/berlin_std_white_vs_var_black_g3440`  
Control: `tournament/berlin_std_white_vs_std_black_g3440`

| Side (as Black) | W | D | L | \(s\) | Scoreline (White–Black) |
| --- | ---: | ---: | ---: | ---: | --- |
| lc0-variant | 14 | 30 | 56 | **0.29** | standard 71 – 29 variant |
| lc0-standard | 29 | 46 | 25 | **0.52** | standard 48 – 52 standard |

\[
\Delta s = 0.29 - 0.52 = -0.23
\]

95% CI **[−0.332, −0.128]**, \(z \approx -4.44\). Bootstrap (20 000 resamples, seed 2026): **[−0.330, −0.130]**.

**Decision:** upper bound \(<0\) → H1 rejected in the harm direction. Variant Black scored **23 points fewer** per 100 games.

### 6.2 H2 — decisive games (same two matches)

| Match | Draw rate \(d\) | Decisive \(1-d\) |
| --- | ---: | ---: |
| Variant Black (H1) | 0.30 | **0.70** |
| Standard Black (control) | 0.46 | **0.54** |

Two-proportion \(z \approx 2.33\). H2 holds, but the extra decisive games are losses, not wins: \(\neg\)H1 \(\wedge\) H2.

### 6.3 Auxiliary — the other colour (not H1)

Directory: `tournament/berlin_var_white_vs_std_black_g3440`  
Same standard Black as the H4 opponent.

| Side (as White) vs standard Black | W | D | L | \(s\) | Scoreline |
| --- | ---: | ---: | ---: | ---: | --- |
| lc0-standard | 25 | 46 | 29 | **0.48** | 48 – 52 |
| lc0-variant | 20 | 31 | 49 | **0.355** | 35.5 – 64.5 |
| lc0-aggressive (`WhiteQ=winonly`) | 11 | 30 | 59 | **0.26** | 26 – 74 |

H4 directory: `tournament/berlin_aggressive_white_vs_std_black_g3440`  
Headers: `WhiteQ=winonly`, `BlackQ=wl`, both FPU `negation`.

H4 vs control White: \(\Delta s = 0.26-0.48 = -0.22\), 95% CI **[−0.319, −0.121]**, \(z \approx -4.36\). \(Q=W\) as White is weaker than packaged DrawScore as White, and weaker than \(W-L\).

### 6.4 Ablation (search only, no retraining)

White is always lc0-standard with classic Q and classic FPU. Black search is swapped on the **same** 3440 weights. Script: `scripts/run_h1_ablation.py`.

| Cell | Black (search) | Points /100 | W | D | L | Directory |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| Control | standard + \(W-L\) | 52 | 29 | 46 | 25 | `berlin_std_white_vs_std_black_g3440` |
| **A** | standard + DrawScore | **53** | 39 | 28 | 33 | `ablation_stdW_stdB_drawscore_negfpu_g3440` |
| **B** | dual + \(W-L\), FPU classic | **44** | 29 | 30 | 41 | `ablation_stdW_varB_wl_negfpu_g3440` |
| **C** | dual + DrawScore, FPU classic | **45** | 29 | 32 | 39 | `ablation_stdW_varB_drawscore_negfpu_g3440` |
| H1 | dual + DrawScore + **parent FPU** | **29** | 14 | 30 | 56 | `berlin_std_white_vs_var_black_g3440` |

Reading:

- A vs control: \(\Delta s=+0.01\), CI crosses 0. DrawScore on the standard net does **not** explain 29/100.
- B and C vs control: CIs cross 0.
- H1 vs C: \(\Delta s=-0.16\), CI **[−0.268, −0.052]**, \(z\approx-2.91\). With dual + DrawScore held fixed, switching FPU from classic negation to parent-relative is the contrast that excludes zero.

Ablation does **not** reopen H1. Learned-in-\(\theta\) damage is still unseen because no net was retrained for the 2×2.

---

## 7. PGN style (diploma four matches)

From `tournament/pgn_analysis.json` (regenerated by `scripts/analyze_tournament_pgns.py`). Exact ECO **C67** FEN was reached **0 / 400** times. These 6×64 / 120-visit nets play chess that *starts* from Berlin; they do not play Berlin theory.

### 7.1 Termination

| Termination | stdW–varB (H1) | stdW–stdB | varW–stdB | aggW–stdB (H4) |
| --- | ---: | ---: | ---: | ---: |
| Checkmate | 70 | 54 | 69 | 70 |
| Fifty moves | 22 | 8 | 3 | 21 |
| Threefold | 2 | 22 | 17 | 2 |
| Insufficient material | 5 | 15 | 10 | 4 |
| Stalemate | 1 | 1 | 1 | 0 |
| Adjudication (max ply) | 0 | 0 | 0 | **3** |

Mean ply: H1 208.8, control 156.8, variant White 173.7, H4 **222.1** (max 518 = 6 Berlin plies + `max_game_ply` 512).

H1 replaces the control’s repetitions with mates and fifty-move games — sharper, and worse for Black.

### 7.2 White’s 4th move (first move after 3…Nf6)

| Move | stdW–stdB | stdW–varB | varW–stdB | aggW–stdB |
| --- | ---: | ---: | ---: | ---: |
| Nd4 | 30 | 38 | 0 | 10 |
| h4 | 21 | 17 | 4 | 7 |
| c4 | 18 | 14 | 0 | 3 |
| O-O | 2 | 2 | 3 | **0** |
| Nh4 | 0 | 0 | 0 | **19** |
| g3 | 1 | 0 | 0 | **12** |
| Ke2 | 0 | 0 | 3 | **11** |
| c3 | 1 | 3 | **18** | 0 |
| d4 | 0 | 0 | **17** | 0 |
| Nc3 | 0 | 0 | **15** | 3 |

Standard White: Nd4 / h4 / c4, almost never 4.O-O. Aggressive White: Nh4, g3, Ke2. Qxd8 occurred once in H1, never in the other three diploma matches.

---

## 8. Also on disk — not the diploma H1 numbers

Use these only with an explicit label. **Do not mix them into H1.**

| Directory | What it is | Scoreline | Cite as H1? |
| --- | --- | --- | --- |
| `berlin_std_white_vs_winonly_black_g3440` | **H3**: standard White vs winonly Black (`BlackQ=asymmetric`) | 72 – 28 (Black \(s=0.28\): 15W 26D 59L) | **No** |
| `berlin_std_white_vs_var_black_g3440_rep2` | Same H1 pairing, `seed_offset=50000000` | 66 – 34 (Black 21W 26D 53L, \(s=0.34\)) | **No** (direction still harm) |
| `berlin_std_white_vs_std_black_g3440_rep2` | Control rematch, same seed offset | 52 – 48 (Black 31W 34D 35L, \(s=0.48\)) | **No** |
| `smoke_g3440` | 1-game smoke | — | No |
| `ablation_smoke_std_ds` | 1-game smoke | — | No |

H3 Black \(s=0.28\) is in the same neighbourhood as H1 variant Black \(s=0.29\). That is consistent with “win-seeking Black without the dual-net package also fails here,” but it was **not** entered into the H1 decision table.

**Deleted and never to be cited:** greedy (\(\tau=0\)) rematch; the first standard-White vs aggressive-Black pairing. Do not quote **32.5**.

---

## 9. Word thesis

Canonical copies:

| Document | Path |
| --- | --- |
| Albanian thesis | `Diploma versions/Albanian Version/Punim_Diplome_Veteluajtja_Asimetrike_Berlin.docx` |
| Albanian faculty filename | `Diploma versions/Albanian Version/Tema_Diplomes_Alban final.docx` |
| English thesis | `Diploma versions/English Version/Thesis_Asymmetric_SelfPlay_Berlin_Defence.docx` |
| Proposal | `Diploma versions/Thesis Proposal/Dissertation_Proposal_Asymmetric_SelfPlay_Berlin_Krasniqi.docx` |

The two Albanian `.docx` files at the repo root are the same Punim as the Albanian Version folder. Figures are embedded in those Word files (captions in the document; there is no `thesis_figures/` tree in this repository).

---

## 10. How to reproduce

Python 3, PyTorch with CUDA, `python-chess`. Engine `requirements.txt` files list the Python deps.

### Tests

```bat
python -m pytest lc0-standard\tests -q
python -m pytest lc0-variant\tests -q
python -m pytest lc0-winonly\tests -q
python -m pytest lc0-aggressive\tests -q
```

Counts used in the thesis annex: **13 / 34 / 21** (standard / variant / aggressive). `lc0-winonly` has the same 21 tests as aggressive (H3 clone).

### Train (do not feed later checkpoints into H1)

```bat
cd lc0-standard
python -u scripts\run_to_target.py --target-games 10000 --visits 120 --games 8 --out-dir checkpoints\run10k
```

Same pattern in `lc0-variant`, `lc0-winonly`, `lc0-aggressive`. The diploma freeze remains **3440** even if the loop continues toward 10 000.

### Diploma matches

From the repo root. Omit `--white-q` / `--black-q` to use each engine’s native tournament Q (see `NATIVE_Q` in `berlin_tournament.py`).

```bat
python -u scripts\berlin_tournament.py --games 100 --visits 120 --checkpoint-games 3440 --white-engine standard --black-engine variant --out-dir tournament\berlin_std_white_vs_var_black_g3440

python -u scripts\berlin_tournament.py --games 100 --visits 120 --checkpoint-games 3440 --white-engine standard --black-engine standard --out-dir tournament\berlin_std_white_vs_std_black_g3440

python -u scripts\berlin_tournament.py --games 100 --visits 120 --checkpoint-games 3440 --white-engine variant --black-engine standard --out-dir tournament\berlin_var_white_vs_std_black_g3440

python -u scripts\berlin_tournament.py --games 100 --visits 120 --checkpoint-games 3440 --white-engine aggressive --black-engine standard --white-q winonly --black-q wl --white-fpu negation --black-fpu negation --out-dir tournament\berlin_aggressive_white_vs_std_black_g3440
```

Ablation:

```bat
python -u scripts\run_h1_ablation.py
```

H3 (not H1):

```bat
python -u scripts\run_h3_match.py
```

PGN summary:

```bat
python scripts\analyze_tournament_pgns.py
```

Long GPU jobs are better launched from **CMD**, not from an IDE terminal that may disconnect.

---

## 11. Decision summary

1. **H1 is rejected as harm.** Packaged DrawScore + dual nets + parent FPU, trained from C65 at this scale, scored 29/100 as Black against 52/100 for standard Black.
2. **H2 holds without rescuing H1.** More decisive games, more losses.
3. **Search ablation** puts the identifiable slice of that harm on parent-relative FPU given dual + DrawScore; DrawScore alone does not produce 29/100.
4. **As White** against the same standard Black: standard 48, variant 35.5, aggressive (`Q=W`) 26. Classical \(W-L\) is strongest in this protocol.
5. **H3** (winonly Black 28/100) is on disk and looks like another failed win-seeking Black; it is not the H1 test.
6. **External validity** is one opening, one shallow net, 120 visits, 3440 games, \(N=100\). Do not generalise to full Lc0 or to Stockfish.

---

## 12. Standing constraints (do not silently undo)

- Do not reopen H1 after seeing the scores (no changing \(N\), FEN, or visits).
- Do not mix later or unequal checkpoints into H1. Freeze stays **3440**.
- Do not put winonly / aggressive weights into the H1 pairing.
- Do not replace Nibbler / official `lc0.exe`; these are separate Python engines.
- Do not cite deleted greedy or first-pairing results (no 32.5).
- Do not retitle the thesis as if H1 succeeded.
- Notify counters in training logs are not game counts; `progress.json` `games_played` is.
- `_rep2` directories are extra seed offsets, not the frozen table.

---

## 13. Primary artefacts

| Artefact | Path |
| --- | --- |
| Thesis (Albanian) | `Diploma versions/Albanian Version/Punim_Diplome_Veteluajtja_Asimetrike_Berlin.docx` |
| Faculty filename | `Diploma versions/Albanian Version/Tema_Diplomes_Alban final.docx` |
| Thesis (English) | `Diploma versions/English Version/Thesis_Asymmetric_SelfPlay_Berlin_Defence.docx` |
| Proposal | `Diploma versions/Thesis Proposal/Dissertation_Proposal_Asymmetric_SelfPlay_Berlin_Krasniqi.docx` |
| H1 PGN | `tournament/berlin_std_white_vs_var_black_g3440/games/` |
| Control PGN | `tournament/berlin_std_white_vs_std_black_g3440/games/` |
| Auxiliary PGN | `tournament/berlin_var_white_vs_std_black_g3440/games/` |
| H4 PGN | `tournament/berlin_aggressive_white_vs_std_black_g3440/games/` |
| H3 PGN (not H1) | `tournament/berlin_std_white_vs_winonly_black_g3440/games/` |
| Match JSON | `*/results.json` and `*/berlin_proof.json` |
| PGN aggregate | `tournament/pgn_analysis.json` |
| Frozen nets | `*/checkpoints/run10k/*-games003440.pt` |
