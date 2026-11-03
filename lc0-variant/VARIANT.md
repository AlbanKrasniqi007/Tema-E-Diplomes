# Variant: two specialists co-evolving in one process

Status: **same search / encoder / 6×64 SE-ResNet topology as `lc0-standard`,
but two nets and asymmetric match utilities.**

`lc0-standard` stays the symmetric **one-net** control (both colors, standard
WDL). `lc0-variant` is **not** “one net with remapped Q”. It is a
**co-evolving pair** behind one UCI executable:

| Agent | Plays | Match goal | What it trains |
| --- | --- | --- | --- |
| **White specialist** | White only (Berlin) | win = draw in *search*; value head still learns real W/D/L | White positions only |
| **Black specialist** | Black only (Berlin) | draw = loss in *search*; value head still learns real W/D/L | Black positions only |

They **share games**. White searches White moves with the White net; Black
searches Black moves with the Black net. One `PUCTSearch` sits on a
`DualNetworkBackend` that routes every leaf by side-to-move (FEN `w`/`b`, or
encoder plane 108: 0 = White, 1 = Black). Inside the tree, White therefore
evaluates Black nodes with the Black net and vice versa — they play each
other, including during search.

When you play Black in Nibbler, the engine is the White specialist (draw-seeking).
When you play White, the engine is the Black specialist (win-seeking).

The diploma freeze is `checkpoints/run10k/lc0-variant-games003440.pt` (already
a dual checkpoint). `DualNetworkBackend.load` still clones a single-net `.pt`
into both specialists if you point it at one.

```bat
python -u scripts\run_to_target.py --target-games 10000 --visits 120 --games 8 --out-dir checkpoints\run10k
```

## Belief vs preference (Lc0 DrawScore)

Each specialist’s value head predicts **chess** win / draw / loss from the
side to move — the same target as AlphaZero (`z ∈ {win, draw, loss}`) and
the Lc0 WDL head. That is how the net “sees” a position from its own games.

Match goals are applied **only** when converting WDL → PUCT `Q`, with Lc0’s
DrawScore formula \(Q = (W-L) + d_s D\):

| Side | Outcome values | Q used in PUCT | Style |
| --- | --- | --- | --- |
| **White** | win = draw (good); only loss is bad | \(Q = 1-2L\) (\(d_s=+1\)) | Draw-seeking |
| **Black** | loss = draw (bad); only win is good | \(Q = 2W-1\) (\(d_s=-1\)) | Win-seeking |

A draw is \(Q = +1\) for White and \(Q = -1\) for Black, so the usual W/L
flip on backup stays zero-sum. UCI `wdl` / `score cp` report the **chess**
WDL (Nibbler expected score), not the utility. Policy learning still
encodes “repeat ways I won / avoid ways I lost”: visit counts come from a
search that used that color’s `Q`.

Do **not** collapse draws into the value-head labels. That made both nets
unable to tell a draw from a win (White) or a loss (Black), which showed
up in Nibbler as ~90% expected score on White’s move and ~4% on Black’s.

## FPU (parent-relative — keep this)

FPU is parent-relative: \(Q - 0.330\sqrt{P}\). Lc0’s reduction formula uses
\(-Q\) because standard (chess) Q is zero-mean; **utility** Q is not (White
\(Q \approx +1\), Black \(Q \approx -1\)). With \(-Q\), Black’s FPU was
\(\approx +1\), so unvisited moves always beat visited ones and search sprayed
1-ply instead of deepening. That made Black’s visit policy near-uniform.

Parent-relative FPU stops the spray. It does **not** give Black its own
policy/value: that is what the two specialists are for. A shared net was
still averaging White’s draw-seeking policy with Black’s win-seeking policy.

## Training loop

1. Play a game from the Berlin with one searcher on the dual backend.
2. Split examples by `TrainingExample.stm`.
3. `Trainer(dual.white)` updates only White; `Trainer(dual.black)` updates
   only Black. Skip a color if its replay is smaller than `batch_size`.
4. Save **one** combined checkpoint
   `{kind: dual, white: {model, optimizer, step}, black: {...}, config}`
   as `lc0-variant-iterNNNN.pt`.

Code: `src/lc0/neural/dual.py`, `src/lc0/value.py`, `src/lc0/search/puct.py`,
`src/lc0/selfplay/game.py`, `scripts/train.py`.
