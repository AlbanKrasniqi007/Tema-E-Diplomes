# Lc0-standard — architecture notes

This engine is a **trainable replica** of Leela Chess Zero’s documented
`classic` search + `NETWORK_SE_WITH_HEADFORMAT` network, not a recompile of
the C++ `lc0` binary. Layout follows
[the lc0 high-level architecture](https://lczero.org/dev/lc0/architecture/):

| Folder | Lc0 counterpart | Role |
| --- | --- | --- |
| `src/lc0/chess/` | `src/chess/` | Board encoding and 1858-move policy map |
| `src/lc0/search/` | `src/search/classic/` | PUCT / MCTS |
| `src/lc0/neural/` | `src/neural/` | SE-ResNet backend |
| `src/lc0/selfplay/` | `src/selfplay/` | Self-play games |
| `src/lc0/training/` | lczero-training | Policy + WDL + moves-left loss |
| `src/lc0/uci/` | UCI entrypoint | GUI / Cute Chess interface |

## Neural network

Taken from [Neural network topology](https://lczero.org/dev/old/nn/)
(NETWORK_SE_WITH_HEADFORMAT):

- Input: **112 × 8 × 8** planes (`INPUT_CLASSICAL_112_PLANE`)
- Body: 3×3 input convolution, then `BLOCKS` residual units
- Each residual unit: conv → conv → **Squeeze-and-Excitation**
  (`GAP → FC → ReLU → FC → split W,B → sigmoid(W)·x + B`) → skip → ReLU
- Policy head: **POLICY_CONVOLUTION** (80×8×8 gathered into 1858 logits)
- Value head: **VALUE_WDL** (softmax over win / draw / loss)
- Extra **moves-left** head (ReLU scalar)

Default size on this machine is **6×64 / SE 16**, which fits an RTX 4060 8 GB
for self-play + training. `configs/paper.yaml` raises this to **10×128**.

## Search

Taken from the [AlphaZero primer](https://lczero.org/dev/lc0/search/alphazero/)
and [technical explanation](https://lczero.org/dev/wiki/technical-explanation-of-leela-chess-zero/):

\[
\text{PUCT} = Q + c_{\text{puct}}\, P\, \frac{\sqrt{N_{\text{parent}}}}{1+N}
\]

\[
c_{\text{puct}} = c_{\text{init}} + c_{\text{factor}} \log\frac{N + c_{\text{base}}}{c_{\text{base}}}
\]

Lc0 defaults used here: `c_init = 1.745`, `c_base = 38739`, `c_factor = 3.894`.

- **FPU (reduction):** \( -\bar Q_{\text{parent}} - 0.330\sqrt{\sum P_{\text{visited}}} \)
- **Unscored virtual visits:** `N_in_flight` is incremented while a minibatch
  is in flight; `Q` is not changed (not virtual loss)
- Root Dirichlet noise during training: \(\varepsilon=0.25\), \(\alpha=0.3\)
- \( Q = W - L \) from the WDL head

`lc3` (event-queue / sharded DAG search) is documented on lczero.org but is
still in development; this replica implements **classic** search, which is the
algorithm used to train published Lc0 nets and the one described in the
AlphaZero papers.

## Training

AlphaZero self-play loop:

1. Play a game with MCTS; store `(planes, visit-policy, ply)`
2. When the game ends, write WDL from the side-to-move view and moves-left
3. Minimise `policy CE + WDL CE + λ · MLH MSE` with AdamW

This is the same RL loop described in the Lc0
[overview](https://lczero.org/dev/overview/) (client produces games, trainer
updates the net), implemented locally instead of the distributed servers.
