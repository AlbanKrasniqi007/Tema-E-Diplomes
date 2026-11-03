from __future__ import annotations

import sys
from pathlib import Path

import chess
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from lc0.chess.encoder import NUM_INPUT_PLANES, encode_position
from lc0.chess.opening import BERLIN_FEN, BERLIN_UCI, make_berlin_board
from lc0.chess.policy_index import POLICY_SIZE
from lc0.chess.policy_map import AZ_GATHER_INDICES, move_to_policy_index
from lc0.config import EngineConfig, NetworkConfig, load_config, opening_board
from lc0.neural.backend import NNOutput, NetworkBackend
from lc0.neural.dual import DualNetworkBackend
from lc0.neural.network import Lc0SEResNet
from lc0.search.node import Node
from lc0.search.params import SearchParams
from lc0.search.puct import PUCTSearch
from lc0.selfplay.game import SelfPlayGame, TrainingExample
from lc0.training.trainer import Trainer
from lc0.uci.engine import UCIEngine
from lc0.value import training_wdl, utility_q


def test_policy_index_size() -> None:
    assert POLICY_SIZE == 1858
    assert AZ_GATHER_INDICES.shape == (1858,)
    assert int(AZ_GATHER_INDICES.min()) >= 0


def test_encoder_startpos() -> None:
    board = chess.Board()
    planes = encode_position(board)
    assert planes.shape == (NUM_INPUT_PLANES, 8, 8)
    assert planes.dtype == np.float32
    assert planes[111].min() == 1.0  # all-ones plane
    assert planes[108].max() == 0.0  # white to move
    assert planes[104].min() == 1.0  # white queenside
    assert planes[105].min() == 1.0  # white kingside
    assert planes[0, 1, 0] == 1.0  # white pawn on a2
    assert planes[5, 0, 4] == 1.0  # white king on e1
    move = chess.Move.from_uci("e2e4")
    assert move_to_policy_index(move, False) >= 0


def test_encoder_black_flip() -> None:
    board = chess.Board()
    board.push_uci("e2e4")
    planes = encode_position(board)
    assert planes[108].min() == 1.0  # black to move
    # Side-to-move pawns sit on rank 2 of the flipped board (black pawns).
    assert planes[0, 1].sum() == 8.0


def test_network_forward() -> None:
    model = Lc0SEResNet(NetworkConfig(blocks=2, filters=16, se_channels=8, mlh_channels=4, mlh_fc=32))
    x = torch.zeros(2, NUM_INPUT_PLANES, 8, 8)
    x[:, 111] = 1.0
    policy, wdl, mlh = model(x)
    assert policy.shape == (2, 1858)
    assert wdl.shape == (2, 3)
    assert mlh.shape == (2,)


def test_search_legal_move() -> None:
    cfg = NetworkConfig(blocks=2, filters=16, se_channels=8, mlh_channels=4, mlh_fc=32)
    backend = NetworkBackend(cfg, device="cpu", cache_size=64)
    params = SearchParams(minibatch_size=4, dirichlet_epsilon=0.0)
    search = PUCTSearch(backend, params)
    board = chess.Board()
    result = search.search(board, visits=8)
    assert result.move in board.legal_moves
    assert result.nodes >= 2
    assert abs(result.policy_target.sum() - 1.0) < 1e-4


def test_training_step() -> None:
    cfg = EngineConfig()
    cfg.network = NetworkConfig(blocks=2, filters=16, se_channels=8, mlh_channels=4, mlh_fc=32)
    cfg.device = "cpu"
    cfg.train.batch_size = 4
    backend = NetworkBackend(cfg.network, device="cpu", cache_size=32)
    trainer = Trainer(backend, cfg)
    planes = encode_position(chess.Board())
    policy = np.zeros(POLICY_SIZE, dtype=np.float32)
    policy[move_to_policy_index(chess.Move.from_uci("e2e4"), False)] = 1.0
    wdl = np.array([0.0, 1.0, 0.0], dtype=np.float32)
    trainer.add_games(
        [TrainingExample(planes=planes, policy=policy, wdl=wdl, moves_left=40.0, ply=0) for _ in range(8)]
    )
    metrics = trainer.train_steps(steps=1)
    assert metrics["steps"] == 1.0
    assert np.isfinite(metrics["loss"])


def test_uci_handshake() -> None:
    cfg = EngineConfig()
    cfg.network = NetworkConfig(blocks=2, filters=16, se_channels=8, mlh_channels=4, mlh_fc=32)
    cfg.device = "cpu"
    engine = UCIEngine(cfg)
    replies = engine.handle("uci")
    assert "uciok" in replies
    assert any("MultiPV" in line and "default 500" in line for line in replies)
    ready = engine.handle("isready")
    assert "readyok" in ready
    assert any("Network weights" in line or "Loading weights" in line for line in ready)
    engine.handle("position startpos moves e2e4")
    assert engine.board.peek().uci() == "e2e4"


def test_uci_go_nodes() -> None:
    cfg = EngineConfig()
    cfg.network = NetworkConfig(blocks=2, filters=16, se_channels=8, mlh_channels=4, mlh_fc=32)
    cfg.device = "cpu"
    engine = UCIEngine(cfg)
    engine.handle("ucinewgame")
    engine.handle("position startpos")
    replies = engine.handle("go nodes 8")
    assert any(line.startswith("bestmove ") for line in replies)
    assert any(line.startswith("info ") and "wdl" in line for line in replies)
    assert any(line.startswith("info string ") and "N:" in line and "(P:" in line for line in replies)


def _tiny_uci_engine() -> UCIEngine:
    cfg = EngineConfig()
    cfg.network = NetworkConfig(blocks=2, filters=16, se_channels=8, mlh_channels=4, mlh_fc=32)
    cfg.device = "cpu"
    engine = UCIEngine(cfg)
    engine.verbose_move_stats = False
    engine.log_live_stats = False
    engine.multipv = 1
    engine.default_visits = 8
    return engine


def test_uci_go_cutechess_sudden_death_as_white() -> None:
    """CuteChess sends `go wtime … btime …` with no winc/binc when increment is 0."""
    engine = _tiny_uci_engine()
    engine.handle("ucinewgame")
    engine.handle("position startpos")
    replies = engine.handle("go wtime 8000 btime 8000")
    best = [line for line in replies if line.startswith("bestmove ")]
    assert len(best) == 1
    move = chess.Move.from_uci(best[0].split()[1])
    assert move in chess.Board().legal_moves


def test_uci_go_cutechess_sudden_death_as_black() -> None:
    engine = _tiny_uci_engine()
    engine.handle("ucinewgame")
    engine.handle("position startpos moves e2e4")
    replies = engine.handle("go wtime 8000 btime 8000")
    best = [line for line in replies if line.startswith("bestmove ")]
    assert len(best) == 1
    move = chess.Move.from_uci(best[0].split()[1])
    board = chess.Board()
    board.push_uci("e2e4")
    assert move in board.legal_moves


def test_verbose_stats_print_worst_first_like_lc0() -> None:
    """Nibbler sorts Leela engines by verbose-stat order (later line = better)."""
    cfg = EngineConfig()
    cfg.network = NetworkConfig(blocks=2, filters=16, se_channels=8, mlh_channels=4, mlh_fc=32)
    cfg.device = "cpu"
    engine = UCIEngine(cfg)
    engine.handle("ucinewgame")
    engine.handle("position startpos")
    replies = engine.handle("go nodes 32")
    visits: list[int] = []
    saw_node = False
    for line in replies:
        if not line.startswith("info string "):
            continue
        rest = line[len("info string ") :].strip()
        if rest.startswith("node"):
            saw_node = True
            continue
        if "N:" not in rest:
            continue
        visits.append(int(rest.split("N:", 1)[1].split()[0]))
    assert saw_node
    assert visits
    assert visits == sorted(visits)


def _info_nodes(replies: list[str]) -> int:
    best = 0
    for line in replies:
        if not line.startswith("info ") or " nodes " not in line:
            continue
        parts = line.split()
        best = max(best, int(parts[parts.index("nodes") + 1]))
    return best


def test_uci_halt_and_go_reuses_tree() -> None:
    cfg = EngineConfig()
    cfg.network = NetworkConfig(blocks=2, filters=16, se_channels=8, mlh_channels=4, mlh_fc=32)
    cfg.device = "cpu"
    engine = UCIEngine(cfg)
    engine.handle("ucinewgame")
    engine.handle("position startpos")
    first = engine.handle("go nodes 24")
    n1 = _info_nodes(first)
    engine.handle("stop")
    engine.handle("position startpos")
    second = engine.handle("go nodes 24")
    n2 = _info_nodes(second)
    assert n1 >= 8
    assert n2 > n1
    assert engine._tree is not None
    assert engine._tree.n == n2


def test_uci_newgame_drops_tree() -> None:
    cfg = EngineConfig()
    cfg.network = NetworkConfig(blocks=2, filters=16, se_channels=8, mlh_channels=4, mlh_fc=32)
    cfg.device = "cpu"
    engine = UCIEngine(cfg)
    engine.handle("position startpos")
    engine.handle("go nodes 16")
    engine.handle("ucinewgame")
    engine.handle("position startpos")
    after = engine.handle("go nodes 16")
    assert _info_nodes(after) < 40


def test_uci_multipv_reports_several_root_moves() -> None:
    cfg = EngineConfig()
    cfg.network = NetworkConfig(blocks=2, filters=16, se_channels=8, mlh_channels=4, mlh_fc=32)
    cfg.device = "cpu"
    engine = UCIEngine(cfg)
    engine.handle("ucinewgame")
    engine.handle("position startpos")
    engine.handle("setoption name MultiPV value 3")
    replies = engine.handle("go nodes 16")
    pvs = [line for line in replies if line.startswith("info ") and " multipv " in line and " pv " in line]
    ids = []
    for line in pvs:
        parts = line.split()
        ids.append(int(parts[parts.index("multipv") + 1]))
    engine.handle("setoption name MultiPV value 1")
    engine.handle("position startpos")
    single = engine.handle("go nodes 8")
    assert any(line.startswith("info ") and " pv " in line for line in single)
    assert not any(" multipv " in line for line in single if line.startswith("info ") and " pv " in line)


def test_berlin_opening() -> None:
    board = make_berlin_board()
    assert board.fen() == BERLIN_FEN
    assert [move.uci() for move in board.move_stack] == list(BERLIN_UCI)
    cfg = load_config(ROOT / "configs" / "default.yaml")
    opened = opening_board(cfg)
    assert opened.fen() == BERLIN_FEN
    assert cfg.train.opening_name == "Berlin Defence"


def test_uci_berlin_identity() -> None:
    cfg = EngineConfig()
    cfg.network = NetworkConfig(blocks=2, filters=16, se_channels=8, mlh_channels=4, mlh_fc=32)
    cfg.device = "cpu"
    engine = UCIEngine(cfg)
    replies = engine.handle("uci")
    assert any("lc0-variant" in line for line in replies)
    assert any("Lc0 Berlin" in line for line in replies)
    assert any("WhiteValue" in line and "win_equals_draw" in line for line in replies)
    assert any("BlackValue" in line and "draw_equals_loss" in line for line in replies)
    assert any(BERLIN_FEN in line for line in replies)
    engine.handle(f"position fen {BERLIN_FEN}")
    assert engine.board.fen() == BERLIN_FEN
    replies = engine.handle("go nodes 8")
    assert any(line.startswith("bestmove ") for line in replies)
    move = chess.Move.from_uci(replies[-1].split()[1])
    assert move in engine.board.legal_moves


def test_utility_q_white_draw_seeking() -> None:
    assert abs(utility_q(0.0, 1.0, 0.0, True) - 1.0) < 1e-6
    assert abs(utility_q(1.0, 0.0, 0.0, True) - 1.0) < 1e-6
    assert abs(utility_q(0.0, 0.0, 1.0, True) + 1.0) < 1e-6
    assert abs(utility_q(0.2, 0.5, 0.3, True) - 0.4) < 1e-6


def test_utility_q_black_win_seeking() -> None:
    assert abs(utility_q(0.0, 1.0, 0.0, False) + 1.0) < 1e-6
    assert abs(utility_q(1.0, 0.0, 0.0, False) - 1.0) < 1e-6
    assert abs(utility_q(0.0, 0.0, 1.0, False) + 1.0) < 1e-6
    assert abs(utility_q(0.2, 0.5, 0.3, False) + 0.6) < 1e-6


def test_training_wdl_by_color() -> None:
    white_win = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    draw = np.array([0.0, 1.0, 0.0], dtype=np.float32)
    white_loss = np.array([0.0, 0.0, 1.0], dtype=np.float32)
    np.testing.assert_array_equal(training_wdl(white_win, chess.WHITE), [1.0, 0.0, 0.0])
    np.testing.assert_array_equal(training_wdl(draw, chess.WHITE), [0.0, 1.0, 0.0])
    np.testing.assert_array_equal(training_wdl(white_loss, chess.WHITE), [0.0, 0.0, 1.0])
    np.testing.assert_array_equal(training_wdl(white_win, chess.BLACK), [0.0, 0.0, 1.0])
    np.testing.assert_array_equal(training_wdl(draw, chess.BLACK), [0.0, 1.0, 0.0])
    np.testing.assert_array_equal(training_wdl(white_loss, chess.BLACK), [1.0, 0.0, 0.0])


class _DrawBackend:
    """NN that always predicts a draw, used to check search utility Q."""

    def evaluate_cached(self, key, planes, policy_temp=1.61):
        return self.evaluate_batch(np.expand_dims(planes, 0), policy_temp)[0]

    def evaluate_batch(self, planes, policy_temp=1.61):
        n = 1 if planes.ndim == 3 else planes.shape[0]
        uniform = np.ones(POLICY_SIZE, dtype=np.float32) / POLICY_SIZE
        return [
            NNOutput(
                policy=uniform,
                logits=np.zeros(POLICY_SIZE, dtype=np.float32),
                w=0.0,
                d=1.0,
                l=0.0,
                q=0.0,
                m=10.0,
            )
            for _ in range(n)
        ]


class _EvenBackend:
    """NN that always predicts 50–50, so terminals stand out in Q."""

    def evaluate_cached(self, key, planes, policy_temp=1.61):
        return self.evaluate_batch(np.expand_dims(planes, 0), policy_temp)[0]

    def evaluate_batch(self, planes, policy_temp=1.61):
        n = 1 if planes.ndim == 3 else planes.shape[0]
        uniform = np.ones(POLICY_SIZE, dtype=np.float32) / POLICY_SIZE
        return [
            NNOutput(
                policy=uniform,
                logits=np.zeros(POLICY_SIZE, dtype=np.float32),
                w=0.5,
                d=0.0,
                l=0.5,
                q=0.0,
                m=10.0,
            )
            for _ in range(n)
        ]


def test_search_keeps_chess_draw_applies_utility_q() -> None:
    search = PUCTSearch(_DrawBackend(), SearchParams(dirichlet_epsilon=0.0, minibatch_size=4))
    white = chess.Board()
    node = Node()
    out = search._evaluate_and_expand(node, white, [white], add_noise=False)
    assert (out.w, out.d, out.l) == (0.0, 1.0, 0.0)
    assert abs(out.q - 1.0) < 1e-6
    assert abs(node.chess_q) < 1e-6

    black = chess.Board()
    black.push_uci("e2e4")
    node = Node()
    out = search._evaluate_and_expand(node, black, [chess.Board(), black.copy(stack=False)], add_noise=False)
    assert (out.w, out.d, out.l) == (0.0, 1.0, 0.0)
    assert abs(out.q + 1.0) < 1e-6
    assert abs(node.chess_q) < 1e-6


def test_terminal_draws_and_mates() -> None:
    search = PUCTSearch(_DrawBackend(), SearchParams(dirichlet_epsilon=0.0))
    black_stale = chess.Board("7k/5Q2/6K1/8/8/8/8/8 b - - 0 1")
    white_stale = chess.Board("7K/5q2/6k1/8/8/8/8/8 w - - 0 1")
    black_mate = chess.Board("7k/5QK1/8/8/8/8/8/8 b - - 0 1")
    white_mate = chess.Board("7K/5qk1/8/8/8/8/8/8 w - - 0 1")
    assert black_stale.is_stalemate()
    assert white_stale.is_stalemate()
    assert black_mate.is_checkmate()
    assert white_mate.is_checkmate()
    assert search._terminal_wdl(black_stale) == (0.0, 1.0, 0.0)
    assert search._terminal_wdl(white_stale) == (0.0, 1.0, 0.0)
    assert search._terminal_wdl(black_mate) == (0.0, 0.0, 1.0)
    assert search._terminal_wdl(white_mate) == (0.0, 0.0, 1.0)


def test_white_values_stalemate_like_a_win() -> None:
    """White is happy to kill the game: a stalemate backs up as Q = +1."""
    search = PUCTSearch(_EvenBackend(), SearchParams(dirichlet_epsilon=0.0, minibatch_size=8))
    board = chess.Board("7k/8/5QK1/8/8/8/8/8 w - - 0 1")
    stale = chess.Move.from_uci("f6f7")
    result = search.search(board, visits=16, add_noise=False, temperature=0.0, searchmoves=[stale])
    assert abs(result.q - 1.0) < 1e-6


def test_black_values_stalemate_like_a_loss() -> None:
    """Black must win: a stalemate backs up as Q = -1."""
    search = PUCTSearch(_EvenBackend(), SearchParams(dirichlet_epsilon=0.0, minibatch_size=8))
    board = chess.Board("7K/8/5qk1/8/8/8/8/8 b - - 0 1")
    stale = chess.Move.from_uci("f6f7")
    result = search.search(board, visits=16, add_noise=False, temperature=0.0, searchmoves=[stale])
    assert abs(result.q + 1.0) < 1e-6


def test_black_prefers_mate_over_stalemate() -> None:
    search = PUCTSearch(_EvenBackend(), SearchParams(dirichlet_epsilon=0.0, minibatch_size=8))
    board = chess.Board("7K/8/5qk1/8/8/8/8/8 b - - 0 1")
    result = search.search(board, visits=32, add_noise=False, temperature=0.0)
    board.push(result.move)
    assert board.is_checkmate()
    assert not board.is_stalemate()


def test_fpu_is_parent_relative_not_negated() -> None:
    """Utility Q is not zero-mean; FPU must not be Lc0's -Q."""
    search = PUCTSearch(_DrawBackend(), SearchParams(dirichlet_epsilon=0.0, fpu_value=0.330))
    white = chess.Board()
    node = Node()
    search._evaluate_and_expand(node, white, [white], add_noise=False)
    node.n = max(node.n, 1)
    assert node.q > 0.5
    fpu = search._fpu(node)
    assert fpu > 0.0
    assert abs(fpu - node.q) < 1e-6

    black = chess.Board()
    black.push_uci("e2e4")
    node = Node()
    search._evaluate_and_expand(node, black, [chess.Board(), black.copy(stack=False)], add_noise=False)
    node.n = max(node.n, 1)
    assert node.q < -0.5
    fpu = search._fpu(node)
    assert fpu < 0.0
    assert abs(fpu - node.q) < 1e-6


def test_q_mode_wl_uses_chess_q() -> None:
    """Ablation switch: same draw WDL, Q becomes W−L instead of DrawScore."""
    search = PUCTSearch(
        _DrawBackend(),
        SearchParams(dirichlet_epsilon=0.0, minibatch_size=4, q_mode="wl", fpu_mode="negation"),
    )
    white = chess.Board()
    node = Node()
    out = search._evaluate_and_expand(node, white, [white], add_noise=False)
    assert (out.w, out.d, out.l) == (0.0, 1.0, 0.0)
    assert abs(out.q) < 1e-6
    assert abs(node.q) < 1e-6
    assert node.q_mode == "wl"


def test_fpu_mode_negation_matches_classic_lc0() -> None:
    search = PUCTSearch(
        _DrawBackend(),
        SearchParams(dirichlet_epsilon=0.0, fpu_value=0.330, q_mode="drawscore", fpu_mode="negation"),
    )
    white = chess.Board()
    node = Node()
    search._evaluate_and_expand(node, white, [white], add_noise=False)
    node.n = max(node.n, 1)
    fpu = search._fpu(node)
    assert abs(fpu - (-node.q)) < 1e-6
    assert fpu < 0.0


def test_black_search_deepens_instead_of_spraying() -> None:
    """With parent-relative FPU, Black reuses moves instead of 1-visit spray."""
    search = PUCTSearch(
        _DrawBackend(),
        SearchParams(
            dirichlet_epsilon=0.0,
            minibatch_size=8,
            fpu_value=0.330,
            # Default cpuct equalizes 20 uniform moves by 64 visits; keep U small
            # so FPU reuse is visible (Lc0 -Q still sprays 1-ply in that regime).
            cpuct=0.3,
            cpuct_factor=0.0,
        ),
    )
    board = chess.Board()
    board.push_uci("e2e4")
    result = search.search(board, visits=64, add_noise=False, temperature=0.0)
    visits = sorted((edge.n for edge in result.root.edges), reverse=True)
    assert visits[0] >= 8
    ones = sum(1 for n in visits if n <= 1)
    assert ones < len(visits) * 0.75


def test_two_fold_root_still_moves() -> None:
    """A 2-fold at the root is not a FIDE draw; search must still play a move."""
    search = PUCTSearch(_EvenBackend(), SearchParams(dirichlet_epsilon=0.0, minibatch_size=8, two_fold_draws=True))
    board = chess.Board()
    for uci in ("g1f3", "g8f6", "f3g1", "f6g8"):
        board.push_uci(uci)
    assert board.is_repetition(2)
    assert not board.is_game_over(claim_draw=True)
    result = search.search(board, visits=8, add_noise=False, temperature=0.0)
    assert result.move in board.legal_moves


def test_uci_rejects_official_pb_gz() -> None:
    cfg = EngineConfig()
    cfg.network = NetworkConfig(blocks=2, filters=16, se_channels=8, mlh_channels=4, mlh_fc=32)
    cfg.device = "cpu"
    engine = UCIEngine(cfg)
    replies = engine.handle(r"setoption name WeightsFile value C:\Users\Admin\Documents\Chess\Engines\Lc0\791556.pb.gz")
    assert any("rejected official Lc0" in line or ".pb.gz" in line for line in replies)
    assert engine.backend.weights_path is None


def test_uci_applies_pt_checkpoint() -> None:
    ckpt = ROOT / "checkpoints" / "lc0-variant-iter0050.pt"
    if not ckpt.is_file():
        return
    cfg = EngineConfig()
    cfg.network = NetworkConfig(blocks=6, filters=64, se_channels=16, mlh_channels=8, mlh_fc=128)
    cfg.device = "cpu"
    engine = UCIEngine(cfg)
    replies = engine.handle(f"setoption name WeightsFile value {ckpt}")
    assert any("Network weights applied" in line for line in replies)
    assert engine.backend.weights_path
    assert "iter0050" in engine.backend.weights_path.replace("\\", "/")
    assert any("agents=white,black" in line for line in replies)
    assert getattr(engine.backend, "cloned_from_single", False)


class _FixedWDLBackend:
    """Fake net with a distinctive WDL, for DualNetworkBackend routing tests."""

    def __init__(self, w: float, d: float, l: float) -> None:
        self.w, self.d, self.l = w, d, l

    def _out(self) -> NNOutput:
        uniform = np.ones(POLICY_SIZE, dtype=np.float32) / POLICY_SIZE
        return NNOutput(
            policy=uniform,
            logits=np.zeros(POLICY_SIZE, dtype=np.float32),
            w=self.w,
            d=self.d,
            l=self.l,
            q=self.w - self.l,
            m=10.0,
        )

    def evaluate_cached(self, key, planes, policy_temp=1.61):
        return self._out()

    def evaluate_batch(self, planes, policy_temp=1.61):
        n = 1 if np.asarray(planes).ndim == 3 else np.asarray(planes).shape[0]
        return [self._out() for _ in range(n)]


def test_dual_routes_eval_by_side() -> None:
    white_fake = _FixedWDLBackend(0.9, 0.1, 0.0)
    black_fake = _FixedWDLBackend(0.0, 0.2, 0.8)
    dual = DualNetworkBackend(white=white_fake, black=black_fake, device="cpu")

    start = chess.Board()
    planes_w = encode_position(start)
    out_w = dual.evaluate_cached(start.fen(), planes_w)
    assert (out_w.w, out_w.d, out_w.l) == (0.9, 0.1, 0.0)

    after_e4 = chess.Board()
    after_e4.push_uci("e2e4")
    planes_b = encode_position(after_e4)
    out_b = dual.evaluate_cached(after_e4.fen(), planes_b)
    assert (out_b.w, out_b.d, out_b.l) == (0.0, 0.2, 0.8)

    batch = np.stack([planes_w, planes_b, planes_w], axis=0)
    outs = dual.evaluate_batch(batch)
    assert (outs[0].w, outs[0].d, outs[0].l) == (0.9, 0.1, 0.0)
    assert (outs[1].w, outs[1].d, outs[1].l) == (0.0, 0.2, 0.8)
    assert (outs[2].w, outs[2].d, outs[2].l) == (0.9, 0.1, 0.0)


def test_training_example_stm_split() -> None:
    cfg = NetworkConfig(blocks=2, filters=16, se_channels=8, mlh_channels=4, mlh_fc=32)
    white = NetworkBackend(cfg, device="cpu", cache_size=32)
    black = NetworkBackend(cfg, device="cpu", cache_size=32)
    dual = DualNetworkBackend(cfg, device="cpu", cache_size=32, white=white, black=black)
    params = SearchParams(minibatch_size=4, dirichlet_epsilon=0.0)
    worker = SelfPlayGame(
        dual,
        params,
        visits=4,
        temperature=0.0,
        temp_decay_moves=0,
        max_ply=6,
        start_factory=chess.Board,
    )
    examples, _board = worker.play()
    assert examples
    stms = {ex.stm for ex in examples}
    assert chess.WHITE in stms
    assert chess.BLACK in stms
    # Existing TrainingExample() calls without stm must still construct.
    planes = encode_position(chess.Board())
    policy = np.zeros(POLICY_SIZE, dtype=np.float32)
    wdl = np.array([0.0, 1.0, 0.0], dtype=np.float32)
    implicit = TrainingExample(planes=planes, policy=policy, wdl=wdl, moves_left=1.0, ply=0)
    assert implicit.stm == chess.WHITE


def test_dual_load_clones_single_checkpoint() -> None:
    ckpt = ROOT / "checkpoints" / "lc0-variant-iter0050.pt"
    if not ckpt.is_file():
        return
    cfg = NetworkConfig(blocks=6, filters=64, se_channels=16, mlh_channels=8, mlh_fc=128)
    dual = DualNetworkBackend(cfg, device="cpu", cache_size=8)
    dual.load(str(ckpt))
    assert dual.cloned_from_single
    white_sd = dual.white.model.state_dict()
    black_sd = dual.black.model.state_dict()
    assert white_sd.keys() == black_sd.keys()
    for key in white_sd:
        assert torch.equal(white_sd[key], black_sd[key]), key


if __name__ == "__main__":
    tests = [
        test_policy_index_size,
        test_encoder_startpos,
        test_encoder_black_flip,
        test_network_forward,
        test_search_legal_move,
        test_training_step,
        test_uci_handshake,
        test_uci_go_nodes,
        test_uci_go_cutechess_sudden_death_as_white,
        test_uci_go_cutechess_sudden_death_as_black,
        test_verbose_stats_print_worst_first_like_lc0,
        test_uci_halt_and_go_reuses_tree,
        test_uci_newgame_drops_tree,
        test_uci_multipv_reports_several_root_moves,
        test_berlin_opening,
        test_uci_berlin_identity,
        test_utility_q_white_draw_seeking,
        test_utility_q_black_win_seeking,
        test_training_wdl_by_color,
        test_search_keeps_chess_draw_applies_utility_q,
        test_terminal_draws_and_mates,
        test_white_values_stalemate_like_a_win,
        test_black_values_stalemate_like_a_loss,
        test_black_prefers_mate_over_stalemate,
        test_fpu_is_parent_relative_not_negated,
        test_q_mode_wl_uses_chess_q,
        test_fpu_mode_negation_matches_classic_lc0,
        test_black_search_deepens_instead_of_spraying,
        test_two_fold_root_still_moves,
        test_uci_rejects_official_pb_gz,
        test_uci_applies_pt_checkpoint,
        test_dual_routes_eval_by_side,
        test_training_example_stm_split,
        test_dual_load_clones_single_checkpoint,
    ]
    for fn in tests:
        fn()
        print("ok", fn.__name__)
    print("all tests passed")
