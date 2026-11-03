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
from lc0.neural.network import Lc0SEResNet
from lc0.search.node import Node
from lc0.search.params import SearchParams
from lc0.search.puct import PUCTSearch
from lc0.selfplay.game import TrainingExample
from lc0.training.trainer import Trainer
from lc0.uci.engine import UCIEngine


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
    assert any("VerboseMoveStats" in line for line in replies)
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


def test_uci_go_cutechess_sudden_death_as_white() -> None:
    cfg = EngineConfig()
    cfg.network = NetworkConfig(blocks=2, filters=16, se_channels=8, mlh_channels=4, mlh_fc=32)
    cfg.device = "cpu"
    engine = UCIEngine(cfg)
    engine.verbose_move_stats = False
    engine.log_live_stats = False
    engine.default_visits = 8
    engine.handle("ucinewgame")
    engine.handle("position startpos")
    replies = engine.handle("go wtime 8000 btime 8000")
    best = [line for line in replies if line.startswith("bestmove ")]
    assert len(best) == 1
    move = chess.Move.from_uci(best[0].split()[1])
    assert move in chess.Board().legal_moves


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
    assert any("lc0-standard" in line for line in replies)
    assert any("Lc0 Berlin" in line for line in replies)
    assert any(BERLIN_FEN in line for line in replies)
    engine.handle(f"position fen {BERLIN_FEN}")
    assert engine.board.fen() == BERLIN_FEN
    replies = engine.handle("go nodes 8")
    assert any(line.startswith("bestmove ") for line in replies)
    move = chess.Move.from_uci(replies[-1].split()[1])
    assert move in engine.board.legal_moves


def test_two_fold_root_still_moves() -> None:
    cfg = NetworkConfig(blocks=2, filters=16, se_channels=8, mlh_channels=4, mlh_fc=32)
    backend = NetworkBackend(cfg, device="cpu", cache_size=64)
    params = SearchParams(minibatch_size=4, dirichlet_epsilon=0.0, two_fold_draws=True)
    search = PUCTSearch(backend, params)
    board = chess.Board()
    for uci in ("g1f3", "g8f6", "f3g1", "f6g8"):
        board.push_uci(uci)
    assert board.is_repetition(2)
    assert not board.is_game_over(claim_draw=True)
    result = search.search(board, visits=8)
    assert result.move in board.legal_moves


class _DrawBackend:
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


def test_drawscore_q_mode_on_standard_net() -> None:
    search = PUCTSearch(
        _DrawBackend(),
        SearchParams(dirichlet_epsilon=0.0, minibatch_size=4, q_mode="drawscore", fpu_mode="negation"),
    )
    white = chess.Board()
    node = Node()
    out = search._evaluate_and_expand(node, white, [white], add_noise=False)
    assert abs(out.q - 1.0) < 1e-6
    assert node.q_mode == "drawscore"


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
        test_berlin_opening,
        test_uci_berlin_identity,
        test_two_fold_root_still_moves,
        test_drawscore_q_mode_on_standard_net,
    ]
    for fn in tests:
        fn()
        print("ok", fn.__name__)
    print("all tests passed")
