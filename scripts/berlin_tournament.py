#!/usr/bin/env python3
"""Berlin matches with fixed colors, recorded as PGN from the Berlin Defence.

Default: lc0-standard (White) vs lc0-variant (Black).
Control: --black-engine standard plays lc0-standard vs itself.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import chess
import chess.pgn
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
BERLIN_UCI = ("e2e4", "e7e5", "g1f3", "b8c6", "f1b5", "g8f6")
BERLIN_FEN = "r1bqkb1r/pppp1ppp/2n2n2/1B2p3/4P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 4 4"
BERLIN_PGN = "1. e4 e5 2. Nf3 Nc6 3. Bb5 Nf6"
ENGINES = ("standard", "variant", "winonly", "aggressive")
Q_MODES = ("wl", "drawscore", "winonly", "asymmetric")
FPU_MODES = ("negation", "parent")
NATIVE_Q = {
    "standard": "wl",
    "variant": "drawscore",
    "winonly": "asymmetric",
    "aggressive": "winonly",
}
NATIVE_FPU = {
    "standard": "negation",
    "variant": "parent",
    "winonly": "negation",
    "aggressive": "negation",
}
ENGINE_TAG = {
    "standard": "std",
    "variant": "var",
    "winonly": "winonly",
    "aggressive": "aggressive",
}
PAIR_OFFSETS = {
    ("standard", "variant"): 0,
    ("standard", "standard"): 1_000_000,
    ("variant", "standard"): 2_000_000,
    ("variant", "variant"): 3_000_000,
    ("standard", "winonly"): 5_000_000,
    ("winonly", "standard"): 6_000_000,
    ("winonly", "winonly"): 7_000_000,
    ("variant", "winonly"): 8_000_000,
    ("winonly", "variant"): 9_000_000,
    ("standard", "aggressive"): 10_000_000,
    ("aggressive", "standard"): 11_000_000,
    ("aggressive", "aggressive"): 12_000_000,
    ("variant", "aggressive"): 13_000_000,
    ("aggressive", "variant"): 14_000_000,
    ("winonly", "aggressive"): 15_000_000,
    ("aggressive", "winonly"): 16_000_000,
}


def _drop_lc0() -> None:
    for key in [k for k in sys.modules if k == "lc0" or k.startswith("lc0.")]:
        del sys.modules[key]


def load_engine_pkg(src: Path) -> SimpleNamespace:
    src = str(src.resolve())
    _drop_lc0()
    while src in sys.path:
        sys.path.remove(src)
    sys.path.insert(0, src)
    from lc0.chess.opening import history_from_board, make_berlin_board
    from lc0.config import load_config
    from lc0.search.params import SearchParams
    from lc0.search.puct import PUCTSearch

    try:
        from lc0.neural.dual import DualNetworkBackend
    except ImportError:
        DualNetworkBackend = None
    from lc0.neural.backend import NetworkBackend

    kept = {k: sys.modules[k] for k in list(sys.modules) if k == "lc0" or k.startswith("lc0.")}
    ns = SimpleNamespace(
        load_config=load_config,
        SearchParams=SearchParams,
        PUCTSearch=PUCTSearch,
        NetworkBackend=NetworkBackend,
        DualNetworkBackend=DualNetworkBackend,
        history_from_board=history_from_board,
        make_berlin_board=make_berlin_board,
        _modules=kept,
    )
    sys.path.remove(src)
    _drop_lc0()
    return ns


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def berlin_start() -> chess.Board:
    board = chess.Board()
    for uci in BERLIN_UCI:
        board.push_uci(uci)
    if board.fen() != chess.Board(BERLIN_FEN).fen():
        raise RuntimeError(f"Berlin setup mismatch: {board.fen()}")
    return board


def first_six_uci(board: chess.Board) -> tuple[str, ...]:
    return tuple(move.uci() for move in board.move_stack[:6])


def result_of(board: chess.Board) -> tuple[str, str]:
    outcome = board.outcome(claim_draw=True)
    if outcome is not None:
        return outcome.result(), outcome.termination.name
    return "1/2-1/2", "adjudication"


def event_name(white_engine: str, black_engine: str) -> str:
    return f"Berlin match: lc0-{white_engine} (White) vs lc0-{black_engine} (Black)"


def short_label(engine: str) -> str:
    return engine


def resolve_mode(value: str | None, native: str) -> str:
    return native if value is None else value


def player_label(engine: str, n_games: int, q_mode: str, fpu_mode: str) -> str:
    native_q = NATIVE_Q[engine]
    native_fpu = NATIVE_FPU[engine]
    name = f"lc0-{engine} (games {n_games}"
    extras = []
    if q_mode != native_q:
        extras.append(f"Q={q_mode}")
    if fpu_mode != native_fpu:
        extras.append(f"FPU={fpu_mode}")
    if extras:
        name += ", " + ", ".join(extras)
    return name + ")"


def weights_path(engine: str, n_games: int) -> Path:
    folder = ROOT / f"lc0-{engine}" / "checkpoints" / "run10k"
    return folder / f"lc0-{engine}-games{n_games:06d}.pt"


def build_pgn(
    board: chess.Board,
    *,
    event: str,
    round_no: int,
    white_name: str,
    black_name: str,
    white_weights: Path,
    black_weights: Path,
    visits: int,
    result: str,
    termination: str,
    seed: int,
    ply_after_berlin: int,
    elapsed_s: float,
    fixed_colors: str,
    q_modes: tuple[str, str] = ("wl", "drawscore"),
    fpu_modes: tuple[str, str] = ("negation", "parent"),
) -> chess.pgn.Game:
    game = chess.pgn.Game()
    game.headers["Event"] = event
    game.headers["Site"] = "Tema e Diplomes"
    game.headers["Date"] = datetime.now(timezone.utc).strftime("%Y.%m.%d")
    game.headers["Round"] = str(round_no)
    game.headers["White"] = white_name
    game.headers["Black"] = black_name
    game.headers["Result"] = result
    game.headers["ECO"] = "C65"
    game.headers["Opening"] = "Ruy Lopez, Berlin Defence"
    game.headers["Variation"] = "Berlin Defence tabiya after 3...Nf6"
    game.headers["TimeControl"] = f"{visits} nodes/move"
    game.headers["PlyCount"] = str(len(board.move_stack))
    game.headers["Termination"] = termination
    game.headers["WhiteType"] = "program"
    game.headers["BlackType"] = "program"
    game.headers["WhiteWeights"] = str(white_weights)
    game.headers["BlackWeights"] = str(black_weights)
    game.headers["FixedColors"] = fixed_colors
    game.headers["WhiteQ"] = q_modes[0]
    game.headers["BlackQ"] = q_modes[1]
    game.headers["WhiteFPU"] = fpu_modes[0]
    game.headers["BlackFPU"] = fpu_modes[1]
    game.headers["BerlinPGN"] = BERLIN_PGN
    game.headers["BerlinFEN"] = BERLIN_FEN
    game.headers["BerlinProof"] = "first six plies are e2e4 e7e5 g1f3 b8c6 f1b5 g8f6"
    game.headers["Seed"] = str(seed)
    game.headers["GameTimeSec"] = f"{elapsed_s:.1f}"
    game.headers["PlyAfterBerlin"] = str(ply_after_berlin)
    node = game
    for i, move in enumerate(board.move_stack, start=1):
        node = node.add_variation(move)
        if i == 6:
            node.comment = f"Berlin Defence tabiya; FEN {BERLIN_FEN}"
    game.headers["Result"] = result
    return game


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def existing_rounds(games_dir: Path) -> set[int]:
    found: set[int] = set()
    if not games_dir.is_dir():
        return found
    for path in games_dir.glob("game_*.pgn"):
        try:
            found.add(int(path.stem.split("_")[1]))
        except (IndexError, ValueError):
            continue
    return found


def verify_pgn(path: Path) -> dict:
    with path.open(encoding="utf-8") as handle:
        game = chess.pgn.read_game(handle)
    if game is None:
        raise RuntimeError(f"empty PGN: {path}")
    board = game.board()
    for move in game.mainline_moves():
        board.push(move)
    six = first_six_uci(board)
    rewind = chess.Board()
    for uci in six:
        rewind.push_uci(uci)
    return {
        "file": path.name,
        "round": game.headers.get("Round"),
        "result": game.headers.get("Result"),
        "white": game.headers.get("White"),
        "black": game.headers.get("Black"),
        "first_six_uci": list(six),
        "berlin_moves_ok": six == BERLIN_UCI,
        "berlin_fen_ok": rewind.fen() == chess.Board(BERLIN_FEN).fen(),
        "ply_count": len(board.move_stack),
    }


def play_game(
    *,
    white_search,
    black_search,
    white_history_fn,
    black_history_fn,
    visits: int,
    temperature: float,
    temp_moves: int,
    max_ply: int,
) -> tuple[chess.Board, int]:
    board = berlin_start()
    ply_after = 0
    while not board.is_game_over(claim_draw=True) and ply_after < max_ply:
        tau = temperature if ply_after < temp_moves else 0.0
        searcher = white_search if board.turn == chess.WHITE else black_search
        hist_fn = white_history_fn if board.turn == chess.WHITE else black_history_fn
        history = hist_fn(board)
        result = searcher.search(
            board,
            history=history,
            visits=visits,
            add_noise=False,
            temperature=tau,
        )
        if result.move not in board.legal_moves:
            break
        board.push(result.move)
        ply_after += 1
    return board, ply_after


def summarize(proof_rows: list[dict], target: int, white_engine: str, black_engine: str) -> dict:
    scores = {"1-0": 0, "0-1": 0, "1/2-1/2": 0}
    for row in proof_rows:
        scores[row["result"]] = scores.get(row["result"], 0) + 1
    white_pts = scores["1-0"] + 0.5 * scores["1/2-1/2"]
    black_pts = scores["0-1"] + 0.5 * scores["1/2-1/2"]
    white_l = short_label(white_engine)
    black_l = short_label(black_engine)
    return {
        "games_completed": len(proof_rows),
        "games_target": target,
        "all_berlin_ok": all(r["berlin_moves_ok"] and r["berlin_fen_ok"] for r in proof_rows) if proof_rows else False,
        "fixed_colors": f"White=lc0-{white_engine} Black=lc0-{black_engine}",
        "scoreline": f"{white_l} {white_pts:g} - {black_pts:g} {black_l}",
        "white_wins": scores["1-0"],
        "black_wins": scores["0-1"],
        "draws": scores["1/2-1/2"],
        "white_points": white_pts,
        "black_points": black_pts,
        "white_score_pct": (100.0 * white_pts / len(proof_rows)) if proof_rows else 0.0,
        "black_score_pct": (100.0 * black_pts / len(proof_rows)) if proof_rows else 0.0,
    }


def make_player(
    engine: str,
    n_games: int,
    *,
    q_mode: str,
    fpu_mode: str,
    backend=None,
    pkg=None,
):
    if pkg is None:
        src = ROOT / f"lc0-{engine}" / "src"
        pkg = load_engine_pkg(src)
    cfg = pkg.load_config(ROOT / f"lc0-{engine}" / "configs" / "default.yaml")
    weights = weights_path(engine, n_games)
    if not weights.is_file():
        raise SystemExit(f"missing {engine} weights: {weights}")
    if backend is None:
        if engine == "variant":
            backend = pkg.DualNetworkBackend(cfg.network, device=cfg.device, weights_path=str(weights))
        else:
            backend = pkg.NetworkBackend(cfg.network, device=cfg.device, weights_path=str(weights))
    params = pkg.SearchParams.from_config(cfg.search, training=False)
    params.q_mode = q_mode
    params.fpu_mode = fpu_mode
    searcher = pkg.PUCTSearch(backend, params)
    return SimpleNamespace(
        engine=engine,
        name=player_label(engine, n_games, q_mode, fpu_mode),
        q_mode=q_mode,
        fpu_mode=fpu_mode,
        weights=weights,
        backend=backend,
        searcher=searcher,
        history_from_board=pkg.history_from_board,
        pkg=pkg,
    )


def black_comparison_payload(treatment: dict, control: dict) -> dict:
    t_pts = float(treatment["black_points"])
    c_pts = float(control["black_points"])
    t_n = int(treatment["games_completed"])
    c_n = int(control["games_completed"])
    t_pct = 100.0 * t_pts / t_n if t_n else 0.0
    c_pct = 100.0 * c_pts / c_n if c_n else 0.0
    delta = t_pts - c_pts
    if delta > 0.25:
        winner = "lc0-variant"
        verdict = (
            f"As Black against the same lc0-standard White, lc0-variant scored "
            f"{t_pts:g}/{t_n} vs lc0-standard Black {c_pts:g}/{c_n}."
        )
    elif delta < -0.25:
        winner = "lc0-standard"
        verdict = (
            f"As Black against the same lc0-standard White, lc0-standard scored "
            f"{c_pts:g}/{c_n} vs lc0-variant Black {t_pts:g}/{t_n}."
        )
    else:
        winner = "tie"
        verdict = (
            f"As Black against the same lc0-standard White, both scored about the same "
            f"(variant {t_pts:g}/{t_n}, standard {c_pts:g}/{c_n})."
        )
    return {
        "question": "Does lc0-standard or lc0-variant perform better as Black vs lc0-standard White?",
        "white_fixed": "lc0-standard",
        "better_as_black": winner,
        "verdict": verdict,
        "variant_as_black": {
            "points": t_pts,
            "wins": treatment["black_wins"],
            "draws": treatment["draws"],
            "losses": treatment["white_wins"],
            "score_pct": t_pct,
            "games": t_n,
        },
        "standard_as_black": {
            "points": c_pts,
            "wins": control["black_wins"],
            "draws": control["draws"],
            "losses": control["white_wins"],
            "score_pct": c_pct,
            "games": c_n,
        },
        "point_difference_variant_minus_standard": delta,
    }


def white_comparison_payload(treatment: dict, control: dict) -> dict:
    t_pts = float(treatment["white_points"])
    c_pts = float(control["white_points"])
    t_n = int(treatment["games_completed"])
    c_n = int(control["games_completed"])
    t_pct = 100.0 * t_pts / t_n if t_n else 0.0
    c_pct = 100.0 * c_pts / c_n if c_n else 0.0
    delta = t_pts - c_pts
    if delta > 0.25:
        winner = "lc0-variant"
        verdict = (
            f"As White against the same lc0-standard Black, lc0-variant scored "
            f"{t_pts:g}/{t_n} vs lc0-standard White {c_pts:g}/{c_n}."
        )
    elif delta < -0.25:
        winner = "lc0-standard"
        verdict = (
            f"As White against the same lc0-standard Black, lc0-standard scored "
            f"{c_pts:g}/{c_n} vs lc0-variant White {t_pts:g}/{t_n}."
        )
    else:
        winner = "tie"
        verdict = (
            f"As White against the same lc0-standard Black, both scored about the same "
            f"(variant {t_pts:g}/{t_n}, standard {c_pts:g}/{c_n})."
        )
    return {
        "question": "Does lc0-standard or lc0-variant perform better as White vs lc0-standard Black?",
        "black_fixed": "lc0-standard",
        "better_as_white": winner,
        "verdict": verdict,
        "variant_as_white": {
            "points": t_pts,
            "wins": treatment["white_wins"],
            "draws": treatment["draws"],
            "losses": treatment["black_wins"],
            "score_pct": t_pct,
            "games": t_n,
        },
        "standard_as_white": {
            "points": c_pts,
            "wins": control["white_wins"],
            "draws": control["draws"],
            "losses": control["black_wins"],
            "score_pct": c_pct,
            "games": c_n,
        },
        "point_difference_variant_minus_standard": delta,
    }


def _black_side_stats(summary: dict) -> dict:
    n = int(summary["games_completed"])
    pts = float(summary["black_points"])
    return {
        "points": pts,
        "wins": summary["black_wins"],
        "draws": summary["draws"],
        "losses": summary["white_wins"],
        "score_pct": (100.0 * pts / n) if n else 0.0,
        "games": n,
    }


def h3_black_comparison_payload(winonly: dict, variant: dict, control: dict) -> dict:
    wo = _black_side_stats(winonly)
    var = _black_side_stats(variant)
    std = _black_side_stats(control)
    ranking = sorted(
        [
            ("lc0-standard", std["points"]),
            ("lc0-winonly", wo["points"]),
            ("lc0-variant", var["points"]),
        ],
        key=lambda item: (-item[1], item[0]),
    )
    best = ranking[0][0]
    if ranking[0][1] - ranking[1][1] <= 0.25:
        best = "tie"
    verdict = (
        f"As Black against the same lc0-standard White (3440 freeze, 120 visits, N=100): "
        f"lc0-standard {std['points']:g}/{std['games']}, "
        f"lc0-winonly {wo['points']:g}/{wo['games']}, "
        f"lc0-variant {var['points']:g}/{var['games']}."
    )
    return {
        "question": (
            "As a new Black concept (win-only Q=W, one net, classic FPU), "
            "how does lc0-winonly score vs lc0-standard White compared with "
            "lc0-variant Black and lc0-standard Black at the same self-play freeze?"
        ),
        "white_fixed": "lc0-standard",
        "checkpoint_games": 3440,
        "visits": 120,
        "games": 100,
        "temp_moves_after_berlin": 8,
        "better_as_black": best,
        "ranking_as_black": [name for name, _ in ranking],
        "verdict": verdict,
        "winonly_as_black": wo,
        "variant_as_black": var,
        "standard_as_black": std,
        "point_difference_winonly_minus_standard": wo["points"] - std["points"],
        "point_difference_winonly_minus_variant": wo["points"] - var["points"],
        "concept": {
            "lc0-winonly": "one net; White Q=W-L; Black Q=W; classic FPU negation",
            "lc0-variant": "dual nets; DrawScore Q; parent-relative FPU",
            "lc0-standard": "one net; Q=W-L both colors; classic FPU negation",
        },
    }


def match_seed_base(
    n_games: int,
    white_engine: str,
    black_engine: str,
    white_q: str,
    black_q: str,
    white_fpu: str,
    black_fpu: str,
    seed_offset: int = 0,
) -> int:
    pairing = PAIR_OFFSETS[(white_engine, black_engine)]
    native = (
        white_q == NATIVE_Q[white_engine]
        and black_q == NATIVE_Q[black_engine]
        and white_fpu == NATIVE_FPU[white_engine]
        and black_fpu == NATIVE_FPU[black_engine]
    )
    extra = 0
    if not native:
        key = f"{white_engine}:{white_q}:{white_fpu}|{black_engine}:{black_q}:{black_fpu}"
        extra = 4_000_000 + (sum(ord(c) * (i + 1) for i, c in enumerate(key)) % 500_000)
    return 100_000 * n_games + pairing + extra + seed_offset


def search_overrides(
    white_engine: str,
    black_engine: str,
    white_q: str,
    black_q: str,
    white_fpu: str,
    black_fpu: str,
) -> bool:
    return not (
        white_q == NATIVE_Q[white_engine]
        and black_q == NATIVE_Q[black_engine]
        and white_fpu == NATIVE_FPU[white_engine]
        and black_fpu == NATIVE_FPU[black_engine]
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--games", type=int, default=100)
    parser.add_argument("--visits", type=int, default=120)
    parser.add_argument("--checkpoint-games", type=int, default=3440)
    parser.add_argument("--white-engine", choices=ENGINES, default="standard")
    parser.add_argument("--black-engine", choices=ENGINES, default="variant")
    parser.add_argument("--white-q", choices=Q_MODES, default=None)
    parser.add_argument("--black-q", choices=Q_MODES, default=None)
    parser.add_argument("--white-fpu", choices=FPU_MODES, default=None)
    parser.add_argument("--black-fpu", choices=FPU_MODES, default=None)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument(
        "--temp-moves",
        type=int,
        default=8,
        help="Visit-softmax sampling for this many plies after the Berlin, then greedy.",
    )
    parser.add_argument("--max-ply", type=int, default=512)
    parser.add_argument(
        "--seed-offset",
        type=int,
        default=0,
        help="Add to the match seed base so a rematch is an independent sample.",
    )
    parser.add_argument("--out-dir", default="")
    args = parser.parse_args()

    n_games = int(args.checkpoint_games)
    white_engine = args.white_engine
    black_engine = args.black_engine
    white_q = resolve_mode(args.white_q, NATIVE_Q[white_engine])
    black_q = resolve_mode(args.black_q, NATIVE_Q[black_engine])
    white_fpu = resolve_mode(args.white_fpu, NATIVE_FPU[white_engine])
    black_fpu = resolve_mode(args.black_fpu, NATIVE_FPU[black_engine])
    event = event_name(white_engine, black_engine)
    if search_overrides(white_engine, black_engine, white_q, black_q, white_fpu, black_fpu):
        event = f"{event} [Q {white_q}/{black_q}, FPU {white_fpu}/{black_fpu}]"
    fixed_colors = f"{white_engine}=White {black_engine}=Black"
    white_tag = ENGINE_TAG[white_engine]
    black_tag = ENGINE_TAG[black_engine]
    if args.out_dir:
        out_dir = Path(args.out_dir)
    elif search_overrides(white_engine, black_engine, white_q, black_q, white_fpu, black_fpu):
        out_dir = (
            ROOT
            / "tournament"
            / f"ablation_{white_tag}W_{black_tag}B_{black_q}_{black_fpu}_g{n_games}"
        )
    else:
        out_dir = ROOT / "tournament" / f"berlin_{white_tag}_white_vs_{black_tag}_black_g{n_games}"
    games_dir = out_dir / "games"
    games_dir.mkdir(parents=True, exist_ok=True)
    tournament_pgn = out_dir / "tournament.pgn"
    results_path = out_dir / "results.json"
    proof_path = out_dir / "berlin_proof.json"

    white = make_player(white_engine, n_games, q_mode=white_q, fpu_mode=white_fpu)
    same_weights = black_engine == white_engine
    if same_weights:
        black = make_player(
            black_engine,
            n_games,
            q_mode=black_q,
            fpu_mode=black_fpu,
            backend=white.backend,
            pkg=white.pkg,
        )
        print(
            f"shared {black_engine} weights; White Q={white_q}/{white_fpu} Black Q={black_q}/{black_fpu}",
            flush=True,
        )
    else:
        black = make_player(black_engine, n_games, q_mode=black_q, fpu_mode=black_fpu)

    print("=" * 72, flush=True)
    print(f"  {event}", flush=True)
    print(f"  White: {white.weights.name}  Q={white_q} FPU={white_fpu}", flush=True)
    print(f"  Black: {black.weights.name}  Q={black_q} FPU={black_fpu}", flush=True)
    print(f"  Games: {args.games}  visits={args.visits}/move  out={out_dir}", flush=True)
    print("=" * 72, flush=True)
    print(
        f"white engine=lc0-{white.engine} device={white.backend.device} weights={white.backend.weights_path}",
        flush=True,
    )
    extra = ""
    if getattr(black.backend, "cloned_from_single", None) is not None:
        extra = f" cloned_from_single={black.backend.cloned_from_single}"
    print(
        f"black engine=lc0-{black.engine} device={black.backend.device} "
        f"weights={black.backend.weights_path}{extra}",
        flush=True,
    )

    done = existing_rounds(games_dir)
    if done:
        print(f"resume: {len(done)} PGN(s) already on disk", flush=True)

    seed_base = match_seed_base(
        n_games,
        white_engine,
        black_engine,
        white_q,
        black_q,
        white_fpu,
        black_fpu,
        seed_offset=int(args.seed_offset),
    )
    started = time.perf_counter()

    def payload_for(rows: list[dict]) -> dict:
        return {
            "event": event,
            "white": white.name,
            "black": black.name,
            "white_engine": white_engine,
            "black_engine": black_engine,
            "white_weights": str(white.weights),
            "black_weights": str(black.weights),
            "white_q": white_q,
            "black_q": black_q,
            "white_fpu": white_fpu,
            "black_fpu": black_fpu,
            "visits": args.visits,
            "temperature_after_berlin": args.temperature,
            "temp_moves_after_berlin": args.temp_moves,
            "seed_offset": int(args.seed_offset),
            "opening": BERLIN_PGN,
            "opening_fen": BERLIN_FEN,
            "fixed_colors": True,
            "summary": summarize(rows, args.games, white_engine, black_engine),
            "games": rows,
        }

    for round_no in range(1, args.games + 1):
        game_path = games_dir / f"game_{round_no:03d}.pgn"
        if round_no in done and game_path.is_file():
            print(f"game {round_no}/{args.games} | skip existing {game_path.name}", flush=True)
            continue
        seed = seed_base + round_no
        seed_everything(seed)
        t0 = time.perf_counter()
        board, ply_after = play_game(
            white_search=white.searcher,
            black_search=black.searcher,
            white_history_fn=white.history_from_board,
            black_history_fn=black.history_from_board,
            visits=args.visits,
            temperature=args.temperature,
            temp_moves=args.temp_moves,
            max_ply=args.max_ply,
        )
        elapsed = time.perf_counter() - t0
        if first_six_uci(board) != BERLIN_UCI:
            raise RuntimeError(f"game {round_no} lost the Berlin opening: {first_six_uci(board)}")
        result, termination = result_of(board)
        game = build_pgn(
            board,
            event=event,
            round_no=round_no,
            white_name=white.name,
            black_name=black.name,
            white_weights=white.weights,
            black_weights=black.weights,
            visits=args.visits,
            result=result,
            termination=termination,
            seed=seed,
            ply_after_berlin=ply_after,
            elapsed_s=elapsed,
            fixed_colors=fixed_colors,
            q_modes=(white_q, black_q),
            fpu_modes=(white_fpu, black_fpu),
        )
        text = str(game) + "\n\n"
        write_text(game_path, text)
        with tournament_pgn.open("a", encoding="utf-8") as handle:
            handle.write(text)
        total_elapsed = time.perf_counter() - started
        remaining = args.games - round_no
        rate = round_no / total_elapsed * 3600 if total_elapsed > 1 else 0.0
        print(
            f"game {round_no}/{args.games} | {result} | {termination} | "
            f"{ply_after} ply after Berlin | {elapsed:.1f}s | "
            f"{rate:.1f} games/h | saved {game_path.name}",
            flush=True,
        )
        live_rows = []
        for n in range(1, round_no + 1):
            p = games_dir / f"game_{n:03d}.pgn"
            if p.is_file():
                live_rows.append(verify_pgn(p))
        write_text(results_path, json.dumps(payload_for(live_rows), indent=2))
        if remaining == 0:
            break

    proof_rows = []
    combined: list[str] = []
    for round_no in range(1, args.games + 1):
        path = games_dir / f"game_{round_no:03d}.pgn"
        if path.is_file():
            proof_rows.append(verify_pgn(path))
            combined.append(path.read_text(encoding="utf-8").rstrip() + "\n\n")
    if combined:
        write_text(tournament_pgn, "".join(combined))
    payload = payload_for(proof_rows)
    summary = payload["summary"]
    write_text(results_path, json.dumps(payload, indent=2))
    write_text(proof_path, json.dumps({"summary": summary, "games": proof_rows}, indent=2))

    if (
        white_engine == "standard"
        and black_engine == "standard"
        and not search_overrides(white_engine, black_engine, white_q, black_q, white_fpu, black_fpu)
    ):
        treatment_dir = ROOT / "tournament" / f"berlin_std_white_vs_var_black_g{n_games}"
        treatment_results = treatment_dir / "results.json"
        if treatment_results.is_file():
            treatment = json.loads(treatment_results.read_text(encoding="utf-8"))
            comparison = black_comparison_payload(treatment["summary"], summary)
            write_text(out_dir / "black_comparison.json", json.dumps(comparison, indent=2))
            canonical = ROOT / "tournament" / f"berlin_std_white_vs_std_black_g{n_games}"
            if out_dir.resolve() == canonical.resolve():
                compare_path = ROOT / "tournament" / f"black_comparison_g{n_games}.json"
                write_text(compare_path, json.dumps(comparison, indent=2))
                print(f"comparison: {compare_path}", flush=True)
            print(comparison["verdict"], flush=True)

    if (
        white_engine == "variant"
        and black_engine == "standard"
        and not search_overrides(white_engine, black_engine, white_q, black_q, white_fpu, black_fpu)
    ):
        control_dir = ROOT / "tournament" / f"berlin_std_white_vs_std_black_g{n_games}"
        control_results = control_dir / "results.json"
        if control_results.is_file():
            control = json.loads(control_results.read_text(encoding="utf-8"))
            comparison = white_comparison_payload(summary, control["summary"])
            compare_path = ROOT / "tournament" / f"white_comparison_g{n_games}.json"
            write_text(compare_path, json.dumps(comparison, indent=2))
            write_text(out_dir / "white_comparison.json", json.dumps(comparison, indent=2))
            print(comparison["verdict"], flush=True)
            print(f"comparison: {compare_path}", flush=True)

    if (
        white_engine == "standard"
        and black_engine == "winonly"
        and not search_overrides(white_engine, black_engine, white_q, black_q, white_fpu, black_fpu)
    ):
        variant_path = ROOT / "tournament" / f"berlin_std_white_vs_var_black_g{n_games}" / "results.json"
        control_path = ROOT / "tournament" / f"berlin_std_white_vs_std_black_g{n_games}" / "results.json"
        if variant_path.is_file() and control_path.is_file():
            variant = json.loads(variant_path.read_text(encoding="utf-8"))
            control = json.loads(control_path.read_text(encoding="utf-8"))
            comparison = h3_black_comparison_payload(
                summary, variant["summary"], control["summary"]
            )
            compare_path = ROOT / "tournament" / f"black_comparison_h3_g{n_games}.json"
            write_text(compare_path, json.dumps(comparison, indent=2))
            write_text(out_dir / "h3_black_comparison.json", json.dumps(comparison, indent=2))
            print(comparison["verdict"], flush=True)
            print(f"comparison: {compare_path}", flush=True)

    print("-" * 72, flush=True)
    print(
        f"TOURNAMENT DONE {summary['games_completed']}/{summary['games_target']} | "
        f"{summary['scoreline']} | berlin_ok={summary['all_berlin_ok']}",
        flush=True,
    )
    print(f"PGNs: {tournament_pgn}", flush=True)
    print(f"per-game: {games_dir}", flush=True)
    if not summary["all_berlin_ok"]:
        raise SystemExit("Berlin proof failed for at least one game")


if __name__ == "__main__":
    main()
