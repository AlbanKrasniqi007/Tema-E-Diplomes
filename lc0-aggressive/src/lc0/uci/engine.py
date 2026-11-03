"""UCI adapter for the H3 win-only Black replica, including Nibbler live analysis."""

from __future__ import annotations

import sys
import threading

import chess

from lc0.chess.opening import BERLIN_FEN, BERLIN_NAME, BERLIN_PGN
from lc0.config import EngineConfig, load_config
from lc0.neural.backend import NetworkBackend
from lc0.search.node import Node
from lc0.search.params import SearchParams
from lc0.search.puct import PUCTSearch

ENGINE_ID = "lc0-winonly"
ENGINE_NAME = "lc0-winonly (Lc0 Berlin H3)"
ENGINE_AUTHOR = "Tema e Diplomes"
INFINITE_VISITS = 10**12


def q_to_cp(q: float) -> int:
    """Lc0 2019 mapping: cp = 111.714640912 * tan(1.5620688421 * Q)."""
    q = max(-0.999, min(0.999, q))
    return int(round(111.714640912 * float(__import__("math").tan(1.5620688421 * q))))


def wdl_permille(edge_or_node) -> tuple[int, int, int]:
    n = max(getattr(edge_or_node, "n", 0), 1)
    w = int(round(1000 * edge_or_node.w / n))
    d = int(round(1000 * edge_or_node.d / n))
    l = max(0, 1000 - w - d)
    return w, d, l


class UCIEngine:
    def __init__(self, cfg: EngineConfig | None = None) -> None:
        self.cfg = cfg or EngineConfig()
        self.backend = NetworkBackend(
            self.cfg.network,
            device=self.cfg.device,
            weights_path=self.cfg.weights_path or None,
        )
        self.params = SearchParams.from_config(self.cfg.search, training=False)
        self.searcher = PUCTSearch(self.backend, self.params)
        self.board = chess.Board()
        self.default_visits = self.cfg.train.visits
        self.verbose_move_stats = True
        self.log_live_stats = True
        self._stop = threading.Event()
        self._search_thread: threading.Thread | None = None

    def emit(self, line: str) -> None:
        sys.stdout.write(line + "\n")
        sys.stdout.flush()

    def handle(self, line: str) -> list[str]:
        line = line.strip()
        if not line:
            return []
        parts = line.split()
        cmd = parts[0]
        if cmd == "uci":
            return [
                f"id name {ENGINE_NAME}",
                f"id author {ENGINE_AUTHOR}",
                f"option name WeightsFile type string default {self.cfg.weights_path}",
                "option name Visits type spin default 200 min 2 max 10000000",
                "option name MiniBatchSize type spin default 32 min 1 max 512",
                "option name VerboseMoveStats type check default true",
                "option name LogLiveStats type check default true",
                "option name MultiPV type spin default 1 min 1 max 500",
                "option name QMode type combo default asymmetric var wl var drawscore var winonly var asymmetric",
                "option name FpuMode type combo default negation var negation var parent",
                f"option name OpeningName type string default {BERLIN_NAME}",
                f"option name OpeningFEN type string default {BERLIN_FEN}",
                f"option name OpeningPGN type string default {BERLIN_PGN}",
                "uciok",
            ] + self._weights_info_lines()
        if cmd == "isready":
            self._wait_search()
            return self._weights_info_lines() + ["readyok"]
        if cmd == "ucinewgame":
            self._wait_search()
            self.board.reset()
            return []
        if cmd == "position":
            self._wait_search()
            self._set_position(parts[1:])
            return []
        if cmd == "go":
            return self._go(parts[1:])
        if cmd == "stop":
            self._stop.set()
            self._wait_search()
            return []
        if cmd == "setoption":
            return self._setoption(parts[1:])
        if cmd == "quit":
            self._stop.set()
            self._wait_search()
            return ["__quit__"]
        if cmd == "d":
            return [str(self.board)]
        return []

    def _wait_search(self) -> None:
        thread = self._search_thread
        if thread is not None and thread.is_alive() and thread is not threading.current_thread():
            thread.join()
        self._search_thread = None

    def _set_position(self, tokens: list[str]) -> None:
        if not tokens:
            return
        if tokens[0] == "startpos":
            self.board.reset()
            rest = tokens[1:]
        elif tokens[0] == "fen":
            if "moves" in tokens:
                idx = tokens.index("moves")
                fen = " ".join(tokens[1:idx])
                rest = tokens[idx:]
            else:
                fen = " ".join(tokens[1:])
                rest = []
            self.board = chess.Board(fen)
        else:
            return
        if rest and rest[0] == "moves":
            for uci in rest[1:]:
                try:
                    self.board.push_uci(uci)
                except ValueError:
                    break

    def _weights_info_lines(self) -> list[str]:
        """Lc0-style lines Nibbler shows in the engine log."""
        path = getattr(self.backend, "weights_path", None)
        err = getattr(self.backend, "weights_error", None)
        if path:
            return [
                f"info string Loading weights file from: {path}",
                f"info string Network weights applied: {path}",
                f"info string engine={ENGINE_ID} q_mode={self.params.q_mode} fpu_mode={self.params.fpu_mode} device={self.backend.device}",
            ]
        if err:
            return [f"info string {err}"]
        return [
            f"info string Network weights: NONE (random untrained net) engine={ENGINE_ID} q_mode={self.params.q_mode} fpu={self.params.fpu_mode}",
            "info string Set WeightsFile to a .pt checkpoint. Official Lc0 .pb.gz files cannot be used here.",
        ]

    def _setoption(self, tokens: list[str]) -> list[str]:
        if "name" not in tokens:
            return []
        name_i = tokens.index("name")
        if "value" in tokens:
            value_i = tokens.index("value")
            name = " ".join(tokens[name_i + 1 : value_i]).lower()
            value = " ".join(tokens[value_i + 1 :])
        else:
            name = " ".join(tokens[name_i + 1 :]).lower()
            value = ""
        if name == "weightsfile" and value:
            try:
                self.backend.load(value)
            except Exception as exc:
                if not self.backend.weights_error:
                    self.backend.weights_error = str(exc)
            return self._weights_info_lines()
        elif name == "visits":
            self.default_visits = int(value)
        elif name == "minibatchsize":
            self.params.minibatch_size = int(value)
        elif name == "verbosemovestats":
            self.verbose_move_stats = value.lower() not in {"false", "0"}
        elif name == "loglivestats":
            self.log_live_stats = value.lower() not in {"false", "0"}
        elif name == "qmode" and value:
            mode = value.strip().lower()
            if mode in {"wl", "drawscore", "winonly", "asymmetric"}:
                self.params.q_mode = mode
        elif name == "fpumode" and value:
            mode = value.strip().lower()
            if mode in {"negation", "parent"}:
                self.params.fpu_mode = mode
        return []

    def _go(self, tokens: list[str]) -> list[str]:
        self._wait_search()
        self._stop.clear()
        args = _parse_go(tokens)
        infinite = "infinite" in args
        visits = INFINITE_VISITS if infinite else args.get("nodes", self.default_visits)
        time_budget = None if infinite else _time_budget_seconds(args, self.board.turn == chess.WHITE)
        searchmoves = args.get("searchmoves") or None
        board = self.board.copy(stack=True)

        if infinite:
            self._search_thread = threading.Thread(
                target=self._run_search,
                kwargs={
                    "board": board,
                    "visits": int(visits),
                    "time_budget": time_budget,
                    "searchmoves": searchmoves,
                    "live": True,
                    "collect": None,
                },
                daemon=True,
            )
            self._search_thread.start()
            return []

        collected: list[str] = []
        try:
            self._run_search(
                board=board,
                visits=int(visits),
                time_budget=time_budget,
                searchmoves=searchmoves,
                live=False,
                collect=collected,
            )
        except Exception as exc:
            legal = list(board.legal_moves)
            fallback = legal[0].uci() if legal else "0000"
            collected.append(f"info string uci go failed: {exc}")
            collected.append(f"bestmove {fallback}")
        return collected

    def _run_search(
        self,
        board: chess.Board,
        visits: int,
        time_budget: float | None,
        searchmoves,
        live: bool,
        collect: list[str] | None,
    ) -> None:
        def sink(line: str) -> None:
            if collect is not None:
                collect.append(line)
            if live:
                self.emit(line)

        def on_info(root: Node, elapsed: float) -> None:
            sink(self._info_line(root, elapsed))
            if self.verbose_move_stats or self.log_live_stats:
                for stats in self.searcher.verbose_move_stats(root):
                    sink(f"info string {stats}")

        result = self.searcher.search(
            board,
            visits=visits,
            time_budget=time_budget,
            add_noise=False,
            temperature=0.0,
            stop_event=self._stop,
            info_callback=on_info if live else None,
            info_every=max(8, self.params.minibatch_size),
            searchmoves=searchmoves,
        )
        sink(self._info_line(result.root, result.elapsed))
        if self.verbose_move_stats:
            for stats in self.searcher.verbose_move_stats(result.root):
                sink(f"info string {stats}")
        move = result.move
        if move not in board.legal_moves:
            legal = list(board.legal_moves)
            move = legal[0] if legal else chess.Move.null()
        sink(f"bestmove {move.uci()}")

    def _info_line(self, root: Node, elapsed: float) -> str:
        if not root.edges:
            return f"info nodes {root.n} time {int(elapsed * 1000)}"
        best = root.best_edge()
        nps = int(root.n / max(elapsed, 1e-6))
        score = q_to_cp(best.q)
        w, d, l = wdl_permille(best)
        pv = self.searcher._principal_variation(root)
        pv_str = " ".join(move.uci() for move in pv)
        return (
            f"info depth {max(len(pv), 1)} seldepth {max(len(pv), 1)} "
            f"score cp {score} wdl {w} {d} {l} nodes {root.n} nps {nps} "
            f"time {int(elapsed * 1000)} pv {pv_str}"
        )


def _time_budget_seconds(args: dict, stm_is_white: bool) -> float | None:
    """UCI clocks. CuteChess omits winc/binc when the increment is 0."""
    if "movetime" in args:
        return max(0.05, args["movetime"] / 1000.0)
    if "wtime" not in args and "btime" not in args:
        return None
    if stm_is_white:
        remaining = int(args.get("wtime", args.get("btime", 0)) or 0)
        increment = int(args.get("winc", 0) or 0)
    else:
        remaining = int(args.get("btime", args.get("wtime", 0)) or 0)
        increment = int(args.get("binc", 0) or 0)
    budget = remaining / 1000.0 / 25.0 + increment / 1000.0
    cap = remaining / 1000.0 * 0.3 if remaining > 0 else 0.05
    return max(0.05, min(budget, cap))


def _parse_go(tokens: list[str]) -> dict:
    out: dict = {}
    i = 0
    keys = {"wtime", "btime", "winc", "binc", "movetime", "nodes", "depth", "movestogo"}
    while i < len(tokens):
        key = tokens[i]
        if key == "infinite":
            out["infinite"] = 1
            i += 1
            continue
        if key == "searchmoves":
            moves = []
            i += 1
            while i < len(tokens) and tokens[i] not in keys and tokens[i] != "infinite":
                moves.append(chess.Move.from_uci(tokens[i]))
                i += 1
            out["searchmoves"] = moves
            continue
        if key in keys and i + 1 < len(tokens):
            out[key] = int(tokens[i + 1])
            i += 2
        else:
            i += 1
    return out


def main(config_path: str | None = None) -> None:
    try:
        sys.stdout.reconfigure(line_buffering=True)
        sys.stdin.reconfigure(line_buffering=True)
    except Exception:
        pass
    cfg = load_config(config_path) if config_path else EngineConfig()
    print(f"engine={ENGINE_ID} loaded", file=sys.stderr, flush=True)
    engine = UCIEngine(cfg)
    while True:
        line = sys.stdin.readline()
        if not line:
            break
        replies = engine.handle(line)
        for reply in replies:
            if reply == "__quit__":
                return
            engine.emit(reply)
