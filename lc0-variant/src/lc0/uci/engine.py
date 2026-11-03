"""UCI adapter for the dual-agent Berlin variant, including Nibbler live analysis."""

from __future__ import annotations

import sys
import threading

import chess

from lc0.chess.opening import BERLIN_FEN, BERLIN_NAME, BERLIN_PGN
from lc0.config import EngineConfig, load_config
from lc0.neural.dual import DualNetworkBackend
from lc0.search.node import Node
from lc0.search.params import SearchParams
from lc0.search.puct import PUCTSearch

ENGINE_ID = "lc0-variant"
ENGINE_NAME = "lc0-variant (Lc0 Berlin dual co-evolution)"
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
        self.backend = DualNetworkBackend(
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
        self.multipv = 500
        self._stop = threading.Event()
        self._search_thread: threading.Thread | None = None
        self._tree: Node | None = None
        self._tree_key: str | None = None
        self._tree_time: float = 0.0

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
                "option name MultiPV type spin default 500 min 1 max 500",
                f"option name OpeningName type string default {BERLIN_NAME}",
                f"option name OpeningFEN type string default {BERLIN_FEN}",
                f"option name OpeningPGN type string default {BERLIN_PGN}",
                "option name WhiteValue type string default win_equals_draw",
                "option name BlackValue type string default draw_equals_loss",
                "uciok",
            ] + self._weights_info_lines()
        if cmd == "isready":
            self._wait_search()
            return self._weights_info_lines() + ["readyok"]
        if cmd == "ucinewgame":
            self._wait_search()
            self.board.reset()
            self._clear_tree()
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

    def _clear_tree(self) -> None:
        self._tree = None
        self._tree_key = None
        self._tree_time = 0.0

    def _position_key(self, board: chess.Board) -> str:
        return board.fen()

    def _reuse_tree_for_board(self, new_board: chess.Board) -> None:
        """Keep the MCTS tree across Halt/Go; descend if one or more moves were played."""
        if self._tree is None or not self._tree_key:
            return
        new_key = self._position_key(new_board)
        if new_key == self._tree_key:
            return
        extra = _continuation_moves(self._tree_key, new_board)
        if extra is None:
            self._clear_tree()
            return
        node = self._tree
        for move in extra:
            edge = next((item for item in node.edges if item.move == move), None)
            if edge is None or edge.child is None:
                self._clear_tree()
                return
            node = edge.child
        self._tree = node
        self._tree_key = new_key
        self._tree_time = 0.0

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
        self._reuse_tree_for_board(self.board)

    def _weights_info_lines(self) -> list[str]:
        """Lc0-style lines Nibbler shows in the engine log."""
        path = getattr(self.backend, "weights_path", None)
        err = getattr(self.backend, "weights_error", None)
        cloned = bool(getattr(self.backend, "cloned_from_single", False))
        agents = "info string agents=white,black"
        if path:
            clone_note = (
                "info string cloned_from_single=true (shared-net checkpoint copied into both specialists)"
                if cloned
                else "info string cloned_from_single=false (dual checkpoint)"
            )
            return [
                f"info string Loading weights file from: {path}",
                f"info string Network weights applied: {path}",
                f"info string engine={ENGINE_ID} device={self.backend.device} agents=white,black",
                clone_note,
            ]
        if err:
            return [f"info string {err}", agents]
        return [
            f"info string Network weights: NONE (random untrained dual nets) engine={ENGINE_ID}",
            agents,
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
            self._clear_tree()
            return self._weights_info_lines()
        elif name == "visits":
            self.default_visits = int(value)
        elif name == "minibatchsize":
            self.params.minibatch_size = int(value)
        elif name == "verbosemovestats":
            self.verbose_move_stats = value.lower() not in {"false", "0"}
        elif name == "loglivestats":
            self.log_live_stats = value.lower() not in {"false", "0"}
        elif name == "multipv":
            self.multipv = max(1, min(500, int(value)))
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
        base_time = self._tree_time if self._tree_key == self._position_key(board) else 0.0

        def sink(line: str) -> None:
            if collect is not None:
                collect.append(line)
            if live:
                self.emit(line)

        def on_info(root: Node, elapsed: float) -> None:
            for line in self._info_lines(root, base_time + elapsed):
                sink(line)
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
            root=self._tree if self._tree_key == self._position_key(board) else None,
        )
        self._tree = result.root
        self._tree_key = self._position_key(board)
        self._tree_time += result.elapsed
        for line in self._info_lines(result.root, self._tree_time):
            sink(line)
        if self.verbose_move_stats:
            for stats in self.searcher.verbose_move_stats(result.root):
                sink(f"info string {stats}")
        move = result.move
        if move not in board.legal_moves:
            legal = list(board.legal_moves)
            move = legal[0] if legal else chess.Move.null()
        sink(f"bestmove {move.uci()}")

    def _info_lines(self, root: Node, elapsed: float) -> list[str]:
        if not root.edges:
            return [f"info nodes {root.n} time {int(elapsed * 1000)}"]
        ranked = sorted(root.edges, key=lambda e: (e.n, e.q, e.prior), reverse=True)
        k = min(max(self.multipv, 1), len(ranked))
        nps = int(root.n / max(elapsed, 1e-6))
        time_ms = int(elapsed * 1000)
        lines: list[str] = []
        for i, edge in enumerate(ranked[:k], start=1):
            score = q_to_cp(edge.chess_q)
            w, d, l = wdl_permille(edge)
            pv = self.searcher.pv_from_edge(edge)
            pv_str = " ".join(move.uci() for move in pv)
            multipv = f" multipv {i}" if self.multipv > 1 else ""
            lines.append(
                f"info depth {max(len(pv), 1)} seldepth {max(len(pv), 1)}"
                f"{multipv} score cp {score} wdl {w} {d} {l} nodes {root.n} "
                f"nps {nps} time {time_ms} pv {pv_str}"
            )
        return lines


def _continuation_moves(old_fen: str, new_board: chess.Board) -> list[chess.Move] | None:
    """Moves that turn ``old_fen`` into ``new_board``, or None if unrelated."""
    try:
        old = chess.Board(old_fen)
    except ValueError:
        return None
    if old.fen() == new_board.fen():
        return []
    rewind = new_board.copy(stack=True)
    undone: list[chess.Move] = []
    while rewind.move_stack:
        undone.append(rewind.pop())
        if rewind.fen() == old.fen():
            undone.reverse()
            return undone
    for move in old.legal_moves:
        probe = old.copy(stack=False)
        probe.push(move)
        if probe.fen() == new_board.fen():
            return [move]
    return None


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
