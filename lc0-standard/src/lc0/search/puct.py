"""Classic PUCT / MCTS search.

Follows the AlphaZero loop documented at
https://lczero.org/dev/lc0/search/alphazero/ and the classic worker steps
at https://lczero.org/dev/wiki/technical-explanation-of-leela-chess-zero/ :

1. gather a minibatch of leaves with PUCT
2. evaluate them with the neural network (or assign a terminal value)
3. expand leaves with the policy prior
4. back up WDL / Q, using unscored virtual visits (N in flight, Q unchanged)

Move selection at the root uses visit count (temperature optional).
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass

import chess
import numpy as np

from lc0.chess.encoder import encode_position
from lc0.chess.policy_index import POLICY_SIZE
from lc0.chess.policy_map import move_to_policy_index
from lc0.neural.backend import NNOutput, NetworkBackend
from lc0.search.node import Edge, Node
from lc0.search.params import SearchParams
from lc0.value import search_q


@dataclass
class SearchResult:
    move: chess.Move
    visits: int
    q: float
    pv: list[chess.Move]
    root: Node
    policy_target: np.ndarray
    elapsed: float
    nodes: int


@dataclass
class _PathStep:
    parent: Node
    edge: Edge


@dataclass
class _LeafTask:
    board: chess.Board
    history: list[chess.Board]
    path: list[_PathStep]
    node: Node
    is_collision: bool = False


class PUCTSearch:
    def __init__(self, backend: NetworkBackend, params: SearchParams | None = None) -> None:
        self.backend = backend
        self.params = params or SearchParams()

    def search(
        self,
        board: chess.Board,
        history: list[chess.Board] | None = None,
        visits: int = 100,
        time_budget: float | None = None,
        add_noise: bool = False,
        temperature: float = 0.0,
        stop_event=None,
        info_callback=None,
        info_every: int = 32,
        searchmoves: list[chess.Move] | None = None,
    ) -> SearchResult:
        started = time.perf_counter()
        history = list(history) if history is not None else _history_from_board(board)
        root = Node()
        root_eval = self._evaluate_and_expand(
            root, board, history, add_noise=add_noise, at_root=True
        )
        if searchmoves:
            allowed = set(searchmoves)
            root.edges = [edge for edge in root.edges if edge.move in allowed]
        if not root.edges and not root.terminal:
            legal = list(board.legal_moves)
            if legal:
                root.edges = [
                    Edge(
                        move=move,
                        prior=1.0 / len(legal),
                        stm_white=board.turn == chess.WHITE,
                        q_mode=self.params.q_mode,
                    )
                    for move in legal
                ]
        root.n = 1
        root.w, root.d, root.l, root.m_sum = root_eval.w, root_eval.d, root_eval.l, root_eval.m
        last_info = 0

        collisions = 0
        while root.n < visits and root.edges:
            if stop_event is not None and stop_event.is_set():
                break
            if time_budget is not None and (time.perf_counter() - started) >= time_budget:
                break
            batch = self._gather_minibatch(root, board, history)
            real = [task for task in batch if not task.is_collision]
            if not real:
                collisions += 1
                if collisions >= self.params.max_collision_events:
                    break
                self._undo_virtuals(batch)
                continue
            collisions = 0
            self._evaluate_batch(real)
            for task in batch:
                if task.is_collision:
                    self._undo_virtuals([task])
                else:
                    self._backup(task)
            if info_callback is not None and root.n - last_info >= info_every:
                last_info = root.n
                info_callback(root, time.perf_counter() - started)

        if not root.edges:
            policy_target = np.zeros(POLICY_SIZE, dtype=np.float32)
            result = SearchResult(
                move=chess.Move.null(),
                visits=0,
                q=root.q,
                pv=[],
                root=root,
                policy_target=policy_target,
                elapsed=time.perf_counter() - started,
                nodes=root.n,
            )
            if info_callback is not None:
                info_callback(root, result.elapsed)
            return result

        move = self._select_root_move(root, board.turn == chess.BLACK, temperature)
        policy_target = self._visit_policy(root, board.turn == chess.BLACK)
        pv = self._principal_variation(root)
        best = next(edge for edge in root.edges if edge.move == move)
        result = SearchResult(
            move=move,
            visits=best.n,
            q=best.q,
            pv=pv,
            root=root,
            policy_target=policy_target,
            elapsed=time.perf_counter() - started,
            nodes=root.n,
        )
        if info_callback is not None:
            info_callback(root, result.elapsed)
        return result

    def verbose_move_stats(self, root: Node) -> list[str]:
        """Lc0-style `info string` lines that Nibbler parses for N/P/Q/U/V/WDL."""
        if not root.edges:
            return []
        parent_n = max(root.n, 1)
        cpuct = self.params.compute_cpuct(parent_n)
        u_coeff = cpuct * math.sqrt(parent_n)
        fpu = self._fpu(root)
        lines: list[str] = []
        for edge in sorted(root.edges, key=lambda e: (e.n, e.q, e.prior), reverse=True):
            q = edge.q if edge.n > 0 else fpu
            u = edge.u(u_coeff)
            n = edge.n
            d = (edge.d / n) if n else 0.0
            m = edge.m
            v = q
            if edge.child is not None and edge.child.n:
                v = -edge.child.q
            try:
                nn_idx = move_to_policy_index(edge.move, False)
            except KeyError:
                nn_idx = 0
            lines.append(
                f"{edge.move.uci():6}  ({nn_idx:4}) "
                f"N: {n:7d} (+{edge.n_in_flight:2d}) (P: {edge.prior * 100:5.2f}%) "
                f"(WL: {q:8.5f}) (D: {d:5.3f}) (M: {m:4.1f}) "
                f"(Q: {q:8.5f}) (U: {u:6.5f}) (S: {q + u:8.5f}) (V: {v:7.4f})"
            )
        return lines

    def _gather_minibatch(
        self,
        root: Node,
        root_board: chess.Board,
        root_history: list[chess.Board],
    ) -> list[_LeafTask]:
        batch: list[_LeafTask] = []
        target = max(1, self.params.minibatch_size)
        collisions = 0
        while len(batch) < target and collisions < self.params.max_collision_events:
            board = root_board.copy(stack=True)
            history = [b.copy(stack=False) for b in root_history]
            path: list[_PathStep] = []
            node = root
            while node.edges and not node.terminal:
                edge = self._select_edge(node)
                edge.n_in_flight += 1
                path.append(_PathStep(parent=node, edge=edge))
                board.push(edge.move)
                history.append(board.copy(stack=False))
                if edge.child is None:
                    break
                node = edge.child
            if not path:
                break
            last = path[-1].edge
            if last.child is None and last.n_in_flight > 1:
                batch.append(
                    _LeafTask(board=board, history=history, path=path, node=node, is_collision=True)
                )
                collisions += 1
                continue
            batch.append(_LeafTask(board=board, history=history, path=path, node=node))
        return batch

    def _select_edge(self, node: Node) -> Edge:
        parent_n = max(node.n, 1)
        cpuct = self.params.compute_cpuct(parent_n)
        u_coeff = cpuct * math.sqrt(parent_n)
        fpu = self._fpu(node)
        best: Edge | None = None
        best_score = -1e9
        for edge in node.edges:
            q = edge.q if edge.n > 0 else fpu
            score = q + edge.u(u_coeff)
            if score > best_score:
                best_score = score
                best = edge
        assert best is not None
        return best

    def _leaf_q(self, w: float, d: float, l: float, stm_white: bool) -> float:
        return search_q(w, d, l, stm_white, self.params.q_mode)

    def _annotate(self, node: Node, stm_white: bool) -> Node:
        node.stm_white = stm_white
        node.q_mode = self.params.q_mode
        return node

    def _fpu(self, node: Node) -> float:
        if self.params.fpu_absolute:
            return self.params.fpu_value
        reduction = self.params.fpu_value * math.sqrt(node.visited_policy)
        if self.params.fpu_mode == "parent":
            return node.q - reduction
        return -node.q - reduction

    def _evaluate_and_expand(
        self,
        node: Node,
        board: chess.Board,
        history: list[chess.Board],
        add_noise: bool,
        at_root: bool = False,
    ) -> NNOutput:
        stm_white = board.turn == chess.WHITE
        node.stm_white = stm_white
        node.q_mode = self.params.q_mode
        if self._is_terminal(board, at_root=at_root):
            w, d, l = self._terminal_wdl(board)
            node.terminal = True
            node.w, node.d, node.l = w, d, l
            node.n = max(node.n, 1)
            return NNOutput(
                policy=np.zeros(POLICY_SIZE, dtype=np.float32),
                logits=np.zeros(POLICY_SIZE, dtype=np.float32),
                w=w,
                d=d,
                l=l,
                q=self._leaf_q(w, d, l, stm_white),
                m=0.0,
            )
        planes = encode_position(board, history)
        key = board.fen()
        output = self.backend.evaluate_cached(key, planes, self.params.policy_softmax_temp)
        output = NNOutput(
            policy=output.policy,
            logits=output.logits,
            w=output.w,
            d=output.d,
            l=output.l,
            q=self._leaf_q(output.w, output.d, output.l, stm_white),
            m=output.m,
        )
        self._expand(node, board, output, add_noise=add_noise)
        return output

    def _evaluate_batch(self, tasks: list[_LeafTask]) -> None:
        pending: list[tuple[_LeafTask, np.ndarray]] = []
        for task in tasks:
            existing = task.path[-1].edge.child if task.path else None
            if existing is not None:
                task.node = existing
                continue
            if self._is_terminal(task.board):
                w, d, l = self._terminal_wdl(task.board)
                stm_white = task.board.turn == chess.WHITE
                task.node.terminal = True
                child = self._annotate(
                    Node(n=1, w=w, d=d, l=l, terminal=True, stm_white=stm_white),
                    stm_white,
                )
                if task.path:
                    task.path[-1].edge.child = child
                task.node = child
                continue
            pending.append((task, encode_position(task.board, task.history)))
        if not pending:
            return
        planes = np.stack([item[1] for item in pending], axis=0)
        outputs = self.backend.evaluate_batch(planes, self.params.policy_softmax_temp)
        for (task, _), output in zip(pending, outputs, strict=True):
            child = self._annotate(Node(), task.board.turn == chess.WHITE)
            self._expand(child, task.board, output, add_noise=False)
            child.n = 1
            child.w, child.d, child.l, child.m_sum = output.w, output.d, output.l, output.m
            if task.path:
                task.path[-1].edge.child = child
            task.node = child

    def _expand(self, node: Node, board: chess.Board, output: NNOutput, add_noise: bool) -> None:
        legal = list(board.legal_moves)
        stm_white = board.turn == chess.WHITE
        node.stm_white = stm_white
        node.q_mode = self.params.q_mode
        if not legal:
            node.terminal = True
            return
        we_are_black = board.turn == chess.BLACK
        priors = []
        for move in legal:
            idx = move_to_policy_index(move, we_are_black)
            priors.append(float(output.policy[idx]))
        arr = np.asarray(priors, dtype=np.float64)
        total = arr.sum()
        if total <= 0:
            arr.fill(1.0 / len(arr))
        else:
            arr /= total
        if add_noise and self.params.dirichlet_epsilon > 0:
            noise = np.random.dirichlet([self.params.dirichlet_alpha] * len(arr))
            arr = (1.0 - self.params.dirichlet_epsilon) * arr + self.params.dirichlet_epsilon * noise
        node.edges = [
            Edge(
                move=move,
                prior=float(p),
                stm_white=stm_white,
                q_mode=self.params.q_mode,
            )
            for move, p in zip(legal, arr, strict=True)
        ]

    def _backup(self, task: _LeafTask) -> None:
        denom = max(task.node.n, 1)
        w = task.node.w / denom
        d = task.node.d / denom
        l = task.node.l / denom
        m = task.node.m_sum / denom
        # Leaf WDL is from the leaf side-to-move. Each step up the tree flips W/L.
        for step in reversed(task.path):
            w, l = l, w
            m = m + 1.0
            edge = step.edge
            edge.n_in_flight = max(0, edge.n_in_flight - 1)
            edge.n += 1
            edge.w += w
            edge.d += d
            edge.l += l
            edge.m_sum += m
            parent = step.parent
            parent.n += 1
            parent.w += w
            parent.d += d
            parent.l += l
            parent.m_sum += m

    def _undo_virtuals(self, tasks: list[_LeafTask]) -> None:
        for task in tasks:
            for step in task.path:
                step.edge.n_in_flight = max(0, step.edge.n_in_flight - 1)

    def _select_root_move(self, root: Node, we_are_black: bool, temperature: float) -> chess.Move:
        if not root.edges:
            return chess.Move.null()
        if temperature <= 1e-6:
            return root.best_edge().move
        visits = np.array([edge.n for edge in root.edges], dtype=np.float64)
        if visits.sum() <= 0:
            return root.edges[0].move
        logits = np.log(np.maximum(visits, 1e-8)) / temperature
        logits -= logits.max()
        probs = np.exp(logits)
        probs /= probs.sum()
        idx = int(np.random.choice(len(root.edges), p=probs))
        return root.edges[idx].move

    def _visit_policy(self, root: Node, we_are_black: bool) -> np.ndarray:
        dist = np.zeros(POLICY_SIZE, dtype=np.float32)
        total = sum(edge.n for edge in root.edges)
        if total <= 0:
            return dist
        for edge in root.edges:
            dist[move_to_policy_index(edge.move, we_are_black)] = edge.n / total
        return dist

    def _principal_variation(self, root: Node, limit: int = 10) -> list[chess.Move]:
        pv: list[chess.Move] = []
        node = root
        while node.edges and len(pv) < limit:
            edge = root.best_edge() if node is root else max(node.edges, key=lambda e: (e.n, e.q))
            pv.append(edge.move)
            if edge.child is None:
                break
            node = edge.child
        return pv

    def _is_terminal(self, board: chess.Board, *, at_root: bool = False) -> bool:
        if board.is_game_over(claim_draw=True):
            return True
        # Two-fold is an Lc0 search shortcut for descendants, not a FIDE end.
        # The root must still pick a legal move so the real game can reach 3-fold.
        if at_root:
            return False
        if self.params.two_fold_draws and board.is_repetition(2):
            return True
        return False

    def _terminal_wdl(self, board: chess.Board) -> tuple[float, float, float]:
        if board.is_checkmate():
            return (0.0, 0.0, 1.0)
        return (0.0, 1.0, 0.0)


def _history_from_board(board: chess.Board) -> list[chess.Board]:
    rewind = board.copy(stack=True)
    positions = [rewind.copy(stack=False)]
    while rewind.move_stack:
        rewind.pop()
        positions.append(rewind.copy(stack=False))
    positions.reverse()
    return positions
