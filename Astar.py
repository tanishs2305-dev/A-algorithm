import heapq
import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional

Graph = Dict[str, Dict[str, float]]   # graph[u][v] = cost of edge u -> v
Heuristic = Dict[str, float]          # h[node]     = estimated cost to goal


@dataclass(eq=False)
class TreeNode:
    """One node of the search tree (the same state can appear several times)."""
    state: str
    g: float
    h: float
    parent: Optional["TreeNode"] = None
    children: List["TreeNode"] = field(default_factory=list)
    order: Optional[int] = None       # expansion number; None = never expanded
    skipped: bool = False             # popped, but a cheaper copy was already expanded

    @property
    def f(self) -> float:
        return self.g + self.h


@dataclass
class Step:
    number: int
    expanded: TreeNode
    open_list: List[TreeNode]         # frontier after this expansion, sorted by f


@dataclass
class SearchResult:
    found: bool
    path: List[str]
    cost: float
    root: TreeNode
    goal_node: Optional[TreeNode]
    steps: List[Step]


def astar(graph: Graph, h: Heuristic, start: str, goal: str) -> SearchResult:
    root = TreeNode(start, 0.0, h[start])
    counter = 0                                   # FIFO tie-breaker for equal f
    frontier = [(root.f, counter, root)]
    best_g: Dict[str, float] = {start: 0.0}       # cheapest g generated per state
    closed: Dict[str, float] = {}                 # state -> g when it was expanded
    steps: List[Step] = []
    order = 0

    def open_list() -> List[TreeNode]:
        live = [(f, c, n) for f, c, n in frontier
                if n.g <= best_g[n.state]
                and not (n.state in closed and n.g >= closed[n.state])]
        return [n for _, _, n in sorted(live, key=lambda t: (t[0], t[1]))]

    while frontier:
        _, _, node = heapq.heappop(frontier)

        if node.state in closed and node.g >= closed[node.state]:
            node.skipped = True
            continue

        closed[node.state] = node.g               # (re)expand only if cheaper
        order += 1
        node.order = order

        if node.state == goal:
            steps.append(Step(order, node, open_list()))
            return SearchResult(True, _path(node), node.g, root, node, steps)

        for nxt, cost in graph.get(node.state, {}).items():
            g = node.g + cost
            if g < best_g.get(nxt, math.inf):     # only keep improvements
                best_g[nxt] = g
                child = TreeNode(nxt, g, h[nxt], parent=node)
                node.children.append(child)
                counter += 1
                heapq.heappush(frontier, (child.f, counter, child))

        steps.append(Step(order, node, open_list()))

    return SearchResult(False, [], 0.0, root, None, steps)


def _path(node: TreeNode) -> List[str]:
    path = []
    while node:
        path.append(node.state)
        node = node.parent
    return path[::-1]


def overestimating_nodes(graph: Graph, h: Heuristic, goal: str) -> List[str]:
    """Nodes whose h is larger than the true cheapest cost to the goal.

    A* is only guaranteed optimal when h never overestimates.
    """
    reverse: Graph = {n: {} for n in h}
    for u, nbrs in graph.items():
        for v, w in nbrs.items():
            reverse[v][u] = min(w, reverse[v].get(u, math.inf))

    dist = {goal: 0.0}
    heap = [(0.0, goal)]
    while heap:
        d, u = heapq.heappop(heap)
        if d > dist.get(u, math.inf):
            continue
        for v, w in reverse[u].items():
            if d + w < dist.get(v, math.inf):
                dist[v] = d + w
                heapq.heappush(heap, (d + w, v))

    return [n for n, d in dist.items() if h[n] > d + 1e-9]