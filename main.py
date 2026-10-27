# import matplotlib
import re
from typing import List, Tuple

from display import (print_result, print_steps, print_tree,
                     save_graph_image, save_tree_image)
from Astar import Graph, Heuristic, astar, overestimating_nodes

Edge = Tuple[str, str, float]


def natural_key(name: str):
    """Sort A, B, C... alphabetically and n2 before n10."""
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", name)]


def read_edges() -> List[Edge]:
    print("Enter each edge as:  FROM TO COST      (example:  A B 1)")
    print("Node names can be any word without spaces. Empty line = finished.")
    edges: List[Edge] = []
    while True:
        parts = input("Edge: ").split()
        if not parts:
            if edges:
                return edges
            print("  Enter at least one edge.")
            continue
        try:
            u, v, w = parts[0], parts[1], float(parts[2])
            if w < 0:
                raise ValueError
            edges.append((u, v, w))
        except (IndexError, ValueError):
            print("  Format: FROM TO COST   (cost must be a number >= 0)")


def read_heuristics(nodes: List[str]) -> Heuristic:
    print("\nEnter the heuristic value h(n) of every node.")
    h: Heuristic = {}
    for n in nodes:
        while True:
            try:
                value = float(input(f"  h({n}) = "))
                if value < 0:
                    raise ValueError
                h[n] = value
                break
            except ValueError:
                print("  Enter a number >= 0.")
    return h


def ask_node(prompt: str, nodes: List[str]) -> str:
    while True:
        text = input(prompt).strip()
        if text in nodes:
            return text
        match = [n for n in nodes if n.lower() == text.lower()]
        if len(match) == 1:
            return match[0]
        print(f"  Choose one of: {', '.join(nodes)}")


def pick_goal(h: Heuristic, nodes: List[str]) -> str:
    """The goal is the node whose heuristic is 0 (asks only if that is unclear)."""
    zero = [n for n in nodes if h[n] == 0]
    if len(zero) == 1:
        print(f"Goal node : {zero[0]}   (detected automatically, since h({zero[0]}) = 0)")
        return zero[0]
    if len(zero) > 1:
        print(f"More than one node has h = 0: {', '.join(zero)}")
        while True:
            goal = ask_node("Which one is the goal? ", nodes)
            if goal in zero:
                return goal
            print(f"  Choose one of: {', '.join(zero)}")
    print("No node has h = 0.")
    return ask_node("Goal node : ", nodes)


def build_graph(nodes: List[str], edges: List[Edge], directed: bool) -> Graph:
    graph: Graph = {n: {} for n in nodes}
    for u, v, w in edges:
        if u == v:
            continue
        for a, b in ([(u, v)] if directed else [(u, v), (v, u)]):
            graph[a][b] = min(w, graph[a].get(b, float("inf")))
    return graph


def main() -> None:
    print("=== A* Search ===\n")
    directed = input("Is the graph directed? (y/n): ").strip().lower().startswith("y")
    edges = read_edges()

    nodes = sorted({n for u, v, _ in edges for n in (u, v)}, key=natural_key)
    h = read_heuristics(nodes)
    print(f"\nNodes: {', '.join(nodes)}")
    start = ask_node("Start node: ", nodes)
    goal = pick_goal(h, nodes)

    graph = build_graph(nodes, edges, directed)

    if h[goal] != 0:
        print(f"\nNote: h({goal}) is not 0; a goal node normally has h = 0.")
    bad = overestimating_nodes(graph, h, goal)
    if bad:
        print("Warning: h overestimates the true cost for "
              f"{', '.join(bad)}, so the result may not be optimal.")

    result = astar(graph, h, start, goal)

    print("\n--- Step by step ---")
    print_steps(result)
    print("\n--- Search tree ---")
    print_tree(result.root)
    print_result(result)

    made_graph = save_graph_image(edges, directed, h, result.path, start, "graph.png")
    made_tree = save_tree_image(result, "search_tree.png")
    if made_graph and made_tree:
        print("\nPictures saved: graph.png and search_tree.png")
    else:
        print("\n(Install matplotlib to also get the pictures: pip install matplotlib)")


if __name__ == "__main__":
    main()