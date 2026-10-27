from collections import deque
from typing import Dict, List, Tuple

from Astar import Heuristic, SearchResult, TreeNode

Edge = Tuple[str, str, float]


# ---------------------------------------------------------------- text output
def _label(n: TreeNode) -> str:
    text = f"{n.state}  (g={n.g:g}, h={n.h:g}, f={n.f:g})"
    if n.order is not None:
        return text + f"   [expanded #{n.order}]"
    if n.skipped:
        return text + "   [skipped: cheaper copy already expanded]"
    return text + "   [not expanded]"


def print_tree(node: TreeNode, prefix: str = "", is_last: bool = True,
               is_root: bool = True) -> None:
    print(_label(node) if is_root
          else prefix + ("└── " if is_last else "├── ") + _label(node))
    child_prefix = "" if is_root else prefix + ("    " if is_last else "│   ")
    for i, child in enumerate(node.children):
        print_tree(child, child_prefix, i == len(node.children) - 1, False)


def print_steps(result: SearchResult) -> None:
    print(f"{'Step':<6}{'Expanded':<10}{'f=g+h':<8}Open list (node:f)")
    for s in result.steps:
        open_txt = ", ".join(f"{n.state}:{n.f:g}" for n in s.open_list) or "-"
        print(f"{s.number:<6}{s.expanded.state:<10}{s.expanded.f:<8g}{open_txt}")


def print_result(result: SearchResult) -> None:
    if not result.found:
        print("\nNo path exists from the start to the goal.")
    else:
        print("\nOptimal path :", " -> ".join(result.path))
        print("Total cost   :", f"{result.cost:g}")


# --------------------------------------------------------------- picture 1: graph
def _layout(edges: List[Edge], directed: bool, nodes: List[str], start: str):
    adj: Dict[str, List[str]] = {n: [] for n in nodes}
    for u, v, _ in edges:
        adj[u].append(v)
        if not directed:
            adj[v].append(u)

    depth, queue = {start: 0}, deque([start])      # BFS layers from the start
    while queue:
        u = queue.popleft()
        for v in adj[u]:
            if v not in depth:
                depth[v] = depth[u] + 1
                queue.append(v)
    extra = max(depth.values()) + 1
    for n in nodes:                                # unreachable nodes: last column
        depth.setdefault(n, extra)

    layers: Dict[int, List[str]] = {}
    for n in nodes:
        layers.setdefault(depth[n], []).append(n)

    y: Dict[str, float] = {}
    for d in sorted(layers):                       # order each column to cut crossings
        group = layers[d]
        def bary(n: str) -> float:
            ps = [y[p] for p in nodes if p in y and n in adj[p] and depth[p] < d]
            return sum(ps) / len(ps) if ps else 0.0
        group.sort(key=lambda n: -bary(n))
        for i, n in enumerate(group):
            y[n] = ((len(group) - 1) / 2 - i) * 1.9
    return {n: (depth[n] * 2.6, y[n]) for n in nodes}, len(layers)


def save_graph_image(edges: List[Edge], directed: bool, h: Heuristic,
                     path: List[str], start: str, filename: str) -> bool:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return False

    nodes = list(h)
    pos, ncols = _layout(edges, directed, nodes, start)
    on_path = set(zip(path, path[1:]))
    pairs = {(u, v) for u, v, _ in edges}

    fig, (ax, tab) = plt.subplots(
        1, 2, figsize=(max(9, 2.6 * ncols + 2.5), 5.5),
        gridspec_kw={"width_ratios": [ncols * 2.6, 1.2]})

    for u, v, w in edges:
        if u == v:
            continue
        hot = (u, v) in on_path or (not directed and (v, u) in on_path)
        color, lw = ("#d62728", 2.8) if hot else ("#555555", 1.3)
        rad = 0.22 if directed and (v, u) in pairs else 0.0
        ax.annotate("", xy=pos[v], xytext=pos[u], zorder=1, arrowprops=dict(
            arrowstyle="-|>" if directed else "-", color=color, lw=lw,
            shrinkA=24, shrinkB=24, connectionstyle=f"arc3,rad={rad}"))
        (x1, y1), (x2, y2) = pos[u], pos[v]
        mx = (x1 + x2) / 2 + 0.5 * rad * (y2 - y1)
        my = (y1 + y2) / 2 - 0.5 * rad * (x2 - x1)
        ax.text(mx, my, f"{w:g}", ha="center", va="center", fontsize=11,
                fontweight="bold", color=color, zorder=2,
                bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none"))

    for n, (x, yy) in pos.items():
        hot = n in path
        ax.scatter([x], [yy], s=1500, zorder=3, linewidth=2, clip_on=False,
                   facecolor="#ffd9d9" if hot else "white",
                   edgecolor="#d62728" if hot else "black")
        ax.text(x, yy, n, ha="center", va="center", fontsize=13, zorder=4)

    xs = [p[0] for p in pos.values()]
    ys = [p[1] for p in pos.values()]
    ax.set_xlim(min(xs) - 1.0, max(xs) + 1.0)
    ax.set_ylim(min(ys) - 1.2, max(ys) + 1.2)
    ax.set_aspect("equal")
    ax.set_axis_off()
    ax.set_title("Graph" + (" (red = optimal path)" if path else ""), pad=14)

    cells = [[n, f"{h[n]:g}"] for n in nodes]
    table = tab.table(cellText=cells, colLabels=["Node", "h(n)"],
                      cellLoc="center", loc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1, 1.8)
    tab.set_axis_off()

    fig.savefig(filename, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return True


# ------------------------------------------------------- picture 2: search tree
def save_tree_image(result: SearchResult, filename: str) -> bool:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.patches import Patch
    except ImportError:
        return False

    pos: Dict[int, Tuple[float, float]] = {}
    leaf = [0]

    def place(node: TreeNode, depth: int) -> float:
        if node.children:
            xs = [place(c, depth + 1) for c in node.children]
            x = sum(xs) / len(xs)
        else:
            x, leaf[0] = float(leaf[0]), leaf[0] + 1
        pos[id(node)] = (x, -float(depth))
        return x

    place(result.root, 0)

    best = set()
    n = result.goal_node
    while n:
        best.add(id(n))
        n = n.parent

    max_depth = int(-min(y for _, y in pos.values()))
    fig, ax = plt.subplots(figsize=(max(7, leaf[0] * 1.9), max(4, (max_depth + 1) * 1.9)))

    def draw(node: TreeNode) -> None:
        x, y = pos[id(node)]
        for c in node.children:
            cx, cy = pos[id(c)]
            hot = id(c) in best
            ax.plot([x, cx], [y, cy], color="#d62728" if hot else "#777777",
                    lw=2.6 if hot else 1.2, zorder=1)
            ax.text((x + cx) / 2, (y + cy) / 2, f"{c.g - node.g:g}", fontsize=10,
                    ha="center", va="center", zorder=2,
                    color="#d62728" if hot else "#333333", fontweight="bold",
                    bbox=dict(boxstyle="round,pad=0.12", fc="white", ec="none"))
            draw(c)

        if id(node) in best:
            fc, ec, ls = "#ffd9d9", "#d62728", "-"
        elif node.order is not None:
            fc, ec, ls = "#dbeafe", "#1f4e8c", "-"
        elif node.skipped:
            fc, ec, ls = "#eeeeee", "#888888", "--"
        else:
            fc, ec, ls = "white", "#888888", "--"
        head = f"#{node.order}  {node.state}" if node.order else node.state
        ax.text(x, y, f"{head}\ng={node.g:g}  h={node.h:g}\nf={node.f:g}",
                ha="center", va="center", fontsize=9, zorder=3,
                bbox=dict(boxstyle="round,pad=0.4", fc=fc, ec=ec, ls=ls, lw=1.8))

    draw(result.root)
    ax.legend(handles=[
        Patch(fc="#ffd9d9", ec="#d62728", label="on optimal path"),
        Patch(fc="#dbeafe", ec="#1f4e8c", label="expanded (#= order)"),
        Patch(fc="white", ec="#888888", ls="--", label="generated, not expanded"),
    ], loc="upper center", bbox_to_anchor=(0.5, 0.0), ncol=3,
        fontsize=9, frameon=False)
    ax.margins(0.12)
    ax.set_axis_off()
    ax.set_title("A* search tree (edge labels = step cost)")
    fig.savefig(filename, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return True