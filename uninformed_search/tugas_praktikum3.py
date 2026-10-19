import time
from collections import deque
import heapq

# ---------------------------------------------------------
# GRAPH BASE: PETA SUMBAR
# ---------------------------------------------------------
PETA_SUMBAR = {
    "Padang":         {"Pariaman": 56, "Padang Panjang": 72, "Solok": 64, "Painan": 77},
    "Pariaman":       {"Padang": 56, "Lubuk Basung": 48, "Padang Panjang": 50},
    "Painan":         {"Padang": 77},
    "Lubuk Basung":   {"Pariaman": 48, "Bukittinggi": 56},
    "Padang Panjang": {"Padang": 72, "Pariaman": 50, "Batusangkar": 36, "Bukittinggi": 19},
    "Bukittinggi":    {"Padang Panjang": 19, "Lubuk Basung": 56, "Payakumbuh": 33},
    "Batusangkar":    {"Padang Panjang": 36, "Solok": 55, "Sawahlunto": 46, "Payakumbuh": 42},
    "Solok":          {"Padang": 64, "Batusangkar": 55, "Sawahlunto": 36},
    "Sawahlunto":     {"Solok": 36, "Batusangkar": 46},
    "Payakumbuh":     {"Bukittinggi": 33, "Batusangkar": 42},
}

class Problem:
    def __init__(self, initial, goal, graph):
        self.initial, self.goal, self.graph = initial, goal, graph

    def actions(self, s):
        return list(self.graph[s])

    def result(self, s, a):
        return a

    def action_cost(self, s, a, s2):
        return self.graph[s][s2]

    def is_goal(self, s):
        return s == self.goal

class Node:
    def __init__(self, state, parent=None, action=None, path_cost=0):
        self.state, self.parent = state, parent
        self.action, self.path_cost = action, path_cost
        self.depth = 0 if parent is None else parent.depth + 1

    def __lt__(self, other):
        return self.state < other.state

def expand(problem, node):
    s = node.state
    for a in problem.actions(s):
        s2 = problem.result(s, a)
        cost = node.path_cost + problem.action_cost(s, a, s2)
        yield Node(s2, node, a, cost)

def path_states(node):
    path = []
    while node is not None:
        path.append(node.state)
        node = node.parent
    return path[::-1]

def is_cycle(node):
    s, p = node.state, node.parent
    while p is not None:
        if p.state == s:
            return True
        p = p.parent
    return False

# ---------------------------------------------------------
# ALGORITMA DENGAN MAX FRONTIER TRACKING
# ---------------------------------------------------------
def breadth_first_search(problem, stats):
    node = Node(problem.initial)
    if problem.is_goal(node.state):
        return node
    frontier = deque([node])
    reached = {problem.initial}
    stats["max_frontier"] = max(stats.get("max_frontier", 0), len(frontier))
    while frontier:
        node = frontier.popleft()
        stats["expanded"] += 1
        for child in expand(problem, node):
            s = child.state
            if problem.is_goal(s):
                return child
            if s not in reached:
                reached.add(s)
                frontier.append(child)
        stats["max_frontier"] = max(stats.get("max_frontier", 0), len(frontier))
    return None

def uniform_cost_search(problem, stats):
    node = Node(problem.initial)
    frontier = [(0, node)]
    reached = {problem.initial: node}
    stats["max_frontier"] = max(stats.get("max_frontier", 0), len(frontier))
    while frontier:
        g, node = heapq.heappop(frontier)
        if g > reached[node.state].path_cost:
            continue
        if problem.is_goal(node.state):
            return node
        stats["expanded"] += 1
        for child in expand(problem, node):
            s = child.state
            if s not in reached or child.path_cost < reached[s].path_cost:
                reached[s] = child
                heapq.heappush(frontier, (child.path_cost, child))
        stats["max_frontier"] = max(stats.get("max_frontier", 0), len(frontier))
    return None

def depth_first_search_tree(problem, stats):
    frontier = [Node(problem.initial)]
    stats["max_frontier"] = max(stats.get("max_frontier", 0), len(frontier))
    while frontier:
        node = frontier.pop()
        if problem.is_goal(node.state):
            return node
        stats["expanded"] += 1
        children = [c for c in expand(problem, node) if not is_cycle(c)]
        frontier.extend(reversed(children))
        stats["max_frontier"] = max(stats.get("max_frontier", 0), len(frontier))
    return None

def depth_first_search_graph(problem, stats):
    frontier = [Node(problem.initial)]
    reached = set()
    stats["max_frontier"] = max(stats.get("max_frontier", 0), len(frontier))
    while frontier:
        node = frontier.pop()
        if problem.is_goal(node.state):
            return node
        if node.state not in reached:
            reached.add(node.state)
            stats["expanded"] += 1
            for child in expand(problem, node):
                if child.state not in reached:
                    frontier.append(child)
        stats["max_frontier"] = max(stats.get("max_frontier", 0), len(frontier))
    return None

CUTOFF = "cutoff"
def depth_limited_search(problem, limit, stats):
    frontier = [Node(problem.initial)]
    result = None
    stats["max_frontier"] = max(stats.get("max_frontier", 0), len(frontier))
    while frontier:
        node = frontier.pop()
        if problem.is_goal(node.state):
            return node
        if node.depth >= limit:
            result = CUTOFF
        elif not is_cycle(node):
            stats["expanded"] += 1
            children = list(expand(problem, node))
            frontier.extend(reversed(children))
        stats["max_frontier"] = max(stats.get("max_frontier", 0), len(frontier))
    return result

def iterative_deepening_search(problem, stats):
    depth = 0
    while True:
        result = depth_limited_search(problem, depth, stats)
        if result != CUTOFF:
            return result
        depth += 1

# ---------------------------------------------------------
# BIDIRECTIONAL BFS
# ---------------------------------------------------------
def bidirectional_bfs(problem, stats):
    if problem.initial == problem.goal:
        return Node(problem.initial)

    node_f = Node(problem.initial)
    node_b = Node(problem.goal)

    frontier_f = deque([node_f])
    frontier_b = deque([node_b])

    reached_f = {problem.initial: node_f}
    reached_b = {problem.goal: node_b}

    stats["max_frontier"] = max(stats.get("max_frontier", 0), len(frontier_f) + len(frontier_b))

    while frontier_f and frontier_b:
        # Expand Forward
        if frontier_f:
            curr_f = frontier_f.popleft()
            stats["expanded"] += 1
            for child in expand(problem, curr_f):
                s = child.state
                if s in reached_b:
                    return _combine_paths(child, reached_b[s])
                if s not in reached_f:
                    reached_f[s] = child
                    frontier_f.append(child)

        # Expand Backward
        if frontier_b:
            curr_b = frontier_b.popleft()
            stats["expanded"] += 1
            for child in expand(problem, curr_b):
                s = child.state
                if s in reached_f:
                    return _combine_paths(reached_f[s], child)
                if s not in reached_b:
                    reached_b[s] = child
                    frontier_b.append(child)

        stats["max_frontier"] = max(stats.get("max_frontier", 0), len(frontier_f) + len(frontier_b))
    return None

def _combine_paths(node_f, node_b):
    path_f = path_states(node_f)
    path_b = path_states(node_b)[::-1]
    full_path = path_f + path_b[1:]
    
    total_cost = node_f.path_cost + node_b.path_cost
    
    res = Node(full_path[-1], path_cost=total_cost)
    res.custom_path = full_path
    res.depth = len(full_path) - 1
    return res

# ---------------------------------------------------------
# 8-PUZZLE CLASS
# ---------------------------------------------------------
class EightPuzzle(Problem):
    GOAL = (1, 2, 3, 4, 5, 6, 7, 8, 0)
    GESER = {"Atas": -3, "Bawah": 3, "Kiri": -1, "Kanan": 1}

    def __init__(self, initial):
        self.initial, self.goal = tuple(initial), self.GOAL

    def actions(self, s):
        i = s.index(0)
        aksi = []
        if i >= 3: aksi.append("Atas")
        if i <= 5: aksi.append("Bawah")
        if i % 3 > 0: aksi.append("Kiri")
        if i % 3 < 2: aksi.append("Kanan")
        return aksi

    def result(self, s, a):
        i = s.index(0)
        j = i + self.GESER[a]
        s = list(s)
        s[i], s[j] = s[j], s[i]
        return tuple(s)

    def action_cost(self, s, a, s2):
        return 1

print("1. Painan -> Lubuk Basung")
masalah1 = Problem("Painan", "Lubuk Basung", PETA_SUMBAR)
for nama, f in [("BFS", breadth_first_search),
                ("UCS", uniform_cost_search),
                ("DFS", depth_first_search_tree),
                ("IDS", iterative_deepening_search)]:
    stats = {"expanded": 0, "max_frontier": 0}
    t0 = time.perf_counter()
    res = f(masalah1, stats)
    dt = (time.perf_counter() - t0) * 1000
    rute = " -> ".join(path_states(res))
    print(f"{nama:<5} | {rute}")
    print(f"      | langkah={res.depth}, biaya={res.path_cost} km, ekspansi={stats['expanded']}, max_frontier={stats['max_frontier']}, waktu={dt:.3f} ms")

print("\n2. Ubah urutan tetangga Padang Panjang (Bukittinggi sebelum Batusangkar)")
PETA_SUMBAR_MOD = {
    "Padang":         {"Pariaman": 56, "Padang Panjang": 72, "Solok": 64, "Painan": 77},
    "Pariaman":       {"Padang": 56, "Lubuk Basung": 48, "Padang Panjang": 50},
    "Painan":         {"Padang": 77},
    "Lubuk Basung":   {"Pariaman": 48, "Bukittinggi": 56},
    "Padang Panjang": {"Padang": 72, "Pariaman": 50, "Bukittinggi": 19, "Batusangkar": 36},
    "Bukittinggi":    {"Padang Panjang": 19, "Lubuk Basung": 56, "Payakumbuh": 33},
    "Batusangkar":    {"Padang Panjang": 36, "Solok": 55, "Sawahlunto": 46, "Payakumbuh": 42},
    "Solok":          {"Padang": 64, "Batusangkar": 55, "Sawahlunto": 36},
    "Sawahlunto":     {"Solok": 36, "Batusangkar": 46},
    "Payakumbuh":     {"Bukittinggi": 33, "Batusangkar": 42},
}

masalah_orig = Problem("Padang", "Payakumbuh", PETA_SUMBAR)
masalah_mod = Problem("Padang", "Payakumbuh", PETA_SUMBAR_MOD)

print("-- Sebelum Perubahan Urutan (Original - Padang -> Payakumbuh):")
for nama, f in [("BFS", breadth_first_search), ("DFS", depth_first_search_tree)]:
    stats = {"expanded": 0, "max_frontier": 0}
    res = f(masalah_orig, stats)
    print(f"{nama:<5} | {' -> '.join(path_states(res))} (ekspansi={stats['expanded']}, biaya={res.path_cost})")

print("-- Sesudah Perubahan Urutan (Bukittinggi dulu - Padang -> Payakumbuh):")
for nama, f in [("BFS", breadth_first_search), ("DFS", depth_first_search_tree)]:
    stats = {"expanded": 0, "max_frontier": 0}
    res = f(masalah_mod, stats)
    print(f"{nama:<5} | {' -> '.join(path_states(res))} (ekspansi={stats['expanded']}, biaya={res.path_cost})")

print("\n3. Pengukuran Max Frontier pada 8-Puzzle 'Berat'")
kasus_berat = (1, 8, 3, 5, 0, 6, 4, 2, 7)
puz_prob = EightPuzzle(kasus_berat)

for nama, f in [("BFS", breadth_first_search), ("IDS", iterative_deepening_search)]:
    stats = {"expanded": 0, "max_frontier": 0}
    t0 = time.perf_counter()
    res = f(puz_prob, stats)
    dt = (time.perf_counter() - t0) * 1000
    print(f"{nama:<5} | langkah={res.depth}, ekspansi={stats['expanded']}, max_frontier={stats['max_frontier']}, waktu={dt:.3f} ms")

print("\n4. DFS Tree-like vs DFS Graph-like (reached table)")
print("-- Pada Peta Sumbar (Padang -> Payakumbuh):")
stats_tree = {"expanded": 0, "max_frontier": 0}
res_tree = depth_first_search_tree(masalah_orig, stats_tree)
print(f"DFS Tree  | {' -> '.join(path_states(res_tree))} | ekspansi={stats_tree['expanded']}, max_frontier={stats_tree['max_frontier']}")

stats_graph = {"expanded": 0, "max_frontier": 0}
res_graph = depth_first_search_graph(masalah_orig, stats_graph)
print(f"DFS Graph | {' -> '.join(path_states(res_graph))} | ekspansi={stats_graph['expanded']}, max_frontier={stats_graph['max_frontier']}")

print("-- Pada 8-Puzzle kasus 'sedang' (4, 1, 3, 7, 2, 6, 0, 5, 8):")
puz_sedang = EightPuzzle((4, 1, 3, 7, 2, 6, 0, 5, 8))
stats_t = {"expanded": 0, "max_frontier": 0}
res_t = depth_first_search_tree(puz_sedang, stats_t)
print(f"DFS Tree  | langkah={res_t.depth if res_t else 'N/A'}, ekspansi={stats_t['expanded']}, max_frontier={stats_t['max_frontier']}")

stats_g = {"expanded": 0, "max_frontier": 0}
res_g = depth_first_search_graph(puz_sedang, stats_g)
print(f"DFS Graph | langkah={res_g.depth if res_g else 'N/A'}, ekspansi={stats_g['expanded']}, max_frontier={stats_g['max_frontier']}")

print("\n5. Pengayaan: Bidirectional BFS pada Peta Sumbar (Padang -> Payakumbuh)")
stats_bfs = {"expanded": 0, "max_frontier": 0}
res_bfs = breadth_first_search(masalah_orig, stats_bfs)
print(f"BFS Standard       | {' -> '.join(path_states(res_bfs))} | ekspansi={stats_bfs['expanded']}, max_frontier={stats_bfs['max_frontier']}")

stats_bibfs = {"expanded": 0, "max_frontier": 0}
res_bibfs = bidirectional_bfs(masalah_orig, stats_bibfs)
rute_bi = " -> ".join(getattr(res_bibfs, 'custom_path', path_states(res_bibfs)))
print(f"Bidirectional BFS  | {rute_bi} | ekspansi={stats_bibfs['expanded']}, max_frontier={stats_bibfs['max_frontier']}, biaya={res_bibfs.path_cost} km")


print("\n6. Pengayaan: Graph Kampus PNP (Politeknik Negeri Padang)")
PETA_PNP = {
    "Gerbang Utama": {"Rektorat": 150, "Parkiran Utama": 100},
    "Parkiran Utama": {"Gerbang Utama": 100, "PKM": 120, "Gedung C": 180},
    "Rektorat": {"Gerbang Utama": 150, "Perpustakaan": 200, "Masjid PNP": 130},
    "Perpustakaan": {"Rektorat": 200, "Jurusan TI": 110, "Kantin Pusat": 140},
    "Masjid PNP": {"Rektorat": 130, "Gedung C": 90, "Jurusan Sipil": 160},
    "PKM": {"Parkiran Utama": 120, "Kantin Pusat": 80, "Jurusan Elektro": 150},
    "Kantin Pusat": {"Perpustakaan": 140, "PKM": 80, "Jurusan Mesin": 100},
    "Gedung C": {"Parkiran Utama": 180, "Masjid PNP": 90, "Jurusan Sipil": 120},
    "Jurusan TI": {"Perpustakaan": 110, "Jurusan Elektro": 70, "Lab Komputer": 50},
    "Jurusan Elektro": {"Jurusan TI": 70, "PKM": 150, "Jurusan Mesin": 90},
    "Jurusan Mesin": {"Kantin Pusat": 100, "Jurusan Elektro": 90, "Jurusan Sipil": 130},
    "Jurusan Sipil": {"Gedung C": 120, "Masjid PNP": 160, "Jurusan Mesin": 130},
    "Lab Komputer": {"Jurusan TI": 50}
}

masalah_pnp = Problem("Gerbang Utama", "Jurusan TI", PETA_PNP)
stats_pnp = {"expanded": 0, "max_frontier": 0}
res_pnp = uniform_cost_search(masalah_pnp, stats_pnp)
print(f"UCS Rute PNP (Gerbang Utama -> Jurusan TI):")
print(f"Rute  : {' -> '.join(path_states(res_pnp))}")
print(f"Jarak : {res_pnp.path_cost} meter")
print(f"Stats : ekspansi={stats_pnp['expanded']}, max_frontier={stats_pnp['max_frontier']}")
