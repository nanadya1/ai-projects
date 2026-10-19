# pencarian_buta.py
# Praktikum 3: Implementasi Algoritma Pencarian Buta (Uninformed Search)

from collections import deque
import heapq
import time

# ---- Langkah 1: Representasi Masalah ----
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
    """Masalah pencarian: initial state, goal, ACTIONS, RESULT, ACTION-COST."""
    def __init__(self, initial, goal, graph):
        self.initial, self.goal, self.graph = initial, goal, graph

    def actions(self, s):            # aksi = pindah ke kota tetangga
        return list(self.graph[s])

    def result(self, s, a):          # transition model
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

    def __lt__(self, other):          # dibutuhkan heapq saat biaya sama
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

# ---- Langkah 2: Breadth-First Search ----
def breadth_first_search(problem, stats):
    node = Node(problem.initial)
    if problem.is_goal(node.state):
        return node
    frontier = deque([node])                 # FIFO queue
    reached = {problem.initial}
    while frontier:
        node = frontier.popleft()
        stats["expanded"] += 1
        for child in expand(problem, node):
            s = child.state
            if problem.is_goal(s):           # early goal test
                return child
            if s not in reached:
                reached.add(s)
                frontier.append(child)
    return None

# ---- Langkah 3: Uniform-Cost Search ----
def uniform_cost_search(problem, stats):
    node = Node(problem.initial)
    frontier = [(0, node)]                   # priority queue (heap)
    reached = {problem.initial: node}
    while frontier:
        g, node = heapq.heappop(frontier)
        if g > reached[node.state].path_cost:
            continue                         # entri usang, lewati
        if problem.is_goal(node.state):      # late goal test
            return node
        stats["expanded"] += 1
        for child in expand(problem, node):
            s = child.state
            if s not in reached or child.path_cost < reached[s].path_cost:
                reached[s] = child
                heapq.heappush(frontier, (child.path_cost, child))
    return None

# ---- Langkah 4: Depth-First Search ----
def is_cycle(node):
    s, p = node.state, node.parent
    while p is not None:
        if p.state == s:
            return True
        p = p.parent
    return False

def depth_first_search(problem, stats):
    frontier = [Node(problem.initial)]       # LIFO queue (stack)
    while frontier:
        node = frontier.pop()
        if problem.is_goal(node.state):
            return node
        stats["expanded"] += 1
        children = [c for c in expand(problem, node) if not is_cycle(c)]
        frontier.extend(reversed(children))  # tetangga pertama diproses dulu
    return None

# ---- Langkah 5: Depth-Limited dan Iterative Deepening Search ----
CUTOFF = "cutoff"

def depth_limited_search(problem, limit, stats):
    frontier = [Node(problem.initial)]
    result = None                            # None berarti failure
    while frontier:
        node = frontier.pop()
        if problem.is_goal(node.state):
            return node
        if node.depth >= limit:
            result = CUTOFF
        elif not is_cycle(node):
            stats["expanded"] += 1
            frontier.extend(reversed(list(expand(problem, node))))
    return result

def iterative_deepening_search(problem, stats):
    depth = 0
    while True:
        result = depth_limited_search(problem, depth, stats)
        if result != CUTOFF:
            return result
        depth += 1

# ---- Langkah 6: Eksperimen Perbandingan ----
def jalankan(nama, fungsi, problem):
    stats = {"expanded": 0}
    t0 = time.perf_counter()
    hasil = fungsi(problem, stats)
    dt = (time.perf_counter() - t0) * 1000
    if hasil is None:
        print(f"{nama:<5} gagal menemukan solusi")
        return
    rute = " -> ".join(path_states(hasil))
    print(f"{nama:<5} | {rute}")
    print(f"      | langkah={hasil.depth}, biaya={hasil.path_cost} km, "
          f"ekspansi={stats['expanded']}, waktu={dt:.3f} ms")

if __name__ == "__main__":
    print("=== PRAKTIKUM 3: UNINFORMED SEARCH (PADANG -> PAYAKUMBUH) ===")
    masalah = Problem("Padang", "Payakumbuh", PETA_SUMBAR)
    for nama, f in [("BFS", breadth_first_search),
                    ("UCS", uniform_cost_search),
                    ("DFS", depth_first_search),
                    ("IDS", iterative_deepening_search)]:
        jalankan(nama, f, masalah)
