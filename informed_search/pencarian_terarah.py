# pencarian_terarah.py
# Praktikum 4: Implementasi Algoritma Pencarian Terarah (Informed Search)

import heapq
import math
import sys
import os

# Import komponen dasar dari pencarian_buta
sys.path.append(os.path.dirname(__file__))
from pencarian_buta import PETA_SUMBAR, Problem, Node, expand, path_states

# ---- Langkah 1: Fungsi Heuristik untuk Peta Sumatera Barat ----
KOORDINAT = {
    "Padang":         (-0.9471, 100.4172),
    "Pariaman":       (-0.6264, 100.1197),
    "Painan":         (-1.3500, 100.5667),
    "Lubuk Basung":   (-0.3333, 100.0333),
    "Padang Panjang": (-0.4667, 100.4000),
    "Bukittinggi":    (-0.3056, 100.3692),
    "Batusangkar":    (-0.4500, 100.5833),
    "Solok":          (-0.7986, 100.6533),
    "Sawahlunto":     (-0.6833, 100.7833),
    "Payakumbuh":     (-0.2247, 100.6317)
}

def jarak_garis_lurus(a, b):
    """Jarak garis lurus (haversine) antara dua kota dalam km."""
    R = 6371.0
    la1, lo1 = map(math.radians, KOORDINAT[a])
    la2, lo2 = map(math.radians, KOORDINAT[b])
    d = (math.sin((la2 - la1) / 2) ** 2
         + math.cos(la1) * math.cos(la2) * math.sin((lo2 - lo1) / 2) ** 2)
    return 2 * R * math.asin(math.sqrt(d))

def buat_h_sld(goal):
    """Membuat fungsi heuristik h(node) = jarak garis lurus ke goal (dibulatkan ke bawah)."""
    tabel = {kota: int(jarak_garis_lurus(kota, goal)) for kota in KOORDINAT}
    def h(node):
        return tabel[node.state]
    h.tabel = tabel
    return h

# ---- Langkah 2: Memeriksa Admissibility dan Consistency ----
def periksa_heuristik(graph, tabel_h, h_bintang):
    """h admissible bila h(n) <= h*(n); consistent bila h(n) <= c(n,a,n') + h(n')."""
    laporan = {"admissible": True, "consistent": True, "pelanggaran": []}
    for kota in graph:
        if tabel_h[kota] > h_bintang[kota]:
            laporan["admissible"] = False
            laporan["pelanggaran"].append(
                f"admissible: h({kota})={tabel_h[kota]} > h*={h_bintang[kota]}")
    for a, tetangga in graph.items():
        for b, c in tetangga.items():
            if tabel_h[a] > c + tabel_h[b]:
                laporan["consistent"] = False
                laporan["pelanggaran"].append(
                    f"consistent: h({a})={tabel_h[a]} > {c} + h({b})={tabel_h[b]}")
    return laporan

# ---- Langkah 3: Best-First Search dan Turunannya ----
def best_first_search(problem, f, stats):
    """Kerangka umum: meng-expand node dengan nilai f(node) terkecil."""
    node = Node(problem.initial)
    frontier = [(f(node), node)]                 # priority queue menurut f
    reached = {problem.initial: node}
    stats["max_frontier"] = max(stats.get("max_frontier", 0), len(frontier))
    while frontier:
        nilai, node = heapq.heappop(frontier)
        if nilai > f(reached[node.state]):
            continue                             # entri usang (lazy deletion)
        if problem.is_goal(node.state):
            return node
        stats["expanded"] += 1
        for child in expand(problem, node):
            s = child.state
            if s not in reached or child.path_cost < reached[s].path_cost:
                reached[s] = child
                heapq.heappush(frontier, (f(child), child))
        stats["max_frontier"] = max(stats.get("max_frontier", 0), len(frontier))
    return None

def uniform_cost(problem, h, stats):
    """f(n) = g(n) — uniform-cost search; heuristik tidak dipakai."""
    return best_first_search(problem, lambda n: n.path_cost, stats)

def greedy_best_first_search(problem, h, stats):
    """f(n) = h(n)"""
    return best_first_search(problem, lambda n: h(n), stats)

def a_star_search(problem, h, stats):
    """f(n) = g(n) + h(n)"""
    return best_first_search(problem, lambda n: n.path_cost + h(node_or_n(n, h)), stats)

def node_or_n(n, h):
    return n

def a_star_search(problem, h, stats):
    """f(n) = g(n) + h(n)"""
    return best_first_search(problem, lambda n: n.path_cost + h(n), stats)

def weighted_a_star_search(problem, h, W, stats):
    """f(n) = g(n) + W x h(n), dengan W > 1"""
    return best_first_search(problem, lambda n: n.path_cost + W * h(n), stats)

# ---- Langkah 4: Iterative-Deepening A* (IDA*) ----
def ida_star_search(problem, h, stats):
    """A* dengan memori linear: batas iterasi berupa f-cost, bukan kedalaman."""
    akar = Node(problem.initial)
    batas = h(akar)
    while True:
        stats["iterasi"] = stats.get("iterasi", 0) + 1
        hasil, batas_baru = _cari_terbatas_f(problem, akar, batas, h, stats)
        if hasil is not None:
            return hasil
        if batas_baru == math.inf:
            return None                          # tidak ada solusi
        batas = batas_baru                       # kontur f berikutnya

def _cari_terbatas_f(problem, node, batas, h, stats):
    f = node.path_cost + h(node)
    if f > batas:
        return None, f                           # calon batas iterasi berikutnya
    if problem.is_goal(node.state):
        return node, batas
    stats["expanded"] += 1
    minimum = math.inf
    for child in expand(problem, node):
        if _siklus(child):
            continue
        hasil, nilai = _cari_terbatas_f(problem, child, batas, h, stats)
        if hasil is not None:
            return hasil, batas
        minimum = min(minimum, nilai)
    return None, minimum

def _siklus(node):
    s, p = node.state, node.parent
    while p is not None:
        if p.state == s:
            return True
        p = p.parent
    return False

# ---- Langkah 5: 8-Puzzle dan Fungsi Heuristiknya ----
class EightPuzzle(Problem):
    GOAL = (1, 2, 3, 4, 5, 6, 7, 8, 0)           # 0 = ubin kosong
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
        i = s.index(0); j = i + self.GESER[a]
        s = list(s); s[i], s[j] = s[j], s[i]
        return tuple(s)

    def action_cost(self, s, a, s2):
        return 1

def h0_nol(node):
    """Heuristik nol: A* dengan h = 0 berperilaku seperti uniform-cost search."""
    return 0

def h1_salah_tempat(node):
    """h1 = jumlah ubin yang salah tempat (ubin kosong tidak dihitung)."""
    return sum(1 for i, v in enumerate(node.state)
               if v != 0 and v != EightPuzzle.GOAL[i])

def h2_manhattan(node):
    """h2 = jumlah jarak Manhattan setiap ubin ke posisi tujuannya."""
    total = 0
    for i, v in enumerate(node.state):
        if v == 0:
            continue
        tujuan = EightPuzzle.GOAL.index(v)
        total += abs(i // 3 - tujuan // 3) + abs(i % 3 - tujuan % 3)
    return total

# ---- Langkah 6: Effective Branching Factor ----
def effective_branching_factor(N, d, toleransi=1e-6):
    """Mencari b* dari N = b* + (b*)^2 + ... + (b*)^d dengan metode bagi dua."""
    if d == 0:
        return 0.0
    def total(b):
        return sum(b ** i for i in range(1, d + 1))
    lo, hi = 1.0000001, 100.0
    while hi - lo > toleransi:
        tengah = (lo + hi) / 2
        if total(tengah) < N:
            lo = tengah
        else:
            hi = tengah
    return round((lo + hi) / 2, 2)

# ---- Langkah 7: Eksperimen pada Peta Sumatera Barat ----
def jalankan(nama, fungsi, problem, h):
    stats = {"expanded": 0, "max_frontier": 0}
    hasil = fungsi(problem, h, stats)
    print(f"{nama:<16} | {' -> '.join(path_states(hasil))}")
    print(f"{'':<16} | biaya={hasil.path_cost} km, ekspansi={stats['expanded']}, "
          f"frontier_maks={stats['max_frontier']}")

if __name__ == "__main__":
    print("=== PRAKTIKUM 4: INFORMED SEARCH (PADANG -> PAYAKUMBUH) ===")
    h = buat_h_sld("Payakumbuh")
    masalah = Problem("Padang", "Payakumbuh", PETA_SUMBAR)
    for nama, f in [("UCS (f=g)", uniform_cost),
                    ("Greedy (f=h)", greedy_best_first_search),
                    ("A* (f=g+h)", a_star_search),
                    ("IDA*", ida_star_search),
                    ("Weighted A* W=2",
                     lambda p, h, s: weighted_a_star_search(p, h, 2, s))]:
        jalankan(nama, f, masalah, h)
