import heapq
import math
import random
import statistics
import time

# ---- 1. Peta Sumatera Barat ----
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

def jarak_garis_lurus(a, b):
    R = 6371.0
    la1, lo1 = map(math.radians, KOORDINAT[a])
    la2, lo2 = map(math.radians, KOORDINAT[b])
    d = (math.sin((la2 - la1) / 2) ** 2
         + math.cos(la1) * math.cos(la2) * math.sin((lo2 - lo1) / 2) ** 2)
    return 2 * R * math.asin(math.sqrt(d))

def buat_h_sld(goal):
    tabel = {kota: int(jarak_garis_lurus(kota, goal)) for kota in KOORDINAT}
    def h(node):
        return tabel[node.state]
    h.tabel = tabel
    return h

def periksa_heuristik(graph, tabel_h, h_bintang):
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

# ---- Best-First Search Kerangka ----
def best_first_search(problem, f, stats):
    node = Node(problem.initial)
    frontier = [(f(node), node)]
    reached = {problem.initial: node}
    stats["max_frontier"] = max(stats.get("max_frontier", 0), len(frontier))
    while frontier:
        nilai, node = heapq.heappop(frontier)
        if nilai > f(reached[node.state]):
            continue
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
    return best_first_search(problem, lambda n: n.path_cost, stats)

def greedy_best_first_search(problem, h, stats):
    return best_first_search(problem, lambda n: h(n), stats)

def a_star_search(problem, h, stats):
    return best_first_search(problem, lambda n: n.path_cost + h(n), stats)

def weighted_a_star_search(problem, h, W, stats):
    return best_first_search(problem, lambda n: n.path_cost + W * h(n), stats)

# ---- IDA* dengan Pencatatan Max Frontier ----
def _siklus(node):
    s, p = node.state, node.parent
    while p is not None:
        if p.state == s: return True
        p = p.parent
    return False

def _cari_terbatas_f(problem, node, batas, h, stats, path_nodes):
    f = node.path_cost + h(node)
    stats["max_frontier"] = max(stats.get("max_frontier", 0), len(path_nodes))
    if f > batas:
        return None, f
    if problem.is_goal(node.state):
        return node, batas
    stats["expanded"] += 1
    minimum = math.inf
    for child in expand(problem, node):
        if _siklus(child):
            continue
        path_nodes.append(child)
        hasil, nilai = _cari_terbatas_f(problem, child, batas, h, stats, path_nodes)
        path_nodes.pop()
        if hasil is not None:
            return hasil, batas
        minimum = min(minimum, nilai)
    return None, minimum

def ida_star_search(problem, h, stats):
    akar = Node(problem.initial)
    batas = h(akar)
    path_nodes = [akar]
    stats["max_frontier"] = 1
    while True:
        stats["iterasi"] = stats.get("iterasi", 0) + 1
        hasil, batas_baru = _cari_terbatas_f(problem, akar, batas, h, stats, path_nodes)
        if hasil is not None:
            return hasil
        if batas_baru == math.inf:
            return None
        batas = batas_baru

# ---- 8-Puzzle Problem & Heuristics ----
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
        i = s.index(0); j = i + self.GESER[a]
        s = list(s); s[i], s[j] = s[j], s[i]
        return tuple(s)

    def action_cost(self, s, a, s2):
        return 1

def h1_salah_tempat(node):
    return sum(1 for i, v in enumerate(node.state)
               if v != 0 and v != EightPuzzle.GOAL[i])

def h2_manhattan(node):
    total = 0
    for i, v in enumerate(node.state):
        if v == 0: continue
        tujuan = EightPuzzle.GOAL.index(v)
        total += abs(i // 3 - tujuan // 3) + abs(i % 3 - tujuan % 3)
    return total

def h3_max(node):
    return max(h1_salah_tempat(node), h2_manhattan(node))

def effective_branching_factor(N, d, toleransi=1e-6):
    if d == 0: return 0.0
    def total(b): return sum(b ** i for i in range(1, d + 1))
    lo, hi = 1.0000001, 100.0
    while hi - lo > toleransi:
        tengah = (lo + hi) / 2
        if total(tengah) < N: lo = tengah
        else: hi = tengah
    return round((lo + hi) / 2, 2)

# ---- Beam Search (Pengayaan) ----
def beam_search(problem, h, k, max_steps=2000):
    akar = Node(problem.initial)
    if problem.is_goal(akar.state): return akar, 0, 1
    
    beam = [akar]
    expanded_count = 0
    max_front = 1
    
    for step in range(max_steps):
        if not beam:
            return None, expanded_count, max_front
        
        all_candidates = []
        for node in beam:
            expanded_count += 1
            for child in expand(problem, node):
                if problem.is_goal(child.state):
                    return child, expanded_count, max(max_front, len(beam))
                all_candidates.append(child)
        
        if not all_candidates:
            return None, expanded_count, max_front
        
        all_candidates.sort(key=lambda n: h(n))
        
        unique_candidates = []
        seen = set()
        for cand in all_candidates:
            if cand.state not in seen:
                seen.add(cand.state)
                unique_candidates.append(cand)
            if len(unique_candidates) == k:
                break
        
        beam = unique_candidates
        max_front = max(max_front, len(beam))
        
    return None, expanded_count, max_front

# ---- Graph Kampus PNP (Pengayaan) ----
GRAPH_PNP = {
    "Gerbang Utama":       {"Gedung Utama/Rektorat": 150, "Parkiran Depan": 100},
    "Gedung Utama/Rektorat": {"Gerbang Utama": 150, "Perpustakaan": 120, "Gedung Sipil": 110},
    "Parkiran Depan":      {"Gerbang Utama": 100, "Gedung Elektro": 130},
    "Gedung Sipil":        {"Gedung Utama/Rektorat": 110, "Gedung TI": 140, "Gedung Mesin": 120},
    "Perpustakaan":        {"Gedung Utama/Rektorat": 120, "Gedung TI": 110, "Masjid Al-Janah": 160},
    "Gedung Elektro":      {"Parkiran Depan": 130, "Kantin Utama": 90, "Gedung TI": 150},
    "Gedung TI":           {"Gedung Sipil": 140, "Perpustakaan": 110, "Gedung Elektro": 150, "Pusat Kegiatan Mahasiswa": 130},
    "Gedung Mesin":        {"Gedung Sipil": 120, "Bengkel Praktik": 100},
    "Kantin Utama":        {"Gedung Elektro": 90, "Masjid Al-Janah": 150, "Pusat Kegiatan Mahasiswa": 110},
    "Masjid Al-Janah":     {"Perpustakaan": 160, "Kantin Utama": 150, "Lapangan Olahraga": 120},
    "Pusat Kegiatan Mahasiswa": {"Gedung TI": 130, "Kantin Utama": 110, "Lapangan Olahraga": 140},
    "Bengkel Praktik":     {"Gedung Mesin": 100},
    "Lapangan Olahraga":   {"Masjid Al-Janah": 120, "Pusat Kegiatan Mahasiswa": 140}
}

# Koordinat disesuaikan agar terbukti Admissible & Consistent
KOORDINAT_PNP = {
    "Gerbang Utama":       (0, 0),
    "Parkiran Depan":      (80, -40),
    "Gedung Utama/Rektorat": (120, 50),
    "Gedung Sipil":        (220, 70),
    "Perpustakaan":        (220, -10),
    "Gedung Elektro":      (180, -40),
    "Gedung TI":           (310, 0),
    "Gedung Mesin":        (310, 140),
    "Kantin Utama":        (250, -120),
    "Masjid Al-Janah":     (350, -90),
    "Pusat Kegiatan Mahasiswa": (360, -50),
    "Bengkel Praktik":     (390, 160),
    "Lapangan Olahraga":   (420, -80)
}

def buat_h_pnp(goal):
    gx, gy = KOORDINAT_PNP[goal]
    tabel = {}
    for node, (x, y) in KOORDINAT_PNP.items():
        tabel[node] = int(math.sqrt((x - gx)**2 + (y - gy)**2))
    def h(node):
        return tabel[node.state]
    h.tabel = tabel
    return h

if __name__ == "__main__":
    print("\n1. TUGAS 1: PAINAN -> LUBUK BASUNG (INFORMED SEARCH & HEURISTIK SLD)")
    
    masalah_p1 = Problem("Painan", "Lubuk Basung", PETA_SUMBAR)
    h_lb = buat_h_sld("Lubuk Basung")
    
    # h* dari UCS
    h_bintang_lb = {}
    for kota in PETA_SUMBAR:
        prob_k = Problem(kota, "Lubuk Basung", PETA_SUMBAR)
        sol_k = uniform_cost(prob_k, h_lb, {"expanded": 0})
        h_bintang_lb[kota] = sol_k.path_cost if sol_k else math.inf
        
    print("\nTabel Heuristik h_SLD ke Lubuk Basung:")
    for k, v in h_lb.tabel.items():
        print(f"  {k:<15}: h_SLD = {v:>3} km | h* = {h_bintang_lb[k]:>3} km")
        
    lap_lb = periksa_heuristik(PETA_SUMBAR, h_lb.tabel, h_bintang_lb)
    print(f"\nPemeriksaan Heuristik Lubuk Basung:")
    print(f"  Admissible : {lap_lb['admissible']}")
    print(f"  Consistent : {lap_lb['consistent']}")
    
    print("\nPerbandingan Algoritma (Painan -> Lubuk Basung):")
    for nama_alg, f_alg in [("UCS", uniform_cost),
                             ("Greedy", greedy_best_first_search),
                             ("A*", a_star_search),
                             ("IDA*", ida_star_search)]:
        st = {"expanded": 0}
        res = f_alg(masalah_p1, h_lb, st)
        rute = " -> ".join(path_states(res))
        print(f"{nama_alg:<8} | Rute: {rute}")
        print(f"         | Biaya: {res.path_cost} km, Expanded: {st['expanded']}, Max Frontier: {st['max_frontier']}")

    print("\n2. TUGAS 2: HEURISTIK INADMISSIBLE (1.5 x h_SLD)")
    
    def h_inadmissible(node):
        return int(1.5 * h_lb(node))
    
    st_std = {"expanded": 0}
    res_std = a_star_search(masalah_p1, h_lb, st_std)
    
    st_inad = {"expanded": 0}
    res_inad = a_star_search(masalah_p1, h_inadmissible, st_inad)
    
    print(f"A* Standard (h_SLD)     | Rute: {' -> '.join(path_states(res_std))} | Biaya: {res_std.path_cost} km | Expanded: {st_std['expanded']}")
    print(f"A* Inadmissible (1.5x)  | Rute: {' -> '.join(path_states(res_inad))} | Biaya: {res_inad.path_cost} km | Expanded: {st_inad['expanded']}")

    print("\n3    . TUGAS 3: PENCATATAN MAX_FRONTIER IDA* VS A* (8-PUZZLE d = 22)")
    
    def instans_d22(seed=20260925):
        rng = random.Random(seed)
        p = EightPuzzle(EightPuzzle.GOAL)
        while True:
            s = EightPuzzle.GOAL
            for _ in range(60):
                s = p.result(s, rng.choice(p.actions(s)))
            st = {"expanded": 0}
            sol = a_star_search(EightPuzzle(s), h2_manhattan, st)
            if sol.depth == 22:
                return s
                
    state_22 = instans_d22()
    prob_p22 = EightPuzzle(state_22)
    
    st_a_star = {"expanded": 0}
    sol_a = a_star_search(prob_p22, h2_manhattan, st_a_star)
    
    st_ida = {"expanded": 0}
    sol_ida = ida_star_search(prob_p22, h2_manhattan, st_ida)
    
    print(f"A* Search  | Langkah: {sol_a.depth}, Expanded: {st_a_star['expanded']}, Max Frontier: {st_a_star['max_frontier']} node")
    print(f"IDA* Search | Langkah: {sol_ida.depth}, Expanded: {st_ida['expanded']}, Max Frontier: {st_ida['max_frontier']} node")

    print("\n4. TUGAS 4: HEURISTIK h3 = max(h1, h2) & DOMINANCE (8-PUZZLE)")
    
    rng = random.Random(20260925)
    kasus_14 = []
    p_ref = EightPuzzle(EightPuzzle.GOAL)
    while len(kasus_14) < 5:
        s = EightPuzzle.GOAL
        for _ in range(40):
            s = p_ref.result(s, rng.choice(p_ref.actions(s)))
        st = {"expanded": 0}
        sol = a_star_search(EightPuzzle(s), h2_manhattan, st)
        if sol.depth == 14 and s not in kasus_14:
            kasus_14.append(s)
            
    print(f"{'Instans':<8} | {'A*(h1)':>10} {'A*(h2)':>10} {'A*(h3=max)':>12}")
    print("-" * 50)
    for idx, ks in enumerate(kasus_14, 1):
        st1, st2, st3 = {"expanded": 0}, {"expanded": 0}, {"expanded": 0}
        a_star_search(EightPuzzle(ks), h1_salah_tempat, st1)
        a_star_search(EightPuzzle(ks), h2_manhattan, st2)
        a_star_search(EightPuzzle(ks), h3_max, st3)
        print(f"Kasus {idx:<2}  | {st1['expanded']:>10} {st2['expanded']:>10} {st3['expanded']:>12}")

    print("\n5. PENGAYAAN 1: BEAM SEARCH DENGAN LEBAR k (8-PUZZLE)")
    
    print(f"{'Lebar k':<8} | {'Tingkat Sukses':>14} | {'Rata Biaya Solusi':>18} | {'Rata Node Expanded':>18}")
    print("-" * 65)
    for k in (5, 10, 20, 50):
        sukses = 0
        biaya_list = []
        expanded_list = []
        for ks in kasus_14:
            sol, exp, front = beam_search(EightPuzzle(ks), h2_manhattan, k)
            if sol is not None:
                sukses += 1
                biaya_list.append(sol.path_cost)
            expanded_list.append(exp)
            
        rate = f"{sukses}/{len(kasus_14)} ({sukses/len(kasus_14)*100:.0f}%)"
        avg_biaya = f"{statistics.mean(biaya_list):.1f}" if biaya_list else "N/A"
        avg_exp = f"{statistics.mean(expanded_list):.1f}"
        print(f"k = {k:<5} | {rate:>14} | {avg_biaya:>18} | {avg_exp:>18}")

    print("\n6. PENGAYAAN 2: A* VS UCS PADA GRAPH KAMPUS PNP")
    
    pnp_prob = Problem("Gerbang Utama", "Gedung TI", GRAPH_PNP)
    h_pnp_ti = buat_h_pnp("Gedung TI")
    
    h_bintang_pnp = {}
    for node in GRAPH_PNP:
        prob_n = Problem(node, "Gedung TI", GRAPH_PNP)
        sol_n = uniform_cost(prob_n, h_pnp_ti, {"expanded": 0})
        h_bintang_pnp[node] = sol_n.path_cost if sol_n else math.inf
        
    lap_pnp = periksa_heuristik(GRAPH_PNP, h_pnp_ti.tabel, h_bintang_pnp)
    print(f"Pemeriksaan Heuristik Euklides PNP (ke Gedung TI):")
    print(f"  Admissible : {lap_pnp['admissible']}")
    print(f"  Consistent : {lap_pnp['consistent']}")
    
    st_pnp_ucs = {"expanded": 0}
    res_pnp_ucs = uniform_cost(pnp_prob, h_pnp_ti, st_pnp_ucs)
    
    st_pnp_astar = {"expanded": 0}
    res_pnp_astar = a_star_search(pnp_prob, h_pnp_ti, st_pnp_astar)
    
    print(f"\nUCS (Gerbang Utama -> Gedung TI) | Rute: {' -> '.join(path_states(res_pnp_ucs))} | Jarak: {res_pnp_ucs.path_cost} m | Expanded: {st_pnp_ucs['expanded']}")
    print(f"A*  (Gerbang Utama -> Gedung TI) | Rute: {' -> '.join(path_states(res_pnp_astar))} | Jarak: {res_pnp_astar.path_cost} m | Expanded: {st_pnp_astar['expanded']}")
