# eksperimen4.py
# Praktikum 4: Eksperimen Kualitas Heuristik & Satisficing Search pada 8-Puzzle

import random
import statistics
from pencarian_terarah import (EightPuzzle, a_star_search, greedy_best_first_search,
                               weighted_a_star_search, h0_nol, h1_salah_tempat,
                               h2_manhattan, effective_branching_factor)

def instans_acak(d, n, seed):
    """Membangkitkan n instans 8-puzzle yang solusi optimalnya tepat d langkah."""
    rng = random.Random(seed)
    contoh = EightPuzzle(EightPuzzle.GOAL)
    hasil = []
    while len(hasil) < n:
        s = EightPuzzle.GOAL
        for _ in range(rng.randint(d, d * 4)):          # acak dari goal: pasti solvable
            s = contoh.result(s, rng.choice(contoh.actions(s)))
        stats = {"expanded": 0}
        solusi = a_star_search(EightPuzzle(s), h2_manhattan, stats)
        if solusi.depth == d and s not in hasil:
            hasil.append(s)
    return hasil

def eksperimen_heuristik(kedalaman=(6, 10, 14, 18, 22), n=5, seed=20260925):
    print("A. Pengaruh Kualitas Heuristik (rata-rata 5 instans per kedalaman)")
    print(f"{'d':>3} | {'UCS (h=0)':>10} {'A*(h1)':>9} {'A*(h2)':>8} | "
          f"{'b* UCS':>7} {'b* h1':>6} {'b* h2':>6}")
    print("-" * 60)
    for d in kedalaman:
        kasus = instans_acak(d, n, seed + d)
        rata = {}
        for nama, h in [("UCS", h0_nol), ("h1", h1_salah_tempat), ("h2", h2_manhattan)]:
            jumlah = []
            for s in kasus:
                stats = {"expanded": 0}
                a_star_search(EightPuzzle(s), h, stats)
                jumlah.append(stats["expanded"])
            m = statistics.mean(jumlah)
            rata[nama] = (round(m), effective_branching_factor(m, d))
        print(f"{d:>3} | {rata['UCS'][0]:>10} {rata['h1'][0]:>9} {rata['h2'][0]:>8} | "
              f"{rata['UCS'][1]:>7} {rata['h1'][1]:>6} {rata['h2'][1]:>6}")

def eksperimen_satisficing(d=22, n=5, seed=20260925):
    kasus = instans_acak(d, n, seed + d)
    print("\nB. Pencarian Satisficing pada Instans d = 22")
    print(f"{'Algoritma':<14} {'ekspansi':>9} {'biaya':>7} {'frontier':>9}")
    print("-" * 45)
    daftar = [("A* (W=1)", lambda p, s: a_star_search(p, h2_manhattan, s)),
              ("WA* W=1.5", lambda p, s: weighted_a_star_search(p, h2_manhattan, 1.5, s)),
              ("WA* W=2", lambda p, s: weighted_a_star_search(p, h2_manhattan, 2, s)),
              ("WA* W=3", lambda p, s: weighted_a_star_search(p, h2_manhattan, 3, s)),
              ("WA* W=5", lambda p, s: weighted_a_star_search(p, h2_manhattan, 5, s)),
              ("Greedy (h2)", lambda p, s: greedy_best_first_search(p, h2_manhattan, s))]
    for nama, f in daftar:
        ek, bi, fr = [], [], []
        for s in kasus:
            stats = {"expanded": 0, "max_frontier": 0}
            hasil = f(EightPuzzle(s), stats)
            ek.append(stats["expanded"])
            bi.append(hasil.path_cost)
            fr.append(stats["max_frontier"])
        print(f"{nama:<14} {statistics.mean(ek):>9.0f} {statistics.mean(bi):>7.1f} "
              f"{statistics.mean(fr):>9.0f}")

if __name__ == "__main__":
    print("=== PRAKTIKUM 4: EKSPERIMEN HEURISTIK 8-PUZZLE ===")
    eksperimen_heuristik()
    eksperimen_satisficing()
