# puzzle8.py
# Praktikum 3: Penerapan Algoritma Uninformed Search pada 8-Puzzle

from pencarian_buta import (Problem, breadth_first_search,
                            iterative_deepening_search, path_states)

class EightPuzzle(Problem):
    GOAL = (1, 2, 3, 4, 5, 6, 7, 8, 0)          # 0 = ubin kosong
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

if __name__ == "__main__":
    print("=== PRAKTIKUM 3: 8-PUZZLE UNINFORMED SEARCH ===")
    kasus = {
        "mudah":  (1, 2, 3, 4, 0, 6, 7, 5, 8),
        "sedang": (4, 1, 3, 7, 2, 6, 0, 5, 8),
        "sulit":  (3, 5, 0, 1, 4, 2, 7, 8, 6),
        "berat":  (1, 8, 3, 5, 0, 6, 4, 2, 7)
    }
    for nama, awal in kasus.items():
        for label, f in [("BFS", breadth_first_search),
                         ("IDS", iterative_deepening_search)]:
            stats = {"expanded": 0}
            hasil = f(EightPuzzle(awal), stats)
            print(f"{nama:<7} {label}: langkah={hasil.depth}, "
                  f"ekspansi={stats['expanded']}")
