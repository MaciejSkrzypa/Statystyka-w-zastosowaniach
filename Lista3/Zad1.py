"""Lista 3, Zadanie 1: test t-Studenta dla dwoch niezaleznych prob."""

from __future__ import annotations

import numpy as np
from scipy import stats


def main() -> None:
    """Porownuje sredni wzrost kobiet i mezczyzn na podstawie danych symulowanych."""
    rng = np.random.default_rng(20260325)
    wzrost_kobiet = rng.normal(loc=165.0, scale=6.0, size=30)
    wzrost_mezczyzn = rng.normal(loc=178.0, scale=7.0, size=30)

    t_stat, p_value = stats.ttest_ind(wzrost_kobiet, wzrost_mezczyzn, equal_var=False)

    print("=== Lista3/Zad1: test t-Studenta dla dwoch niezaleznych prob ===")
    print("Zbior danych: wzrost kobiet ~N(165,6), wzrost mezczyzn ~N(178,7), po 30 obserwacji")
    print("H0: sredni wzrost kobiet i mezczyzn jest taki sam.")
    print("H1: sredni wzrost kobiet i mezczyzn jest rozny.")
    print(f"Sredni wzrost kobiet   = {np.mean(wzrost_kobiet):.2f} cm")
    print(f"Sredni wzrost mezczyzn = {np.mean(wzrost_mezczyzn):.2f} cm")
    print(f"Statystyka t = {t_stat:.4f}, p-value = {p_value:.6f}")
    print("Decyzja przy alpha=0.05:", "odrzucamy H0" if p_value < 0.05 else "brak podstaw do odrzucenia H0")


if __name__ == "__main__":
    main()
