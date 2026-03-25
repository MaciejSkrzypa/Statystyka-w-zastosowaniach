"""Zadanie 3: Test t-Studenta dla dwoch niezaleznych prob."""

from __future__ import annotations

import numpy as np
from scipy import stats


def main() -> None:
    """Porownuje srednie dwoch niezaleznych grup."""
    seed = 20260325
    rng = np.random.default_rng(seed)

    loc_a, scale_a, size_a = 100.0, 10.0, 45
    loc_b, scale_b, size_b = 105.0, 10.0, 45
    group_a = rng.normal(loc=loc_a, scale=scale_a, size=size_a)
    group_b = rng.normal(loc=loc_b, scale=scale_b, size=size_b)

    t_stat, p_value = stats.ttest_ind(group_a, group_b, equal_var=False)

    print("=== Lista2/Zad3: Test t dla dwoch niezaleznych prob ===")
    print(f"Seed losowosci = {seed}")
    print("Zbior danych: dwie syntetyczne grupy normalne (A i B) do testu t dla prob niezaleznych")
    print(
        "Generowanie danych:\n"
        f"- Grupa A: N(loc={loc_a}, scale={scale_a}), size={size_a}\n"
        f"- Grupa B: N(loc={loc_b}, scale={scale_b}), size={size_b}"
    )
    print("H0: srednie grup sa rowne, H1: srednie grup sa rozne")
    print(f"Srednia grupa A = {np.mean(group_a):.4f}")
    print(f"Srednia grupa B = {np.mean(group_b):.4f}")
    print(f"Statystyka t = {t_stat:.4f}, p-value = {p_value:.6f}")
    print("Decyzja (alpha=0.05):", "odrzucamy H0" if p_value < 0.05 else "brak podstaw do odrzucenia H0")


if __name__ == "__main__":
    main()
