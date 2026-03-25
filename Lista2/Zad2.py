"""Zadanie 2: Test t-Studenta dla jednej proby."""

from __future__ import annotations

import numpy as np
from scipy import stats


def main() -> None:
    """Testuje hipoteze H0 o zadanej sredniej populacyjnej."""
    rng = np.random.default_rng(20260325)
    mu0 = 100.0
    sample = rng.normal(loc=103.0, scale=12.0, size=40)
    t_stat, p_value = stats.ttest_1samp(sample, popmean=mu0)

    print("=== Lista2/Zad2: Test t dla jednej proby ===")
    print("Zbior danych: syntetyczna proba z rozkladu normalnego N(103, 12), size=40")
    print(f"H0: mu = {mu0}, H1: mu != {mu0}")
    print(f"Srednia proby = {np.mean(sample):.4f}, statystyka t = {t_stat:.4f}, p-value = {p_value:.6f}")
    print("Decyzja (alpha=0.05):", "odrzucamy H0" if p_value < 0.05 else "brak podstaw do odrzucenia H0")


if __name__ == "__main__":
    main()
