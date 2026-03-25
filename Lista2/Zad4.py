"""Zadanie 4: Test chi-kwadrat dla zgodnosci rozkladu."""

from __future__ import annotations

import numpy as np
from scipy.stats import chisquare


def main() -> None:
    """Porownuje czestosci obserwowane i oczekiwane dla zmiennej kategorycznej."""
    rng = np.random.default_rng(20260325)
    categories = np.array(["A", "B", "C"])
    observed_sample = rng.choice(categories, size=300, p=[0.45, 0.30, 0.25])

    observed_counts = np.array([(observed_sample == cat).sum() for cat in categories], dtype=float)
    expected_counts = np.full(shape=3, fill_value=100.0)

    chi2_stat, p_value = chisquare(f_obs=observed_counts, f_exp=expected_counts)

    print("=== Lista2/Zad4: Test chi-kwadrat zgodnosci ===")
    print("Zbior danych: syntetyczne dane kategoryczne A/B/C (multinomial), size=300")
    print(f"Obserwowane liczebnosci: {dict(zip(categories, observed_counts.astype(int), strict=False))}")
    print(f"Oczekiwane liczebnosci:  {dict(zip(categories, expected_counts.astype(int), strict=False))}")
    print(f"Statystyka chi2 = {chi2_stat:.4f}, p-value = {p_value:.6f}")
    print("Decyzja (alpha=0.05):", "odrzucamy H0" if p_value < 0.05 else "brak podstaw do odrzucenia H0")


if __name__ == "__main__":
    main()
