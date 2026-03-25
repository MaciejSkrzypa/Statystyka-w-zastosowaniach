"""Zadanie 1: Test Z dla jednej proby przy znanej wariancji."""

from __future__ import annotations

import numpy as np
from scipy.stats import norm


def main() -> None:
    """Przeprowadza test Z dla hipotezy H0: mu = 50."""
    rng = np.random.default_rng(20260325)
    mu0 = 50.0
    sigma = 10.0
    sample = rng.normal(loc=52.0, scale=sigma, size=60)
    n = sample.size
    x_bar = float(np.mean(sample))
    z_stat = (x_bar - mu0) / (sigma / np.sqrt(n))
    p_value = 2 * (1 - norm.cdf(abs(z_stat)))

    print("=== Lista2/Zad1: Test Z dla jednej proby ===")
    print("Zbior danych: syntetyczna proba z rozkladu normalnego N(52, 10), size=60")
    print(f"H0: mu = {mu0}, H1: mu != {mu0}")
    print(f"Srednia proby = {x_bar:.4f}, statystyka Z = {z_stat:.4f}, p-value = {p_value:.6f}")
    print("Decyzja (alpha=0.05):", "odrzucamy H0" if p_value < 0.05 else "brak podstaw do odrzucenia H0")


if __name__ == "__main__":
    main()
