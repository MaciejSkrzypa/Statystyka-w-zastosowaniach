"""Zadanie 5: ANOVA dla trzech grup oraz test post hoc Tukeya."""

from __future__ import annotations

import numpy as np
from scipy import stats
from statsmodels.stats.multicomp import pairwise_tukeyhsd


def main() -> None:
    """Wykonuje jednoczynnikowa ANOVA i test post hoc dla trzech grup."""
    rng = np.random.default_rng(20260325)
    g1 = rng.normal(loc=10.0, scale=2.0, size=35)
    g2 = rng.normal(loc=12.0, scale=2.0, size=35)
    g3 = rng.normal(loc=15.0, scale=2.0, size=35)

    f_stat, p_value = stats.f_oneway(g1, g2, g3)

    print("=== Lista2/Zad5: ANOVA ===")
    print("Zbior danych: trzy syntetyczne grupy normalne: G1~N(10,2), G2~N(12,2), G3~N(15,2), size=35")
    print(f"Srednie: G1={np.mean(g1):.4f}, G2={np.mean(g2):.4f}, G3={np.mean(g3):.4f}")
    print(f"Statystyka F = {f_stat:.4f}, p-value = {p_value:.6f}")
    print("Decyzja (alpha=0.05):", "odrzucamy H0" if p_value < 0.05 else "brak podstaw do odrzucenia H0")

    if p_value < 0.05:
        all_values = np.concatenate([g1, g2, g3])
        groups = np.array(["G1"] * g1.size + ["G2"] * g2.size + ["G3"] * g3.size)
        tukey = pairwise_tukeyhsd(endog=all_values, groups=groups, alpha=0.05)
        print("\nTest post hoc Tukeya:")
        print(tukey.summary())


if __name__ == "__main__":
    main()
