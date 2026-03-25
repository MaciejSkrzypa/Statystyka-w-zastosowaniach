"""Lista 3, Zadanie 4: porownanie testu parametrycznego i nieparametrycznego."""

from __future__ import annotations

import numpy as np
from scipy import stats


def main() -> None:
    """Porownuje test t i test Manna-Whitneya dla danych nienormalnych."""
    rng = np.random.default_rng(20260325)
    grupa_a = rng.exponential(scale=1.0, size=60)
    grupa_b = rng.exponential(scale=1.15, size=60)

    t_stat, p_t = stats.ttest_ind(grupa_a, grupa_b, equal_var=False)
    u_stat, p_u = stats.mannwhitneyu(grupa_a, grupa_b, alternative="two-sided")

    print("=== Lista3/Zad4: test t kontra test Manna-Whitneya ===")
    print("Zbior danych: dwie grupy z rozkladu wykladniczego (scale=1.0 i 1.15), po 60 obserwacji")
    print(f"Srednia grupa A = {np.mean(grupa_a):.4f}")
    print(f"Srednia grupa B = {np.mean(grupa_b):.4f}")
    print(f"Test t: statystyka = {t_stat:.4f}, p-value = {p_t:.6f}")
    print(f"Test Manna-Whitneya: statystyka U = {u_stat:.4f}, p-value = {p_u:.6f}")
    print(
        "Interpretacja: dla rozkladow odleglych od normalnego test nieparametryczny "
        "jest zwykle bezpieczniejszym wyborem."
    )


if __name__ == "__main__":
    main()
