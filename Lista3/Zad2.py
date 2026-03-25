"""Lista 3, Zadanie 2: test t dla jednej proby i wplyw zmiany sredniej."""

from __future__ import annotations

import numpy as np
from scipy import stats


def run_one_sample_test(sample: np.ndarray, mu0: float) -> tuple[float, float, float]:
    """Zwraca srednia proby, statystyke t i p-value dla testu jednej proby."""
    t_stat, p_value = stats.ttest_1samp(sample, popmean=mu0)
    return float(np.mean(sample)), float(t_stat), float(p_value)


def main() -> None:
    """Sprawdza, czy srednia wzrostu rozni sie od 170 cm dla dwoch scenariuszy."""
    rng = np.random.default_rng(20260325)
    mu0 = 170.0

    probka_1 = rng.normal(loc=169.5, scale=8.0, size=30)
    probka_2 = rng.normal(loc=174.0, scale=8.0, size=30)

    m1, t1, p1 = run_one_sample_test(probka_1, mu0)
    m2, t2, p2 = run_one_sample_test(probka_2, mu0)

    print("=== Lista3/Zad2: test t dla jednej proby ===")
    print("Zbior danych: dwa scenariusze prob normalnych: N(169.5,8) i N(174,8), po 30 obserwacji")
    print("H0: srednia wzrostu = 170 cm.")
    print("H1: srednia wzrostu != 170 cm.")
    print("\nScenariusz 1 (srednia bliska 170):")
    print(f"Srednia proby = {m1:.2f} cm, t = {t1:.4f}, p-value = {p1:.6f}")
    print("Decyzja:", "odrzucamy H0" if p1 < 0.05 else "brak podstaw do odrzucenia H0")

    print("\nScenariusz 2 (srednia przesunieta):")
    print(f"Srednia proby = {m2:.2f} cm, t = {t2:.4f}, p-value = {p2:.6f}")
    print("Decyzja:", "odrzucamy H0" if p2 < 0.05 else "brak podstaw do odrzucenia H0")


if __name__ == "__main__":
    main()
