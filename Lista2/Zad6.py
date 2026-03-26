"""Zadanie 6: Testy normalnosci Shapiro-Wilka i Kolmogorowa-Smirnowa."""

from __future__ import annotations

import numpy as np
from scipy import stats


def report_normality(sample: np.ndarray, name: str, alpha: float = 0.05) -> None:
    """Wypisuje wyniki testu Shapiro-Wilka i Kolmogorowa-Smirnowa dla jednej zmiennej."""
    shapiro_stat, shapiro_p = stats.shapiro(sample)
    mean = float(np.mean(sample))
    std = float(np.std(sample, ddof=1))
    ks_stat, ks_p = stats.kstest(sample, "norm", args=(mean, std))

    print(f"\n{name}")
    print(f"Parametry probki: srednia = {mean:.4f}, odchylenie standardowe = {std:.4f}")
    print(f"Shapiro-Wilk: statystyka W = {shapiro_stat:.4f}, p-value = {shapiro_p:.6f}")
    print(f"Kolmogorow-Smirnow: statystyka D = {ks_stat:.4f}, p-value = {ks_p:.6f}")
    print(
        f"Wniosek Shapiro-Wilk (alpha={alpha}):",
        "odrzucamy normalnosc" if shapiro_p < alpha else "brak podstaw do odrzucenia normalnosci",
    )
    print(
        f"Wniosek Kolmogorow-Smirnow (alpha={alpha}):",
        "odrzucamy normalnosc" if ks_p < alpha else "brak podstaw do odrzucenia normalnosci",
    )


def main() -> None:
    """Generuje dane i wykonuje testy normalnosci dla kazdej zmiennej."""
    rng = np.random.default_rng(20260325)
    n = 150

    # Para zalezna liniowo z szumem normalnym.
    x_corr = rng.normal(0, 1, n)
    y_corr = 0.85 * x_corr + rng.normal(0, 0.35, n)

    # Para niezalezna: dwa niezalezne losowania z N(0, 1).
    x_uncorr = rng.normal(0, 1, n)
    y_uncorr = rng.normal(0, 1, n)

    print("=== Lista2/Zad6: Testy normalnosci (Shapiro-Wilk i Kolmogorow-Smirnow) ===")
    print("Zbior danych: dwie pary danych syntetycznych, n=150")
    print("Para skorelowana: x~N(0,1), y=0.85*x+epsilon, epsilon~N(0,0.35)")
    print("Para nieskorelowana: x~N(0,1), y~N(0,1), losowane niezaleznie")

    report_normality(x_corr, "Dane skorelowane - zmienna x")
    report_normality(y_corr, "Dane skorelowane - zmienna y")
    report_normality(x_uncorr, "Dane nieskorelowane - zmienna x")
    report_normality(y_uncorr, "Dane nieskorelowane - zmienna y")


if __name__ == "__main__":
    main()
