"""Zadanie 7: Test istotnosci dla wspolczynnika korelacji Pearsona i Spearmana."""

from __future__ import annotations

import numpy as np
from scipy import stats


def report(name: str, x: np.ndarray, y: np.ndarray) -> None:
    """Wypisuje wspolczynniki korelacji i ich istotnosc statystyczna."""
    pearson_r, pearson_p = stats.pearsonr(x, y)
    spearman_rho, spearman_p = stats.spearmanr(x, y)

    print(f"\n{name}")
    print(f"Pearson r = {pearson_r:.4f}, p-value = {pearson_p:.6f}")
    print(f"Spearman rho = {spearman_rho:.4f}, p-value = {spearman_p:.6f}")
    print(
        "Wniosek (alpha=0.05):",
        "korelacja istotna" if (pearson_p < 0.05 or spearman_p < 0.05) else "brak istotnej korelacji",
    )


def main() -> None:
    """Porownuje przypadek danych skorelowanych i nieskorelowanych."""
    rng = np.random.default_rng(20260325)
    n = 150

    x_corr = rng.normal(0, 1, n)
    y_corr = 0.8 * x_corr + rng.normal(0, 0.4, n)

    x_uncorr = rng.normal(0, 1, n)
    y_uncorr = rng.normal(0, 1, n)

    print("=== Lista2/Zad7: Istotnosc korelacji ===")
    print("Zbior danych: dwie pary danych syntetycznych, n=150")
    print("Para skorelowana: x~N(0,1), y=0.8*x+epsilon, epsilon~N(0,0.4)")
    print("Para nieskorelowana: x~N(0,1), y~N(0,1), losowane niezaleznie")
    report("Dane skorelowane:", x_corr, y_corr)
    report("Dane nieskorelowane:", x_uncorr, y_uncorr)


if __name__ == "__main__":
    main()
