"""Zadanie 6: Istotnosc korelacji dla danych skorelowanych i nieskorelowanych."""

from __future__ import annotations

import numpy as np
from scipy import stats


def report(title: str, x: np.ndarray, y: np.ndarray) -> None:
    """Wypisuje wspolczynniki i p-value dla korelacji Pearsona i Spearmana."""
    pearson_r, pearson_p = stats.pearsonr(x, y)
    spearman_rho, spearman_p = stats.spearmanr(x, y)

    print(f"\n{title}")
    print(f"Pearson r = {pearson_r:.4f}, p-value = {pearson_p:.6f}")
    print(f"Spearman rho = {spearman_rho:.4f}, p-value = {spearman_p:.6f}")
    print(
        "Wniosek (alpha=0.05):",
        "korelacja istotna" if (pearson_p < 0.05 or spearman_p < 0.05) else "brak istotnej korelacji",
    )


def main() -> None:
    """Generuje dane skorelowane i nieskorelowane oraz porownuje wyniki testow."""
    rng = np.random.default_rng(20260325)
    n = 150

    x_corr = rng.normal(0, 1, n)
    y_corr = 0.85 * x_corr + rng.normal(0, 0.35, n)

    x_uncorr = rng.normal(0, 1, n)
    y_uncorr = rng.normal(0, 1, n)

    print("=== Lista2/Zad6: Korelacja dla danych skorelowanych i nieskorelowanych ===")
    print("Zbior danych: dwie pary danych syntetycznych (skorelowane i nieskorelowane), n=150")
    report("Dane skorelowane:", x_corr, y_corr)
    report("Dane nieskorelowane:", x_uncorr, y_uncorr)


if __name__ == "__main__":
    main()
