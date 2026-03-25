"""Zadanie 2: Wplyw liczebnosci proby na szerokosc przedzialu ufnosci."""

from __future__ import annotations

import numpy as np
from scipy import stats


def confidence_interval_95(sample: np.ndarray) -> tuple[float, float]:
    """Wyznacza dwustronny przedzial ufnosci 95% dla sredniej (rozklad t-Studenta)."""
    n = sample.size
    mean = float(np.mean(sample))
    std = float(np.std(sample, ddof=1))
    se = std / np.sqrt(n)
    t_crit = stats.t.ppf(0.975, df=n - 1)
    return mean - t_crit * se, mean + t_crit * se


def main() -> None:
    """Porownuje przedzialy ufnosci dla roznych liczebnosci prob."""
    rng = np.random.default_rng(20260325)
    population = rng.normal(loc=100, scale=15, size=100_000)
    pop_mean = float(np.mean(population))

    print("=== Zadanie 2: Liczebnosc proby a wiarygodnosc rezultatu ===")
    print("Zbior danych: syntetyczna populacja N(100, 15), liczebnosc 100000; proby: n=10, 50, 500")
    print(f"Populacja: srednia referencyjna = {pop_mean:.4f}")
    print("Rozmiar | Srednia proby | CI95 dolny | CI95 gorny | Szerokosc CI")

    for n in [10, 50, 500]:
        sample = rng.choice(population, size=n, replace=False)
        mean_s = float(np.mean(sample))
        ci_low, ci_high = confidence_interval_95(sample)
        width = ci_high - ci_low
        print(f"{n:7d} | {mean_s:13.4f} | {ci_low:10.4f} | {ci_high:11.4f} | {width:11.4f}")


if __name__ == "__main__":
    main()
