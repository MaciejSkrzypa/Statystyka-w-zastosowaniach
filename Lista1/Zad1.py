"""Zadanie 1: Wplyw liczebnosci proby na oszacowania sredniej i odchylenia."""

from __future__ import annotations

import numpy as np


def sample_stats(sample: np.ndarray) -> tuple[float, float]:
    """Zwraca srednia i odchylenie standardowe proby (ddof=1)."""
    return float(np.mean(sample)), float(np.std(sample, ddof=1))


def main() -> None:
    """Generuje populacje i porownuje statystyki dla prob o roznych rozmiarach."""
    rng = np.random.default_rng(20260325)
    population = rng.normal(loc=50, scale=10, size=100_000)
    pop_mean = float(np.mean(population))
    pop_std = float(np.std(population, ddof=0))

    print("=== Zadanie 1: Proba a populacja ===")
    print("Zbior danych: syntetyczna populacja N(50, 10), liczebnosc 100000; proby: n=10, 50, 1000")
    print(f"Populacja: srednia = {pop_mean:.4f}, odchylenie = {pop_std:.4f}")
    print("Rozmiar | Srednia proby | Odchylenie proby | Blad sredniej | Blad odchylenia")

    for n in [10, 50, 1000]:
        sample = rng.choice(population, size=n, replace=False)
        mean_s, std_s = sample_stats(sample)
        print(
            f"{n:7d} | {mean_s:13.4f} | {std_s:16.4f} | "
            f"{abs(mean_s - pop_mean):12.4f} | {abs(std_s - pop_std):15.4f}"
        )


if __name__ == "__main__":
    main()
