"""Zadanie 4: Porownanie metod pobierania probek."""

from __future__ import annotations

import numpy as np


def stats_line(name: str, sample: np.ndarray) -> str:
    """Buduje linie podsumowania statystyk dla metody probkowania."""
    mean = float(np.mean(sample))
    std = float(np.std(sample, ddof=1))
    return f"{name:16s} | srednia = {mean:8.4f} | odchylenie = {std:8.4f}"


def main() -> None:
    """Porownuje probkowanie losowe, warstwowe i systematyczne."""
    rng = np.random.default_rng(20260325)
    n_total = 100_000

    strata_sizes = [20_000, 30_000, 50_000]
    strata_means = [45.0, 52.0, 60.0]
    strata_std = [7.0, 8.0, 10.0]

    groups = []
    labels = []
    for idx, (size, mean, std) in enumerate(zip(strata_sizes, strata_means, strata_std), start=1):
        groups.append(rng.normal(loc=mean, scale=std, size=size))
        labels.extend([idx] * size)
    population = np.concatenate(groups)
    labels = np.array(labels)

    sample_size = 500
    simple_random = rng.choice(population, size=sample_size, replace=False)

    stratified_parts = []
    for idx, size in enumerate(strata_sizes, start=1):
        stratum = population[labels == idx]
        n_i = int(round(sample_size * size / n_total))
        stratified_parts.append(rng.choice(stratum, size=n_i, replace=False))
    stratified = np.concatenate(stratified_parts)

    step = n_total // sample_size
    start = rng.integers(0, step)
    systematic = population[start::step][:sample_size]

    pop_mean = float(np.mean(population))
    pop_std = float(np.std(population, ddof=0))
    print("=== Zadanie 4: Metody pobierania probek ===")
    print("Zbior danych: syntetyczna populacja warstwowa (3 warstwy), porownanie probkowania: losowe, warstwowe, systematyczne")
    print(f"Populacja         | srednia = {pop_mean:8.4f} | odchylenie = {pop_std:8.4f}")
    print(stats_line("Losowa prosta", simple_random))
    print(stats_line("Warstwowa", stratified))
    print(stats_line("Systematyczna", systematic))


if __name__ == "__main__":
    main()
