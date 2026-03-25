"""Zadanie 3: Reprezentatywnosc proby w populacji bimodalnej."""

from __future__ import annotations

import numpy as np


def summarize(data: np.ndarray) -> tuple[float, float]:
    """Zwraca srednia i odchylenie standardowe."""
    return float(np.mean(data)), float(np.std(data, ddof=1))


def main() -> None:
    """Porownuje probe losowa i probe niereprezentatywna z parametrami populacji."""
    rng = np.random.default_rng(20260325)
    subgroup_a = rng.normal(loc=40, scale=5, size=50_000)
    subgroup_b = rng.normal(loc=60, scale=5, size=50_000)
    population = np.concatenate([subgroup_a, subgroup_b])

    pop_mean, pop_std = summarize(population)
    random_sample = rng.choice(population, size=400, replace=False)
    biased_sample = rng.choice(subgroup_a, size=400, replace=False)

    rand_mean, rand_std = summarize(random_sample)
    bias_mean, bias_std = summarize(biased_sample)

    print("=== Zadanie 3: Reprezentatywnosc proby ===")
    print("Zbior danych: populacja bimodalna (podzbior A: N(40,5), podzbior B: N(60,5)); analiza proby losowej i stronniczej")
    print(f"Populacja    -> srednia = {pop_mean:.4f}, odchylenie = {pop_std:.4f}")
    print(f"Proba losowa -> srednia = {rand_mean:.4f}, odchylenie = {rand_std:.4f}")
    print(f"Proba stronnicza (tylko podzbior A) -> srednia = {bias_mean:.4f}, odchylenie = {bias_std:.4f}")
    print(
        "Roznica wzgledem populacji: "
        f"proba losowa (srednia {abs(rand_mean-pop_mean):.4f}), "
        f"proba stronnicza (srednia {abs(bias_mean-pop_mean):.4f})"
    )


if __name__ == "__main__":
    main()
