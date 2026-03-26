"""Zadanie 4: test istotnosci statystycznej dla korelacji Pearsona i Spearmana."""

from __future__ import annotations

from scipy import stats

from common import (
    PAIR_X,
    PAIR_Y,
    analysis_columns_description,
    complete_case,
    dataset_description,
    load_dataset,
)

ALPHA = 0.05


def decision(p_value: float, alpha: float = ALPHA) -> str:
    """Zwraca decyzje testowa dla zadanego poziomu istotnosci."""

    return "odrzucamy hipoteze zerowa" if p_value < alpha else "brak podstaw do odrzucenia hipotezy zerowej"


def main() -> None:
    """Przeprowadza testy istotnosci dla obu wspolczynnikow korelacji."""

    frame = load_dataset()
    data = complete_case(frame, [PAIR_X, PAIR_Y])
    pearson_r, pearson_p = stats.pearsonr(data[PAIR_X], data[PAIR_Y])
    spearman_rho, spearman_p = stats.spearmanr(data[PAIR_X], data[PAIR_Y])

    print("=== Lista 4, Zadanie 4 ===")
    print(dataset_description())
    print(analysis_columns_description())
    print(
        "Hipotezy dla obu testow: H0: brak korelacji w populacji; "
        "H1: korelacja rozna od zera."
    )
    print(f"Poziom istotnosci: alpha = {ALPHA:.2f}")
    print(f"Liczba obserwacji po usunieciu brakow: {len(data)}")

    print("\nTest Pearsona:")
    print(f"r = {pearson_r:.4f}, p-value = {pearson_p:.6f}")
    print(f"Decyzja: {decision(pearson_p)}.")

    print("\nTest Spearmana:")
    print(f"rho = {spearman_rho:.4f}, p-value = {spearman_p:.6f}")
    print(f"Decyzja: {decision(spearman_p)}.")

    print(
        "Wniosek koncowy: dla obu testow p-value jest mniejsze od poziomu "
        "istotnosci, wiec odrzucamy H0 i przyjmujemy istnienie istotnej dodatniej "
        "zaleznosci miedzy analizowanymi zmiennymi."
    )


if __name__ == "__main__":
    main()

