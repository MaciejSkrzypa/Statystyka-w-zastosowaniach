"""Zadanie 3: wspolczynnik korelacji rang Spearmana i porownanie z Pearsonem."""

from __future__ import annotations

from scipy import stats

from common import (
    PAIR_X,
    PAIR_Y,
    analysis_columns_description,
    complete_case,
    compare_coefficients,
    correlation_label,
    dataset_description,
    load_dataset,
)


def main() -> None:
    """Oblicza korelacje rang Spearmana i porownuje wynik z Pearsonem."""

    frame = load_dataset()
    data = complete_case(frame, [PAIR_X, PAIR_Y])
    pearson_r, _ = stats.pearsonr(data[PAIR_X], data[PAIR_Y])
    spearman_rho, spearman_p = stats.spearmanr(data[PAIR_X], data[PAIR_Y])
    label = correlation_label(spearman_rho)

    print("=== Lista 4, Zadanie 3 ===")
    print(dataset_description())
    print(analysis_columns_description())
    print(
        f"Analizowana para kolumn: {PAIR_X} vs {PAIR_Y}; "
        f"liczba obserwacji po usunieciu brakow = {len(data)}."
    )
    print(f"Pearson r = {pearson_r:.4f}")
    print(f"Spearman rho = {spearman_rho:.4f}")
    print(f"Wartosc p dla Spearmana = {spearman_p:.6f}")
    print(f"Ocena sily zaleznosci wg Spearmana: {label.name} ({label.description}).")
    print(compare_coefficients(pearson_r, spearman_rho))
    print(
        "Interpretacja: korelacja rang jest odporniejsza na wartosci odstajace "
        "i lepiej wychwytuje zaleznosc monotoniczna niz korelacja Pearsona."
    )


if __name__ == "__main__":
    main()

