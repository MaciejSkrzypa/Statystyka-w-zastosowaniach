"""Zadanie 1: przygotowanie zbioru danych i statystyki opisowe."""

from __future__ import annotations

from common import (
    ANALYSIS_COLUMNS,
    analysis_columns_description,
    dataset_description,
    descriptive_summary,
    format_frame,
    load_dataset,
    missing_summary,
)


def main() -> None:
    """Wczytuje syntetyczny zbior danych i wypisuje podstawowe statystyki."""

    frame = load_dataset()

    print("=== Lista 4, Zadanie 1 ===")
    print(dataset_description())
    print(analysis_columns_description())
    print(f"Liczba obserwacji: {len(frame)}")

    print("\nBraki danych:")
    print(format_frame(missing_summary(frame)))

    print("\nStatystyki opisowe:")
    summary = descriptive_summary(frame, ANALYSIS_COLUMNS)
    print(format_frame(summary))


if __name__ == "__main__":
    main()

