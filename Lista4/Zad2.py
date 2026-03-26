"""Zadanie 2: wspolczynnik korelacji Pearsona i wykres rozrzutu."""

from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib
import numpy as np
from scipy import stats

from common import (
    PAIR_X,
    PAIR_Y,
    analysis_columns_description,
    complete_case,
    correlation_label,
    dataset_description,
    load_dataset,
)

matplotlib.use("Agg")

import matplotlib.pyplot as plt

SCRIPT_DIR = Path(__file__).resolve().parent
OUTPUT_PATH = SCRIPT_DIR / "Zad2_scatter.png"


def main() -> None:
    """Oblicza korelacje Pearsona i zapisuje wykres rozrzutu."""

    frame = load_dataset()
    data = complete_case(frame, [PAIR_X, PAIR_Y])
    pearson_r, pearson_p = stats.pearsonr(data[PAIR_X], data[PAIR_Y])
    label = correlation_label(pearson_r)

    print("=== Lista 4, Zadanie 2 ===")
    print(dataset_description())
    print(analysis_columns_description())
    print(
        f"Analizowana para kolumn: {PAIR_X} vs {PAIR_Y}; "
        f"liczba obserwacji po usunieciu brakow = {len(data)}."
    )
    print(f"Wspolczynnik korelacji Pearsona r = {pearson_r:.4f}")
    print(f"Wartosc p = {pearson_p:.6f}")
    print(f"Ocena sily zaleznosci: {label.name} ({label.description}).")

    x = data[PAIR_X].to_numpy()
    y = data[PAIR_Y].to_numpy()
    slope, intercept = np.polyfit(x, y, deg=1)
    grid = np.linspace(x.min(), x.max(), 200)

    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    ax.scatter(x, y, alpha=0.75, color="#1f77b4", edgecolor="white", linewidth=0.5)
    ax.plot(grid, slope * grid + intercept, color="#d62728", linewidth=2.0, label="linia regresji")
    ax.set_title("Zaleznosc miedzy doswiadczeniem i dochodem")
    ax.set_xlabel("experience_years")
    ax.set_ylabel("income_pln")
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUTPUT_PATH, dpi=160)
    plt.close(fig)

    print(f"Wykres rozrzutu zapisano w pliku: {OUTPUT_PATH}")
    print(f"Wniosek praktyczny: {label.description.capitalize()}.")


if __name__ == "__main__":
    main()
