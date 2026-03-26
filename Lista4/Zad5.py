"""Zadanie 5: macierz korelacji i mapa cieplna dla calego zbioru danych."""

from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib

from common import (
    ANALYSIS_COLUMNS,
    analysis_columns_description,
    dataset_description,
    format_frame,
    load_dataset,
)

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import seaborn as sns

SCRIPT_DIR = Path(__file__).resolve().parent
OUTPUT_PATH = SCRIPT_DIR / "Zad5_heatmap.png"


def main() -> None:
    """Oblicza macierz korelacji i zapisuje jej wizualizacje."""

    frame = load_dataset()
    corr = frame[ANALYSIS_COLUMNS].corr(method="pearson")

    print("=== Lista 4, Zadanie 5 ===")
    print(dataset_description())
    print(analysis_columns_description())
    print(
        "Macierz korelacji obliczono dla wszystkich czterech kolumn ilosciowych "
        "z wykorzystaniem korelacji Pearsona i dopasowania pairwise complete observations."
    )
    print("\nMacierz korelacji:")
    print(format_frame(corr.round(3)))

    fig, ax = plt.subplots(figsize=(6.8, 5.4))
    sns.heatmap(
        corr,
        annot=True,
        fmt=".2f",
        cmap="RdBu_r",
        vmin=-1.0,
        vmax=1.0,
        center=0.0,
        square=True,
        linewidths=0.6,
        cbar_kws={"label": "wspolczynnik korelacji"},
        ax=ax,
    )
    ax.set_title("Macierz korelacji dla zbioru syntetycznego")
    fig.tight_layout()
    fig.savefig(OUTPUT_PATH, dpi=160)
    plt.close(fig)

    print(f"Mapa cieplna zapisano w pliku: {OUTPUT_PATH}")
    print(
        "Wniosek: najsilniejsza zaleznosc dotyczy pary experience_years i wellbeing_score, "
        "a silna dodatnia korelacja wystepuje tez miedzy experience_years i income_pln."
    )


if __name__ == "__main__":
    main()
