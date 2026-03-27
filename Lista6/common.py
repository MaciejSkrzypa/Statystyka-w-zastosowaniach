"""Wspolne narzedzia dla zadan z Listy 6."""

from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib.pyplot as plt
import pandas as pd

LIST_DIR = Path(__file__).resolve().parent
DATA_DIR = LIST_DIR.parent / "Dane do listy 6"


def load_csv(filename: str) -> pd.DataFrame:
    """Wczytuje wskazany plik CSV z katalogu danych Listy 6."""

    return pd.read_csv(DATA_DIR / filename).dropna().reset_index(drop=True)


def save_figure(fig: plt.Figure, filename: str) -> Path:
    """Zapisuje wykres w katalogu listy i zwraca sciezke do pliku."""

    path = LIST_DIR / filename
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def print_frame(title: str, frame: pd.DataFrame, precision: int = 4) -> None:
    """Wypisuje tabele w czytelnym formacie tekstowym."""

    print(title)
    print(frame.to_string(index=False, float_format=lambda value: f"{value:.{precision}f}"))
