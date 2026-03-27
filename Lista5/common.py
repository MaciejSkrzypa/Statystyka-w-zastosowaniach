"""Wspolne narzedzia dla zadan z Listy 5."""

from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor

LIST_DIR = Path(__file__).resolve().parent
DATA_DIR = LIST_DIR.parent / "Dane do listy 5"


def load_boston() -> pd.DataFrame:
    """Wczytuje dane o cenach mieszkan z lokalnego pliku CSV."""

    frame = pd.read_csv(DATA_DIR / "BostonHousing.csv")
    frame["chas"] = pd.to_numeric(frame["chas"], errors="coerce")
    return frame.apply(pd.to_numeric, errors="coerce").dropna().reset_index(drop=True)


def load_insurance() -> pd.DataFrame:
    """Wczytuje dane insurance z lokalnego pliku CSV."""

    return pd.read_csv(DATA_DIR / "insurance.csv").dropna().reset_index(drop=True)


def load_mpg() -> pd.DataFrame:
    """Wczytuje dane mpg i usuwa brakujace wartosci."""

    frame = pd.read_csv(DATA_DIR / "mpg.csv")
    frame["horsepower"] = pd.to_numeric(frame["horsepower"], errors="coerce")
    return frame.dropna().reset_index(drop=True)


def save_figure(fig: plt.Figure, filename: str) -> Path:
    """Zapisuje wykres w katalogu listy i zwraca sciezke do pliku."""

    path = LIST_DIR / filename
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def rmse(y_true: pd.Series | np.ndarray, y_pred: pd.Series | np.ndarray) -> float:
    """Liczy pierwiastek z bledu sredniokwadratowego."""

    return float(np.sqrt(np.mean((np.asarray(y_true) - np.asarray(y_pred)) ** 2)))


def add_constant(features: pd.DataFrame) -> pd.DataFrame:
    """Dodaje wyraz wolny do macierzy projektujacej."""

    return sm.add_constant(features, has_constant="add")


def vif_table(features: pd.DataFrame) -> pd.DataFrame:
    """Buduje tabele VIF dla przekazanych predyktorow."""

    design = add_constant(features)
    rows: list[dict[str, float | str]] = []
    for idx, name in enumerate(design.columns):
        if name == "const":
            continue
        rows.append(
            {
                "variable": name,
                "vif": float(variance_inflation_factor(design.values, idx)),
            }
        )
    return pd.DataFrame(rows).sort_values("vif", ascending=False).reset_index(drop=True)


def backward_elimination(
    features: pd.DataFrame, target: pd.Series, alpha: float = 0.05
) -> tuple[list[str], list[dict[str, float | str]], sm.regression.linear_model.RegressionResultsWrapper]:
    """Usuwa kolejno najmniej istotne predyktory na podstawie p-value."""

    selected = list(features.columns)
    steps: list[dict[str, float | str]] = []

    while True:
        model = sm.OLS(target, add_constant(features[selected])).fit()
        p_values = model.pvalues.drop("const", errors="ignore")
        worst_name = str(p_values.idxmax())
        worst_p = float(p_values.max())
        steps.append({"variable": worst_name, "p_value": worst_p, "kept": worst_p <= alpha})
        if worst_p <= alpha:
            return selected, steps, model
        selected.remove(worst_name)


def print_frame(title: str, frame: pd.DataFrame, precision: int = 4) -> None:
    """Wypisuje tabele w czytelnym formacie tekstowym."""

    print(title)
    print(frame.to_string(index=False, float_format=lambda value: f"{value:.{precision}f}"))
