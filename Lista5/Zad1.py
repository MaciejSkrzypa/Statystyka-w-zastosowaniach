"""Zadanie 1: regresja wielokrotna dla cen mieszkan."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import statsmodels.api as sm
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

from common import load_boston, rmse, save_figure


def main() -> None:
    """Buduje pelny model regresji dla danych BostonHousing i raportuje dopasowanie."""

    frame = load_boston()
    target_name = "medv"
    features = frame.drop(columns=[target_name])
    target = frame[target_name]

    ols_model = sm.OLS(target, sm.add_constant(features, has_constant="add")).fit()
    sklearn_model = LinearRegression().fit(features, target)
    predictions = sklearn_model.predict(features)

    print("=== Lista5/Zad1: Regresja wielokrotna na danych o cenach mieszkan ===")
    print(
        "Zbior danych: lokalny plik 'Dane do listy 5/BostonHousing.csv'; "
        "modelowana kolumna: medv; predyktory: wszystkie pozostale kolumny."
    )
    print(f"Liczba obserwacji po wczytaniu: {len(frame)}")
    print(f"Liczba predyktorow: {features.shape[1]}")
    print(f"R2 (sklearn) = {r2_score(target, predictions):.4f}")
    print(f"R2 (OLS) = {ols_model.rsquared:.4f}")
    print(f"R2 skorygowane = {ols_model.rsquared_adj:.4f}")
    print(f"RMSE na calym zbiorze = {rmse(target, predictions):.4f}")
    print("Najwieksze co do wartosci bezwzglednej wspolczynniki OLS:")
    print(ols_model.params.drop("const").abs().sort_values(ascending=False).head(5).to_string())

    fig, ax = plt.subplots(figsize=(6.8, 4.8))
    ax.scatter(target, predictions, alpha=0.75, color="#2a6f97", edgecolor="white", linewidth=0.5)
    diagonal = np.linspace(target.min(), target.max(), 100)
    ax.plot(diagonal, diagonal, color="#d62828", linestyle="--", linewidth=2, label="idealne dopasowanie")
    ax.set_title("Ceny rzeczywiste vs przewidywane")
    ax.set_xlabel("Rzeczywiste medv")
    ax.set_ylabel("Przewidywane medv")
    ax.legend()
    output_path = save_figure(fig, "Zad1_pred_vs_actual.png")
    print(f"Wykres zapisano w pliku: {output_path}")


if __name__ == "__main__":
    main()
