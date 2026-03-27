"""Zadanie 4: analiza reszt i weryfikacja zalozen regresji."""

from __future__ import annotations

import matplotlib.pyplot as plt
import scipy.stats as stats
import statsmodels.api as sm
from statsmodels.stats.diagnostic import het_breuschpagan
from statsmodels.stats.stattools import durbin_watson

from common import add_constant, backward_elimination, load_boston, save_figure


def main() -> None:
    """Bada normalnosc, homoscedastycznosc i niezaleznosc reszt wybranego modelu."""

    frame = load_boston()
    target = frame["medv"]
    features = frame.drop(columns=["medv"])
    selected, _, model = backward_elimination(features, target)

    fitted = model.predict(add_constant(features[selected]))
    residuals = target - fitted

    shapiro_stat, shapiro_p = stats.shapiro(residuals)
    bp_stat, bp_p, _, _ = het_breuschpagan(residuals, add_constant(features[selected]))
    dw_stat = durbin_watson(residuals)

    print("=== Lista5/Zad4: Weryfikacja zalozen regresji - analiza reszt ===")
    print(
        "Zbior danych: lokalny plik 'Dane do listy 5/BostonHousing.csv'; "
        f"analizowany model po eliminacji wstecznej, predyktory: {', '.join(selected)}."
    )
    print(f"Liczba obserwacji: {len(frame)}")
    print(f"Test Shapiro-Wilka dla reszt: W = {shapiro_stat:.4f}, p-value = {shapiro_p:.6f}")
    print(f"Test Breuscha-Pagana: statystyka = {bp_stat:.4f}, p-value = {bp_p:.6f}")
    print(f"Statystyka Durbin-Watsona = {dw_stat:.4f}")
    print(
        "Wniosek o normalnosci reszt:",
        "brak podstaw do odrzucenia normalnosci" if shapiro_p >= 0.05 else "reszty odbiegaja od normalnosci",
    )
    print(
        "Wniosek o homoscedastycznosci:",
        "brak podstaw do stwierdzenia heteroskedastycznosci" if bp_p >= 0.05 else "wystepuje heteroskedastycznosc",
    )
    print(
        "Wniosek o niezaleznosci reszt:",
        "statystyka Durbin-Watsona jest bliska 2" if 1.5 <= dw_stat <= 2.5 else "mozliwa autokorelacja reszt",
    )

    fig, ax = plt.subplots(figsize=(6.8, 4.8))
    ax.scatter(fitted, residuals, alpha=0.75, color="#1d3557", edgecolor="white", linewidth=0.5)
    ax.axhline(0.0, color="#d62828", linestyle="--", linewidth=2)
    ax.set_title("Reszty vs przewidywania")
    ax.set_xlabel("Wartosci przewidywane")
    ax.set_ylabel("Reszty")
    output_path = save_figure(fig, "Zad4_residuals_vs_fitted.png")
    print(f"Wykres zapisano w pliku: {output_path}")


if __name__ == "__main__":
    main()
