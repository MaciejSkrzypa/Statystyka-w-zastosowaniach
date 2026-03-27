"""Zadanie 2: konsumpcja kawy a poziom stresu."""

from __future__ import annotations

import matplotlib.pyplot as plt
import scipy.stats as stats
import statsmodels.formula.api as smf
from statsmodels.stats.anova import anova_lm

from common import load_csv, save_figure


def main() -> None:
    """Sprawdza zwiazek kawy ze stresem, roznice miedzy miejscami pracy i efekt wieku."""

    frame = load_csv("coffee_stress_data.csv")
    pearson_r, pearson_p = stats.pearsonr(frame["CoffeeCupsPerDay"], frame["StressLevel"])

    stress_model = smf.ols("StressLevel ~ C(Workplace)", data=frame).fit()
    stress_anova = anova_lm(stress_model, typ=2)

    coffee_age_model = smf.ols("CoffeeCupsPerDay ~ Age", data=frame).fit()

    print("=== Lista6/Zad2: Konsumpcja kawy a poziom stresu ===")
    print(
        "Zbior danych: lokalny plik 'Dane do listy 6/coffee_stress_data.csv'; "
        "kolumny analizowane: CoffeeCupsPerDay, StressLevel, Workplace, Age."
    )
    print(f"Liczba obserwacji: {len(frame)}")
    print(
        f"Korelacja CoffeeCupsPerDay vs StressLevel: r = {pearson_r:.4f}, p-value = {pearson_p:.6f}"
    )
    print("ANOVA dla poziomu stresu wedlug Workplace:")
    print(stress_anova.to_string(float_format=lambda value: f"{value:.6f}"))
    print(
        f"Regresja CoffeeCupsPerDay ~ Age: wspolczynnik Age = {coffee_age_model.params['Age']:.4f}, "
        f"p-value = {coffee_age_model.pvalues['Age']:.6f}, R2 = {coffee_age_model.rsquared:.4f}"
    )

    fig, ax = plt.subplots(figsize=(6.8, 4.8))
    palette = {"Office": "#457b9d", "Home": "#2a9d8f", "Other": "#f4a261"}
    ax.scatter(
        frame["CoffeeCupsPerDay"],
        frame["StressLevel"],
        c=frame["Workplace"].map(palette),
        alpha=0.75,
        edgecolor="white",
        linewidth=0.4,
    )
    ax.set_title("Kawa a stres")
    ax.set_xlabel("CoffeeCupsPerDay")
    ax.set_ylabel("StressLevel")
    output_path = save_figure(fig, "Zad2_coffee_vs_stress.png")
    print(f"Wykres zapisano w pliku: {output_path}")


if __name__ == "__main__":
    main()
