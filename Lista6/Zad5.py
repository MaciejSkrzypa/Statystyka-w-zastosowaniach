"""Zadanie 5: zadowolenie z zycia a aktywnosc fizyczna."""

from __future__ import annotations

import matplotlib.pyplot as plt
import scipy.stats as stats
import statsmodels.formula.api as smf
from statsmodels.stats.anova import anova_lm

from common import load_csv, save_figure


def main() -> None:
    """Analizuje korelacje, roznice miedzy zatrudnieniem i interakcje z plcia."""

    frame = load_csv("life_satisfaction_activity.csv")
    pearson_r, pearson_p = stats.pearsonr(frame["ActivityDays"], frame["LifeSatisfaction"])

    employed = frame.loc[frame["EmploymentStatus"] == "Employed", "LifeSatisfaction"]
    unemployed = frame.loc[frame["EmploymentStatus"] == "Unemployed", "LifeSatisfaction"]
    t_stat, t_p = stats.ttest_ind(employed, unemployed, equal_var=False)

    interaction_model = smf.ols(
        "LifeSatisfaction ~ ActivityDays * C(Gender) + C(EmploymentStatus)", data=frame
    ).fit()
    interaction_anova = anova_lm(interaction_model, typ=2)

    print("=== Lista6/Zad5: Zadowolenie z zycia a aktywnosc fizyczna ===")
    print(
        "Zbior danych: lokalny plik 'Dane do listy 6/life_satisfaction_activity.csv'; "
        "kolumny analizowane: ActivityDays, LifeSatisfaction, Gender, EmploymentStatus."
    )
    print(f"Liczba obserwacji: {len(frame)}")
    print(
        f"Korelacja ActivityDays vs LifeSatisfaction: r = {pearson_r:.4f}, p-value = {pearson_p:.6f}"
    )
    print(
        f"Test t dla EmploymentStatus: t = {t_stat:.4f}, p-value = {t_p:.6f}, "
        f"srednia Employed = {employed.mean():.3f}, srednia Unemployed = {unemployed.mean():.3f}"
    )
    print("ANOVA dla modelu z interakcja ActivityDays x Gender:")
    print(interaction_anova.to_string(float_format=lambda value: f"{value:.6f}"))

    fig, ax = plt.subplots(figsize=(6.8, 4.8))
    colors = frame["Gender"].map({"Female": "#e76f51", "Male": "#457b9d"})
    ax.scatter(frame["ActivityDays"], frame["LifeSatisfaction"], c=colors, alpha=0.75, edgecolor="white", linewidth=0.4)
    ax.set_title("Aktywnosc fizyczna a zadowolenie z zycia")
    ax.set_xlabel("ActivityDays")
    ax.set_ylabel("LifeSatisfaction")
    output_path = save_figure(fig, "Zad5_activity_vs_satisfaction.png")
    print(f"Wykres zapisano w pliku: {output_path}")


if __name__ == "__main__":
    main()
