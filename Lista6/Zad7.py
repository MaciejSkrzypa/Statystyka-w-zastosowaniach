"""Zadanie 7: analiza danych medycznych."""

from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd
import scipy.stats as stats
import statsmodels.formula.api as smf
from statsmodels.stats.anova import anova_lm

from common import load_csv, save_figure


def main() -> None:
    """Bada wplyw palenia, BMI i wieku na poziom cholesterolu."""

    frame = load_csv("medical_data.csv")
    smokers = frame.loc[frame["Smoker"] == "Yes", "Cholesterol"]
    nonsmokers = frame.loc[frame["Smoker"] == "No", "Cholesterol"]
    t_stat, t_p = stats.ttest_ind(smokers, nonsmokers, equal_var=False)
    pearson_r, pearson_p = stats.pearsonr(frame["BMI"], frame["Cholesterol"])

    frame = frame.assign(
        AgeGroup=pd.cut(
            frame["Age"],
            bins=[17, 34, 49, 80],
            labels=["18-34", "35-49", "50+"],
            include_lowest=True,
        )
    )
    age_model = smf.ols("Cholesterol ~ C(AgeGroup)", data=frame).fit()
    age_anova = anova_lm(age_model, typ=2)

    print("=== Lista6/Zad7: Analiza danych medycznych ===")
    print(
        "Zbior danych: lokalny plik 'Dane do listy 6/medical_data.csv'; "
        "kolumny analizowane: Cholesterol, BMI, Smoker, Age, Gender."
    )
    print(f"Liczba obserwacji: {len(frame)}")
    print(
        f"Test t dla Smoker: t = {t_stat:.4f}, p-value = {t_p:.6f}, "
        f"srednia palacych = {smokers.mean():.3f}, srednia niepalacych = {nonsmokers.mean():.3f}"
    )
    print(
        f"Korelacja BMI vs Cholesterol: r = {pearson_r:.4f}, p-value = {pearson_p:.6f}"
    )
    print("ANOVA dla Cholesterol wg grup wieku:")
    print(age_anova.to_string(float_format=lambda value: f"{value:.6f}"))

    fig, ax = plt.subplots(figsize=(6.8, 4.8))
    colors = frame["Smoker"].map({"Yes": "#e76f51", "No": "#2a9d8f"})
    ax.scatter(frame["BMI"], frame["Cholesterol"], c=colors, alpha=0.75, edgecolor="white", linewidth=0.4)
    ax.set_title("BMI a cholesterol")
    ax.set_xlabel("BMI")
    ax.set_ylabel("Cholesterol")
    output_path = save_figure(fig, "Zad7_bmi_vs_cholesterol.png")
    print(f"Wykres zapisano w pliku: {output_path}")


if __name__ == "__main__":
    main()
