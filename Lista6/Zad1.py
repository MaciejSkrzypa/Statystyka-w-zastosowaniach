"""Zadanie 1: analiza wynikow uczniow."""

from __future__ import annotations

import matplotlib.pyplot as plt
import scipy.stats as stats
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.stats.anova import anova_lm

from common import load_csv, save_figure


def main() -> None:
    """Bada roznice wynikow miedzy plciami, korelacje z nauka i interakcje z SES."""

    frame = load_csv("student_math_scores.csv")
    females = frame.loc[frame["Gender"] == "Female", "MathScore"]
    males = frame.loc[frame["Gender"] == "Male", "MathScore"]
    t_stat, t_p = stats.ttest_ind(females, males, equal_var=False)
    pearson_r, pearson_p = stats.pearsonr(frame["StudyHours"], frame["MathScore"])

    interaction_model = smf.ols(
        "MathScore ~ StudyHours + C(Gender) * C(SocioeconomicStatus)", data=frame
    ).fit()
    interaction_anova = anova_lm(interaction_model, typ=2)

    print("=== Lista6/Zad1: Analiza wynikow uczniow ===")
    print(
        "Zbior danych: lokalny plik 'Dane do listy 6/student_math_scores.csv'; "
        "kolumny analizowane: Gender, SocioeconomicStatus, StudyHours, MathScore."
    )
    print(f"Liczba obserwacji: {len(frame)}")
    print(
        f"Test t (Female vs Male): t = {t_stat:.4f}, p-value = {t_p:.6f}, "
        f"srednia Female = {females.mean():.3f}, srednia Male = {males.mean():.3f}"
    )
    print(
        f"Korelacja StudyHours vs MathScore: r = {pearson_r:.4f}, p-value = {pearson_p:.6f}"
    )
    print("ANOVA dla modelu z interakcja Gender x SocioeconomicStatus:")
    print(interaction_anova.to_string(float_format=lambda value: f"{value:.6f}"))

    fig, ax = plt.subplots(figsize=(6.8, 4.8))
    colors = frame["Gender"].map({"Female": "#e76f51", "Male": "#457b9d"})
    ax.scatter(frame["StudyHours"], frame["MathScore"], c=colors, alpha=0.75, edgecolor="white", linewidth=0.4)
    ax.set_title("Godziny nauki a wynik z matematyki")
    ax.set_xlabel("StudyHours")
    ax.set_ylabel("MathScore")
    output_path = save_figure(fig, "Zad1_studyhours_vs_score.png")
    print(f"Wykres zapisano w pliku: {output_path}")


if __name__ == "__main__":
    main()
