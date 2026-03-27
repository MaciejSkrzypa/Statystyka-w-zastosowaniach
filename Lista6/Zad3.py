"""Zadanie 3: preferencje filmowe."""

from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd
import scipy.stats as stats
import seaborn as sns

from common import load_csv, save_figure


def main() -> None:
    """Analizuje zwiazki miedzy wiekiem, plcia, gatunkiem filmu i ocena."""

    frame = load_csv("movie_preferences.csv")
    contingency = pd.crosstab(frame["AgeGroup"], frame["Genre"])
    chi2_stat, chi2_p, dof, expected = stats.chi2_contingency(contingency)

    ratings_female = frame.loc[frame["Gender"] == "Female", "Rating"]
    ratings_male = frame.loc[frame["Gender"] == "Male", "Rating"]
    t_stat, t_p = stats.ttest_ind(ratings_female, ratings_male, equal_var=False)

    age_order = {"<18": 0, "18-25": 1, "26-40": 2, "41+": 3}
    ordered_age = frame["AgeGroup"].map(age_order)
    spearman_rho, spearman_p = stats.spearmanr(ordered_age, frame["Rating"])

    print("=== Lista6/Zad3: Badanie preferencji filmowych ===")
    print(
        "Zbior danych: lokalny plik 'Dane do listy 6/movie_preferences.csv'; "
        "kolumny analizowane: AgeGroup, Gender, Genre, Rating."
    )
    print(f"Liczba obserwacji: {len(frame)}")
    print(f"Test chi-kwadrat AgeGroup x Genre: chi2 = {chi2_stat:.4f}, dof = {dof}, p-value = {chi2_p:.6f}")
    print(
        f"Test t dla ocen wg Gender: t = {t_stat:.4f}, p-value = {t_p:.6f}, "
        f"srednia Female = {ratings_female.mean():.3f}, srednia Male = {ratings_male.mean():.3f}"
    )
    print(
        f"Spearman dla uporzadkowanej AgeGroup i Rating: rho = {spearman_rho:.4f}, p-value = {spearman_p:.6f}"
    )

    fig, ax = plt.subplots(figsize=(7.0, 4.8))
    sns.heatmap(contingency, annot=True, fmt="d", cmap="Blues", cbar=False, ax=ax)
    ax.set_title("AgeGroup vs Genre")
    output_path = save_figure(fig, "Zad3_agegroup_genre_heatmap.png")
    print(f"Wykres zapisano w pliku: {output_path}")


if __name__ == "__main__":
    main()
