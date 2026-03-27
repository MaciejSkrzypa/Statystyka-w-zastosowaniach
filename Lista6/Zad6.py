"""Zadanie 6: analiza kampanii reklamowej."""

from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd
import scipy.stats as stats
import seaborn as sns
import statsmodels.formula.api as smf
from statsmodels.stats.anova import anova_lm

from common import load_csv, save_figure


def main() -> None:
    """Porownuje skutecznosc typow reklam, pory dnia i konwersje."""

    frame = load_csv("ad_campaign_data.csv")
    click_type_model = smf.ols("Clicks ~ C(AdType)", data=frame).fit()
    click_time_model = smf.ols("Clicks ~ C(TimeOfDay)", data=frame).fit()
    click_type_anova = anova_lm(click_type_model, typ=2)
    click_time_anova = anova_lm(click_time_model, typ=2)
    contingency = pd.crosstab(frame["AdType"], frame["Conversion"])
    chi2_stat, chi2_p, dof, _ = stats.chi2_contingency(contingency)

    conversion_rates = frame.groupby("AdType", as_index=False)["Conversion"].mean()

    print("=== Lista6/Zad6: Analiza wynikow kampanii reklamowej ===")
    print(
        "Zbior danych: lokalny plik 'Dane do listy 6/ad_campaign_data.csv'; "
        "kolumny analizowane: AdType, TimeOfDay, Clicks, Conversion."
    )
    print(f"Liczba obserwacji: {len(frame)}")
    print("ANOVA dla Clicks wg AdType:")
    print(click_type_anova.to_string(float_format=lambda value: f"{value:.6f}"))
    print("ANOVA dla Clicks wg TimeOfDay:")
    print(click_time_anova.to_string(float_format=lambda value: f"{value:.6f}"))
    print(f"Test chi-kwadrat AdType x Conversion: chi2 = {chi2_stat:.4f}, dof = {dof}, p-value = {chi2_p:.6f}")
    print("Sredni wskaznik konwersji wg typu reklamy:")
    print(conversion_rates.to_string(index=False, float_format=lambda value: f"{value:.4f}"))

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.4))
    sns.barplot(data=frame, x="AdType", y="Clicks", ax=axes[0], color="#a8dadc")
    axes[0].set_title("Srednia liczba klikniec wg AdType")
    sns.barplot(data=conversion_rates, x="AdType", y="Conversion", ax=axes[1], color="#e9c46a")
    axes[1].set_title("Srednia konwersja wg AdType")
    axes[1].set_ylabel("Conversion rate")
    output_path = save_figure(fig, "Zad6_adtype_effectiveness.png")
    print(f"Wykres zapisano w pliku: {output_path}")


if __name__ == "__main__":
    main()
