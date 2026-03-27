"""Zadanie 4: analiza danych o zakupach online."""

from __future__ import annotations

import matplotlib.pyplot as plt
import scipy.stats as stats
import seaborn as sns
import statsmodels.formula.api as smf
from statsmodels.stats.anova import anova_lm

from common import load_csv, save_figure


def main() -> None:
    """Bada zaleznosci miedzy metoda platnosci, kategoria, koszykiem i wartoscia zamowienia."""

    frame = load_csv("online_shopping_data.csv")
    payment_model = smf.ols("OrderValue ~ C(PaymentMethod)", data=frame).fit()
    category_model = smf.ols("OrderValue ~ C(Category)", data=frame).fit()
    payment_anova = anova_lm(payment_model, typ=2)
    category_anova = anova_lm(category_model, typ=2)
    pearson_r, pearson_p = stats.pearsonr(frame["ItemsInCart"], frame["OrderValue"])

    print("=== Lista6/Zad4: Zakupy online ===")
    print(
        "Zbior danych: lokalny plik 'Dane do listy 6/online_shopping_data.csv'; "
        "kolumny analizowane: Category, PaymentMethod, ItemsInCart, OrderValue."
    )
    print(f"Liczba obserwacji: {len(frame)}")
    print("ANOVA dla OrderValue wg PaymentMethod:")
    print(payment_anova.to_string(float_format=lambda value: f"{value:.6f}"))
    print("ANOVA dla OrderValue wg Category:")
    print(category_anova.to_string(float_format=lambda value: f"{value:.6f}"))
    print(
        f"Korelacja ItemsInCart vs OrderValue: r = {pearson_r:.4f}, p-value = {pearson_p:.6f}"
    )

    fig, ax = plt.subplots(figsize=(7.0, 4.8))
    sns.boxplot(data=frame, x="PaymentMethod", y="OrderValue", ax=ax, color="#a8dadc")
    ax.set_title("OrderValue wg PaymentMethod")
    output_path = save_figure(fig, "Zad4_ordervalue_by_payment.png")
    print(f"Wykres zapisano w pliku: {output_path}")


if __name__ == "__main__":
    main()
