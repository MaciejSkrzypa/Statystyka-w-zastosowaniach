"""Zadanie 6: ograniczenia regresji liniowej dla zaleznosci nieliniowej."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm

from common import load_insurance, rmse, save_figure


def main() -> None:
    """Porownuje model liniowy i model z transformacja kwadratowa dla danych insurance."""

    frame = load_insurance()
    features_linear = pd.DataFrame({"age": frame["age"]})
    features_quadratic = pd.DataFrame(
        {"age": frame["age"], "age_sq": frame["age"] ** 2}
    )
    target = frame["charges"]

    linear_model = sm.OLS(target, sm.add_constant(features_linear, has_constant="add")).fit()
    quadratic_model = sm.OLS(target, sm.add_constant(features_quadratic, has_constant="add")).fit()
    pred_linear = linear_model.predict(sm.add_constant(features_linear, has_constant="add"))
    pred_quadratic = quadratic_model.predict(sm.add_constant(features_quadratic, has_constant="add"))

    print("=== Lista5/Zad6: Ograniczenia regresji dla danych nieliniowych ===")
    print(
        "Zbior danych: lokalny plik 'Dane do listy 5/insurance.csv'; "
        "analiza zaleznosci charges od age."
    )
    print(f"Liczba obserwacji: {len(frame)}")
    print(f"Model liniowy -> R2 = {linear_model.rsquared:.4f}, RMSE = {rmse(target, pred_linear):.4f}")
    print(
        f"Model z transformacja kwadratowa -> R2 = {quadratic_model.rsquared:.4f}, RMSE = {rmse(target, pred_quadratic):.4f}"
    )
    better_model = "kwadratowy" if quadratic_model.rsquared > linear_model.rsquared else "liniowy"
    print(f"Lepiej dopasowany model: {better_model}.")

    age_grid = np.linspace(frame["age"].min(), frame["age"].max(), 200)
    fig, ax = plt.subplots(figsize=(6.8, 4.8))
    ax.scatter(frame["age"], target, alpha=0.45, color="#264653", edgecolor="white", linewidth=0.4)
    ax.plot(age_grid, linear_model.predict(sm.add_constant(pd.DataFrame({"age": age_grid}), has_constant="add")), color="#e76f51", linewidth=2, label="model liniowy")
    quad_frame = pd.DataFrame({"age": age_grid, "age_sq": age_grid**2})
    ax.plot(age_grid, quadratic_model.predict(sm.add_constant(quad_frame, has_constant="add")), color="#2a9d8f", linewidth=2, label="model z age^2")
    ax.set_title("Charges vs age: model liniowy i kwadratowy")
    ax.set_xlabel("age")
    ax.set_ylabel("charges")
    ax.legend()
    output_path = save_figure(fig, "Zad6_nonlinear_fit.png")
    print(f"Wykres zapisano w pliku: {output_path}")


if __name__ == "__main__":
    main()
