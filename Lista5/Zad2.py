"""Zadanie 2: eliminacja wsteczna predyktorow."""

from __future__ import annotations

import pandas as pd
import statsmodels.api as sm

from common import add_constant, backward_elimination, load_boston, print_frame, rmse


def main() -> None:
    """Wykonuje eliminacje wsteczna i porownuje model pelny z redukowanym."""

    frame = load_boston()
    target = frame["medv"]
    features = frame.drop(columns=["medv"])

    full_model = sm.OLS(target, add_constant(features)).fit()
    selected, steps, reduced_model = backward_elimination(features, target)

    print("=== Lista5/Zad2: Wybor zmiennych - eliminacja wsteczna ===")
    print(
        "Zbior danych: lokalny plik 'Dane do listy 5/BostonHousing.csv'; "
        "analiza dla kolumny medv i wszystkich poczatkowych predyktorow."
    )
    print(f"Liczba obserwacji: {len(frame)}")
    print("Kolejne kroki eliminacji:")
    print_frame("Zmienne usuwane wedlug najwiekszego p-value:", pd.DataFrame(steps), precision=6)
    print(f"Predyktory pozostawione w modelu koncowym: {', '.join(selected)}")
    print(f"Model pelny:    R2 = {full_model.rsquared:.4f}, R2 adj = {full_model.rsquared_adj:.4f}, AIC = {full_model.aic:.2f}, BIC = {full_model.bic:.2f}, RMSE = {rmse(target, full_model.predict(add_constant(features))):.4f}")
    print(f"Model zreduk.:  R2 = {reduced_model.rsquared:.4f}, R2 adj = {reduced_model.rsquared_adj:.4f}, AIC = {reduced_model.aic:.2f}, BIC = {reduced_model.bic:.2f}, RMSE = {rmse(target, reduced_model.predict(add_constant(features[selected]))):.4f}")


if __name__ == "__main__":
    main()
