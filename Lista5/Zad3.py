"""Zadanie 3: analiza multikolinearnosci przy uzyciu VIF."""

from __future__ import annotations

import statsmodels.api as sm

from common import add_constant, load_boston, print_frame, rmse, vif_table


def main() -> None:
    """Liczy VIF dla predyktorow i sprawdza wplyw usuniecia najbardziej kolinearnej zmiennej."""

    frame = load_boston()
    target = frame["medv"]
    features = frame.drop(columns=["medv"])

    vif_before = vif_table(features)
    variable_to_drop = str(vif_before.iloc[0]["variable"])
    reduced_features = features.drop(columns=[variable_to_drop])
    vif_after = vif_table(reduced_features)

    full_model = sm.OLS(target, add_constant(features)).fit()
    reduced_model = sm.OLS(target, add_constant(reduced_features)).fit()

    print("=== Lista5/Zad3: Problemy wspolzaleznosci zmiennych - VIF ===")
    print(
        "Zbior danych: lokalny plik 'Dane do listy 5/BostonHousing.csv'; "
        "analiza VIF dla wszystkich poczatkowych predyktorow modelu medv."
    )
    print_frame("VIF przed usunieciem zmiennej:", vif_before, precision=4)
    print(f"Usunieta zmienna z najwyzszym VIF: {variable_to_drop}")
    print_frame("VIF po usunieciu najbardziej kolinearnej zmiennej:", vif_after, precision=4)
    print(
        f"Model pelny:   R2 = {full_model.rsquared:.4f}, R2 adj = {full_model.rsquared_adj:.4f}, RMSE = {rmse(target, full_model.predict(add_constant(features))):.4f}"
    )
    print(
        f"Model po redukcji: R2 = {reduced_model.rsquared:.4f}, R2 adj = {reduced_model.rsquared_adj:.4f}, RMSE = {rmse(target, reduced_model.predict(add_constant(reduced_features))):.4f}"
    )


if __name__ == "__main__":
    main()
