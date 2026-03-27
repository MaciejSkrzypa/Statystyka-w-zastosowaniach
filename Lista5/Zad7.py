"""Zadanie 7: porownanie modeli regresji dla roznych zestawow predyktorow."""

from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd
import statsmodels.api as sm

from common import backward_elimination, load_mpg, print_frame, rmse, save_figure, vif_table


def prepare_full_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Buduje pelny zestaw predyktorow dla zbioru mpg."""

    numeric = frame[
        ["cylinders", "displacement", "horsepower", "weight", "acceleration", "model_year"]
    ]
    origin_dummies = pd.get_dummies(frame["origin"], prefix="origin", drop_first=True, dtype=int)
    return pd.concat([numeric, origin_dummies], axis=1)


def model_summary(name: str, features: pd.DataFrame, target: pd.Series) -> dict[str, float | str]:
    """Dopasowuje model i zwraca podstawowe miary wraz z podsumowaniem VIF."""

    model = sm.OLS(target, sm.add_constant(features, has_constant="add")).fit()
    vif = vif_table(features)
    return {
        "model": name,
        "n_features": features.shape[1],
        "r2": float(model.rsquared),
        "rmse": rmse(target, model.predict(sm.add_constant(features, has_constant="add"))),
        "max_vif": float(vif["vif"].max()),
        "mean_vif": float(vif["vif"].mean()),
    }


def main() -> None:
    """Porownuje trzy modele mpg z roznymi zestawami predyktorow."""

    frame = load_mpg()
    target = frame["mpg"]
    full_features = prepare_full_features(frame)
    reduced_columns, _, _ = backward_elimination(full_features, target)
    reduced_features = full_features[reduced_columns]
    hard_features = frame[
        ["cylinders", "displacement", "horsepower", "weight", "acceleration", "model_year"]
    ]

    results = pd.DataFrame(
        [
            model_summary("pelny", full_features, target),
            model_summary("po eliminacji", reduced_features, target),
            model_summary("twarde liczbowe", hard_features, target),
        ]
    )

    print("=== Lista5/Zad7: Porownanie modeli regresji dla roznych predyktorow ===")
    print(
        "Zbior danych: lokalny plik 'Dane do listy 5/mpg.csv'; "
        "zmienna objasniana: mpg."
    )
    print("Model pelny: zmienne liczbowe + zakodowana zmienna origin.")
    print(f"Model po eliminacji: {', '.join(reduced_columns)}")
    print("Model twardy: cylinders, displacement, horsepower, weight, acceleration, model_year.")
    print_frame("Porownanie modeli:", results, precision=4)

    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    ax.bar(results["model"], results["r2"], color=["#1d3557", "#457b9d", "#a8dadc"])
    ax.set_title("Porownanie R2 dla trzech modeli mpg")
    ax.set_xlabel("Model")
    ax.set_ylabel("R2")
    output_path = save_figure(fig, "Zad7_model_comparison.png")
    print(f"Wykres zapisano w pliku: {output_path}")


if __name__ == "__main__":
    main()
