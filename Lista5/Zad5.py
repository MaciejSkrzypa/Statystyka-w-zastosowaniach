"""Zadanie 5: ocena generalizacji na zbiorze treningowym i testowym."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split

from common import backward_elimination, load_boston, rmse, save_figure


def main() -> None:
    """Porownuje jakosc modelu na zbiorze treningowym i testowym."""

    frame = load_boston()
    target = frame["medv"]
    base_features = frame.drop(columns=["medv"])
    selected, _, _ = backward_elimination(base_features, target)
    features = base_features[selected]

    x_train, x_test, y_train, y_test = train_test_split(
        features, target, test_size=0.30, random_state=20260327
    )
    model = LinearRegression().fit(x_train, y_train)
    train_pred = model.predict(x_train)
    test_pred = model.predict(x_test)

    train_r2 = r2_score(y_train, train_pred)
    test_r2 = r2_score(y_test, test_pred)
    train_mae = mean_absolute_error(y_train, train_pred)
    test_mae = mean_absolute_error(y_test, test_pred)
    train_rmse = rmse(y_train, train_pred)
    test_rmse = rmse(y_test, test_pred)

    print("=== Lista5/Zad5: Ocena jakosci modelu na danych treningowych i testowych ===")
    print(
        "Zbior danych: lokalny plik 'Dane do listy 5/BostonHousing.csv'; "
        f"model trenowany na predyktorach po eliminacji wstecznej: {', '.join(selected)}."
    )
    print(f"Liczba obserwacji treningowych: {len(x_train)}")
    print(f"Liczba obserwacji testowych: {len(x_test)}")
    print(f"Train -> R2 = {train_r2:.4f}, MAE = {train_mae:.4f}, RMSE = {train_rmse:.4f}")
    print(f"Test  -> R2 = {test_r2:.4f}, MAE = {test_mae:.4f}, RMSE = {test_rmse:.4f}")
    if train_r2 - test_r2 > 0.10 and train_rmse < test_rmse:
        conclusion = "model wykazuje oznaki przeuczenia"
    elif train_r2 < 0.50 and test_r2 < 0.50:
        conclusion = "model moze byc niedouczony"
    else:
        conclusion = "model generalizuje stabilnie"
    print(f"Wniosek: {conclusion}.")

    fig, ax = plt.subplots(figsize=(6.8, 4.8))
    ax.scatter(y_test, test_pred, alpha=0.75, color="#457b9d", edgecolor="white", linewidth=0.5)
    diagonal = np.linspace(min(y_test.min(), test_pred.min()), max(y_test.max(), test_pred.max()), 100)
    ax.plot(diagonal, diagonal, color="#e63946", linestyle="--", linewidth=2, label="idealne dopasowanie")
    ax.set_title("Zbior testowy: ceny rzeczywiste vs przewidywane")
    ax.set_xlabel("Rzeczywiste medv")
    ax.set_ylabel("Przewidywane medv")
    ax.legend()
    output_path = save_figure(fig, "Zad5_test_predictions.png")
    print(f"Wykres zapisano w pliku: {output_path}")


if __name__ == "__main__":
    main()
