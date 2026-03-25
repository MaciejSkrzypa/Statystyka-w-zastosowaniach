"""Zadanie 5: Statystyki opisowe i wizualizacja dla danych z seaborn."""

from __future__ import annotations

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from scipy import stats

matplotlib.use("Agg")


def main() -> None:
    """Liczy statystyki opisowe i zapisuje trzy wykresy dla zbioru `tips`."""
    data = sns.load_dataset("tips")
    bill = data["total_bill"].dropna().to_numpy()
    tip = data["tip"].dropna().to_numpy()

    mean_v = float(np.mean(bill))
    median_v = float(np.median(bill))
    q1, q3 = np.percentile(bill, [25, 75])
    std_v = float(np.std(bill, ddof=1))
    skew_v = float(stats.skew(bill, bias=False))
    kurt_v = float(stats.kurtosis(bill, fisher=True, bias=False))

    print("=== Zadanie 5: Statystyki opisowe na zbiorze seaborn 'tips' ===")
    print('Zbior danych: seaborn.load_dataset("tips"); analiza kolumn total_bill i tip')
    print(f"Liczba obserwacji: {bill.size}")
    print(f"Srednia total_bill: {mean_v:.4f}")
    print(f"Mediana total_bill: {median_v:.4f}")
    print(f"Kwartyl Q1: {q1:.4f}, kwartyl Q3: {q3:.4f}")
    print(f"Odchylenie standardowe: {std_v:.4f}")
    print(f"Skosnosc: {skew_v:.4f}")
    print(f"Kurtoza (nadmiarowa): {kurt_v:.4f}")

    if abs(skew_v) < 0.5:
        symmetry = "w przyblizeniu symetryczny"
    elif skew_v > 0:
        symmetry = "prawoskosny"
    else:
        symmetry = "lewoskosny"
    print(f"Interpretacja ksztaltu rozkladu: rozklad jest {symmetry}.")

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(bill, bins=20, color="#4C72B0", edgecolor="black")
    ax.set_title("Histogram zmiennej total_bill")
    ax.set_xlabel("total_bill")
    ax.set_ylabel("Liczebnosc")
    fig.tight_layout()
    fig.savefig("Lista1/Zad5_histogram.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.boxplot(bill, vert=False)
    ax.set_title("Boxplot zmiennej total_bill")
    ax.set_xlabel("total_bill")
    fig.tight_layout()
    fig.savefig("Lista1/Zad5_boxplot.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.scatter(bill, tip, alpha=0.7)
    ax.set_title("Wykres rozrzutu: total_bill vs tip")
    ax.set_xlabel("total_bill")
    ax.set_ylabel("tip")
    fig.tight_layout()
    fig.savefig("Lista1/Zad5_scatter.png", dpi=150)
    plt.close(fig)

    print("Wykresy zapisano: Lista1/Zad5_histogram.png, Lista1/Zad5_boxplot.png, Lista1/Zad5_scatter.png")


if __name__ == "__main__":
    main()
