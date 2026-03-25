"""Lista 3, Zadanie 5: wplyw liczebnosci proby na wartosc p."""

from __future__ import annotations

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

matplotlib.use("Agg")


def main() -> None:
    """Analizuje zaleznosc miedzy liczebnoscia proby a wartoscia p."""
    rng = np.random.default_rng(20260325)
    sample_sizes = [10, 20, 30, 50, 80, 120, 200, 400, 700, 1000]
    p_values: list[float] = []

    print("=== Lista3/Zad5: liczebnosc proby a wartosc p ===")
    print("Zbior danych: dwie grupy normalne A~N(50,10), B~N(50.8,10); zmienna liczebnosc proby od 10 do 1000")
    print("Liczebnosc | Srednia A | Srednia B | p-value")

    for n in sample_sizes:
        grupa_a = rng.normal(loc=50.0, scale=10.0, size=n)
        grupa_b = rng.normal(loc=50.8, scale=10.0, size=n)
        _, p_value = stats.ttest_ind(grupa_a, grupa_b, equal_var=False)
        p_values.append(float(p_value))
        print(f"{n:10d} | {np.mean(grupa_a):8.3f} | {np.mean(grupa_b):8.3f} | {p_value:7.5f}")

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(sample_sizes, p_values, marker="o")
    ax.axhline(0.05, color="red", linestyle="--", label="alpha = 0.05")
    ax.set_xlabel("Liczebnosc proby")
    ax.set_ylabel("Wartosc p")
    ax.set_title("Wplyw liczebnosci proby na wartosc p")
    ax.legend()
    fig.tight_layout()
    fig.savefig("Lista3/Zad5_p_value_vs_n.png", dpi=150)
    plt.close(fig)

    print("Wykres zapisano w pliku: Lista3/Zad5_p_value_vs_n.png")


if __name__ == "__main__":
    main()
