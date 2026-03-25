"""Lista 3, Zadanie 3: test chi-kwadrat niezaleznosci."""

from __future__ import annotations

import numpy as np
from scipy.stats import chi2_contingency


def main() -> None:
    """Sprawdza zaleznosc miedzy plcia a preferowanym typem muzyki."""
    rng = np.random.default_rng(20260325)

    muzyka = np.array(["pop", "rock", "muzyka_klasyczna"])
    mezczyzni = rng.multinomial(120, [0.30, 0.50, 0.20])
    kobiety = rng.multinomial(120, [0.45, 0.30, 0.25])
    tabela = np.vstack([kobiety, mezczyzni])

    chi2_stat, p_value, dof, expected = chi2_contingency(tabela)

    print("=== Lista3/Zad3: test chi-kwadrat niezaleznosci ===")
    print("Zbior danych: syntetyczna tabela kontyngencji plec x muzyka (2x3), po 120 obserwacji na plec")
    print("Wiersze: [kobiety, mezczyzni], kolumny: [pop, rock, muzyka_klasyczna]")
    print("Tabela obserwowana:")
    print(tabela)
    print("Tabela oczekiwana:")
    print(np.round(expected, 2))
    print(f"Statystyka chi2 = {chi2_stat:.4f}, stopnie swobody = {dof}, p-value = {p_value:.6f}")
    print("Decyzja przy alpha=0.05:", "odrzucamy H0 (zaleznosc)" if p_value < 0.05 else "brak podstaw do odrzucenia H0")


if __name__ == "__main__":
    main()
