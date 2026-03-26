"""Wspolne narzedzia do zadan z Listy 4.

Modul generuje powtarzalny syntetyczny zbior danych oraz udostepnia
pomocnicze funkcje do opisu brakow danych, statystyk opisowych i doboru
przekazow tekstowych wykorzystywanych w osobnych plikach `ZadX.py`.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

DATASET_SEED = 20260326
ANALYSIS_COLUMNS = [
    "experience_years",
    "income_pln",
    "savings_pln",
    "wellbeing_score",
]
PAIR_X = "experience_years"
PAIR_Y = "income_pln"


@dataclass(frozen=True)
class CorrelationLabel:
    """Krotka etykieta opisujaca sile zaleznosci korelacyjnej."""

    name: str
    description: str


def load_dataset(seed: int = DATASET_SEED) -> pd.DataFrame:
    """Tworzy powtarzalny syntetyczny zbior danych.

    Parameters
    ----------
    seed:
        Ziarno generatora losowego.

    Returns
    -------
    pandas.DataFrame
        Ramka danych z czterema zmiennymi ilosciowymi i kilkoma brakami.
    """

    rng = np.random.default_rng(seed)
    n = 150
    latent = rng.normal(0.0, 1.0, n)

    experience_years = 9.5 + 3.8 * latent + rng.normal(0.0, 0.7, n)
    income_pln = 3600.0 + 850.0 * np.exp(0.42 * latent) + rng.normal(0.0, 140.0, n)
    savings_pln = 500.0 + 0.28 * income_pln + 95.0 * latent + rng.normal(0.0, 180.0, n)
    wellbeing_score = 50.0 + 6.5 * np.tanh(latent) + rng.normal(0.0, 1.5, n)

    frame = pd.DataFrame(
        {
            "experience_years": experience_years,
            "income_pln": income_pln,
            "savings_pln": savings_pln,
            "wellbeing_score": wellbeing_score,
        }
    )

    # Celowe odchylenia: kilka brakow i dwa punkty odstajace, zeby pokazac
    # roznice miedzy korelacja Pearsona i Spearmana.
    frame.loc[[11, 47], "income_pln"] = np.nan
    frame.loc[[18], "experience_years"] = np.nan
    frame.loc[[71], "savings_pln"] = np.nan
    frame.loc[[89], "wellbeing_score"] = np.nan
    frame.loc[5, "income_pln"] *= 0.58
    frame.loc[102, "income_pln"] *= 1.32
    frame.loc[24, "savings_pln"] += 1300.0

    return frame


def missing_summary(frame: pd.DataFrame) -> pd.DataFrame:
    """Zwraca zestawienie brakow danych dla kazdej kolumny.

    Parameters
    ----------
    frame:
        Analizowana ramka danych.

    Returns
    -------
    pandas.DataFrame
        Tabela z liczba brakow i ich udzialem procentowym.
    """

    total = len(frame)
    missing = frame.isna().sum()
    share = (missing / total * 100.0).round(2)
    return pd.DataFrame({"braki": missing, "udzial_proc": share})


def descriptive_summary(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """Oblicza srednia, mediane i odchylenie standardowe.

    Parameters
    ----------
    frame:
        Analizowana ramka danych.
    columns:
        Lista kolumn liczbowych do podsumowania.

    Returns
    -------
    pandas.DataFrame
        Tabela statystyk opisowych.
    """

    summary = pd.DataFrame(index=columns)
    summary["liczba_obserwacji"] = frame[columns].count()
    summary["srednia"] = frame[columns].mean(numeric_only=True)
    summary["mediana"] = frame[columns].median(numeric_only=True)
    summary["odchylenie_std"] = frame[columns].std(numeric_only=True, ddof=1)
    return summary


def correlation_label(value: float) -> CorrelationLabel:
    """Nadaje opis slowny sile korelacji na podstawie wartosci bezwzglednej."""

    magnitude = abs(value)
    if magnitude < 0.2:
        return CorrelationLabel("bardzo slaba", "zaleznosc praktycznie pomijalna")
    if magnitude < 0.4:
        return CorrelationLabel("slaba", "zaleznosc widoczna, ale raczej niewielka")
    if magnitude < 0.6:
        return CorrelationLabel("umiarkowana", "zaleznosc ma znaczenie, lecz nie dominuje")
    if magnitude < 0.8:
        return CorrelationLabel("silna", "zaleznosc jest wyrazna i praktycznie istotna")
    return CorrelationLabel("bardzo silna", "zaleznosc jest bardzo wyrazna")


def complete_case(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """Zwraca podzbior obserwacji bez brakow w zadanych kolumnach."""

    return frame.loc[:, columns].dropna().copy()


def format_frame(frame: pd.DataFrame, float_format: str = "{:.3f}") -> str:
    """Formatuje ramke danych do czytelnego wyswietlenia w terminalu."""

    return frame.to_string(float_format=lambda value: float_format.format(value))


def dataset_description() -> str:
    """Zwraca opis zrodla danych uzywany w komunikatach `print`."""

    return (
        "Zbior danych: syntetyczny, wygenerowany z rozkladu normalnego dla czynnika "
        "ukrytego oraz nieliniowej transformacji do dochodu i oszczednosci; "
        f"ziarno losowe = {DATASET_SEED}."
    )


def analysis_columns_description() -> str:
    """Zwraca opis analizowanych kolumn."""

    return (
        "Analizowane kolumny: experience_years, income_pln, savings_pln, wellbeing_score."
    )


def compare_coefficients(pearson_value: float, spearman_value: float) -> str:
    """Tworzy krotki komentarz porownujacy dwa wspolczynniki."""

    difference = spearman_value - pearson_value
    if abs(difference) < 0.03:
        return "Wartosci Pearsona i Spearmana sa bardzo zblizone."
    if difference > 0:
        return "Spearman jest wyzszy od Pearsona, co wskazuje na nieliniowy, ale monotoniczny zwiazek."
    return "Pearson jest wyzszy od Spearmana, co sugeruje bardziej liniowy charakter zaleznosci."
