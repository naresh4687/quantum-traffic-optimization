"""Small statistics helpers for comparing QUBO energy with simulator performance."""

from __future__ import annotations

import numpy as np


def rank_average(values: np.ndarray) -> np.ndarray:
    """Ranks starting at 1, with tied values sharing the average of their ranks."""
    values = np.asarray(values, dtype=float)
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values))
    i = 0
    while i < len(values):
        j = i
        while j + 1 < len(values) and values[order[j + 1]] == values[order[i]]:
            j += 1
        ranks[order[i : j + 1]] = (i + j) / 2.0 + 1.0
        i = j + 1
    return ranks


def pearson(x: np.ndarray, y: np.ndarray) -> float:
    """Pearson correlation; NaN if either input is constant."""
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    if x.shape != y.shape or x.ndim != 1 or len(x) < 2:
        raise ValueError("need two equal-length 1-D inputs with at least 2 points")
    xc, yc = x - x.mean(), y - y.mean()
    denom = float(np.sqrt((xc**2).sum() * (yc**2).sum()))
    return float("nan") if denom == 0.0 else float((xc * yc).sum() / denom)


def spearman(x: np.ndarray, y: np.ndarray) -> float:
    """Spearman rank correlation (Pearson on average ranks); NaN if either is constant."""
    return pearson(rank_average(x), rank_average(y))


def rank_of(values: np.ndarray, index: int) -> int:
    """1-based rank of ``values[index]`` when sorted ascending (1 = smallest); ties share the best rank."""
    values = np.asarray(values, dtype=float)
    return int((values < values[index]).sum()) + 1
