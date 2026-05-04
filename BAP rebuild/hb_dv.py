"""
DV-style Hyperblock Classifier.

Collects intervals on the attributes to form hyperblocks. Based on CWU-VKD-LAB DV2.0,
which uses General Line Coordinates and collects attribute intervals.

Exposes fit(X, y) and predict(X) compatible with sklearn/BAP.
"""

from __future__ import annotations

import numpy as np
from typing import Any

try:
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False


def _as_array(X: Any) -> np.ndarray:
    if hasattr(X, "values"):
        return np.asarray(X)
    return np.asarray(X, dtype=float)


def _as_labels(y: Any) -> np.ndarray:
    if hasattr(y, "values"):
        arr = np.asarray(y)
    else:
        arr = np.asarray(y)
    return np.asarray(arr, dtype=object if arr.dtype == object else str)


class Hyperblock:
    """Axis-aligned hyperblock defined by intervals per attribute."""

    __slots__ = ("mins", "maxs", "class_")

    def __init__(self, mins: np.ndarray, maxs: np.ndarray, class_: str):
        self.mins = np.asarray(mins, dtype=float)
        self.maxs = np.asarray(maxs, dtype=float)
        self.class_ = str(class_)

    def contains(self, x: np.ndarray) -> bool:
        return bool(np.all((x >= self.mins) & (x <= self.maxs)))

    def distance_to(self, x: np.ndarray) -> float:
        """L-infinity distance from point to hyperblock boundary."""
        diff = np.maximum(0, np.maximum(self.mins - x, x - self.maxs))
        return float(np.linalg.norm(diff, ord=np.inf))


class HyperblockClassifierDV:
    """
    DV-style hyperblock classifier.

    Collects intervals on each attribute: for each class, builds axis-aligned
    hyperblocks from the attribute-wise min/max ranges of training points.
    Uses multiple HBs per class by partitioning points into spatial regions.
    Prediction: containment in HB -> class; else nearest HB.
    """

    def __init__(
        self,
        n_intervals_per_attr: int = 5,
        k_nearest_hbs: int = 5,
        random_state: int | None = None,
    ):
        self.n_intervals_per_attr = n_intervals_per_attr
        self.k_nearest_hbs = k_nearest_hbs
        self.random_state = random_state
        self.hyperblocks_: list[Hyperblock] = []
        self.classes_: np.ndarray = np.array([])
        self.n_features_in_: int = 0
        self.feature_names_in_: list[str] = []

    def fit(self, X: Any, y: Any) -> "HyperblockClassifierDV":
        if hasattr(X, "columns"):
            self.feature_names_in_ = list(X.columns)
        else:
            self.feature_names_in_ = []
        X = _as_array(X)
        y = _as_labels(y)
        if X.ndim != 2:
            raise ValueError("X must be 2D (n_samples, n_features)")
        n_samples, n_features = X.shape
        self.n_features_in_ = n_features
        if not self.feature_names_in_:
            self.feature_names_in_ = [f"dim_{d}" for d in range(n_features)]
        self.classes_ = np.unique(y)

        # Collect intervals per attribute per class, form HBs
        self.hyperblocks_ = self._build_interval_hyperblocks(X, y)
        return self

    def _build_interval_hyperblocks(self, X: np.ndarray, y: np.ndarray) -> list[Hyperblock]:
        """
        Build HBs from attribute intervals. For each class: partition each
        attribute into intervals (percentiles or equal-width), form HBs from
        combinations that cover same-class points.
        """
        hbs: list[Hyperblock] = []
        classes = np.unique(y)

        for cls in classes:
            mask = y == cls
            X_cls = X[mask]
            if len(X_cls) == 0:
                continue

            # Primary HB: axis-aligned bounding box of class (intervals = [min, max] per attr)
            mins = np.min(X_cls, axis=0)
            maxs = np.max(X_cls, axis=0)
            # Avoid zero-volume
            for d in range(len(mins)):
                if mins[d] == maxs[d]:
                    eps = 1e-10
                    mins[d] -= eps
                    maxs[d] += eps
            hbs.append(Hyperblock(mins, maxs, str(cls)))

            # Additional HBs: partition by quantiles on first attribute
            n_splits = min(self.n_intervals_per_attr, len(X_cls) - 1)
            if n_splits >= 2:
                quantiles = np.percentile(X_cls[:, 0], np.linspace(0, 100, n_splits + 1))
                for i in range(n_splits):
                    lo, hi = quantiles[i], quantiles[i + 1]
                    if lo >= hi:
                        continue
                    mask = (X_cls[:, 0] >= lo) & (X_cls[:, 0] <= hi)
                    sub = X_cls[mask]
                    if len(sub) >= 2:
                        smin = np.min(sub, axis=0)
                        smax = np.max(sub, axis=0)
                        if not np.all(smin == smax):
                            hbs.append(Hyperblock(smin, smax, str(cls)))
        return hbs

    def get_hyperblock_edges(self) -> list[dict[str, float | str]]:
        """Return list of hyperblock edge dicts (class, dim_min, dim_max per dimension)."""
        rows = []
        for i, hb in enumerate(self.hyperblocks_):
            row: dict[str, float | str] = {"hb_id": i, "class": hb.class_}
            for d, name in enumerate(self.feature_names_in_):
                if d < len(hb.mins) and d < len(hb.maxs):
                    row[f"{name}_min"] = float(hb.mins[d])
                    row[f"{name}_max"] = float(hb.maxs[d])
            rows.append(row)
        return rows

    def predict(self, X: Any) -> np.ndarray:
        X = _as_array(X)
        if X.ndim == 1:
            X = X.reshape(1, -1)
        n = X.shape[0]
        preds = np.empty(n, dtype=object)

        for i in range(n):
            x = X[i]
            for hb in self.hyperblocks_:
                if hb.contains(x):
                    preds[i] = hb.class_
                    break
            else:
                dists = [(hb.distance_to(x), hb.class_) for hb in self.hyperblocks_]
                dists.sort(key=lambda d: d[0])
                k = min(self.k_nearest_hbs, len(dists))
                from collections import Counter
                top_classes = [c for _, c in dists[:k]]
                preds[i] = Counter(top_classes).most_common(1)[0][0]
        return preds
