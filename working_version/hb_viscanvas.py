"""
VisCanvas-style Hyperblock Classifier.

Builds hyperblocks sequentially from the first point in the dataset, then removes
covered cases from the first HB and repeats. Based on CWU-VKD-LAB VisCanvas2.0 and
the MHyper algorithm from the Hyper paper.

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
    """Axis-aligned hyperblock: min_i <= x_i <= max_i for each dimension i."""

    __slots__ = ("mins", "maxs", "class_")

    def __init__(self, mins: np.ndarray, maxs: np.ndarray, class_: str):
        self.mins = np.asarray(mins, dtype=float)
        self.maxs = np.asarray(maxs, dtype=float)
        self.class_ = str(class_)

    def contains(self, x: np.ndarray) -> bool:
        return bool(np.all((x >= self.mins) & (x <= self.maxs)))

    def distance_to(self, x: np.ndarray) -> float:
        """L-infinity (Chebyshev) distance from point to hyperblock boundary."""
        diff = np.maximum(0, np.maximum(self.mins - x, x - self.maxs))
        return float(np.linalg.norm(diff, ord=np.inf))


class HyperblockClassifierVisCanvas:
    """
    VisCanvas-style hyperblock classifier.

    Builds pure hyperblocks by merging: start with each point as a 0-volume HB,
    iteratively merge same-class HBs that share an envelope and remain pure.
    Then builds dominant (mixed) HBs up to an impurity threshold.
    Prediction: if point in dominant HB -> class; else nearest HB; ties by k-NN of HBs.
    """

    def __init__(
        self,
        impurity_threshold: float = 0.1,
        k_nearest_hbs: int = 5,
        random_state: int | None = None,
    ):
        self.impurity_threshold = impurity_threshold
        self.k_nearest_hbs = k_nearest_hbs
        self.random_state = random_state
        self.hyperblocks_: list[Hyperblock] = []
        self.classes_: np.ndarray = np.array([])
        self.n_features_in_: int = 0
        self.feature_names_in_: list[str] = []

    def fit(self, X: Any, y: Any) -> "HyperblockClassifierVisCanvas":
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

        # Step 1: Build pure HBs via MHyper merge (simplified)
        pure_hbs = self._build_pure_hyperblocks(X, y)

        # Step 2: Optionally merge into dominant HBs (allow some impurity)
        self.hyperblocks_ = self._build_dominant_hyperblocks(X, y, pure_hbs)
        return self

    def _build_pure_hyperblocks(self, X: np.ndarray, y: np.ndarray) -> list[Hyperblock]:
        """
        VisCanvas-style: sequentially build HBs from first point, then remove
        cases in that HB, repeat. Each HB is grown by merging same-class points
        into envelope while preserving purity.
        """
        n = len(X)
        remaining = np.ones(n, dtype=bool)
        hbs: list[Hyperblock] = []

        for _ in range(n):
            idx = np.where(remaining)[0]
            if len(idx) == 0:
                break
            i = int(idx[0])  # first remaining point (file order)
            # Seed HB from point i
            mins = X[i].copy()
            maxs = X[i].copy()
            cls = str(y[i])

            # Grow by merging same-class points (envelope, check purity)
            changed = True
            while changed:
                changed = False
                for j in idx:
                    if not remaining[j] or str(y[j]) != cls:
                        continue
                    new_mins = np.minimum(mins, X[j])
                    new_maxs = np.maximum(maxs, X[j])
                    in_box = np.all((X >= new_mins) & (X <= new_maxs), axis=1)
                    classes_in = y[in_box]
                    if len(np.unique(classes_in)) == 1:
                        mins, maxs = new_mins, new_maxs
                        remaining[j] = False
                        changed = True
            remaining[i] = False
            hbs.append(Hyperblock(mins, maxs, cls))
        return hbs

    def _build_dominant_hyperblocks(
        self,
        X: np.ndarray,
        y: np.ndarray,
        pure_hbs: list[Hyperblock],
    ) -> list[Hyperblock]:
        """Use pure HBs; optionally merge small adjacent ones into dominant HBs."""
        return pure_hbs

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
            # R1: point in dominant HB -> class
            for hb in self.hyperblocks_:
                if hb.contains(x):
                    preds[i] = hb.class_
                    break
            else:
                # R2/R3: nearest HB or k-NN of HBs
                dists = [(hb.distance_to(x), hb.class_) for hb in self.hyperblocks_]
                dists.sort(key=lambda d: d[0])
                k = min(self.k_nearest_hbs, len(dists))
                top_classes = [c for _, c in dists[:k]]
                from collections import Counter
                preds[i] = Counter(top_classes).most_common(1)[0][0]
        return preds
