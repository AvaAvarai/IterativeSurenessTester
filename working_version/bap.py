#!/usr/bin/env python3.11
"""
Bidirectional Active Processing (BAP) implementation.
Reimplements the algorithm from Bidirectional Active Processing.md.
Single classifier baseline: Decision Tree, K-Nearest Neighbors, Support Vector Machine.
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit, StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from hb_viscanvas import HyperblockClassifierVisCanvas
from hb_dv import HyperblockClassifierDV


# --- Configuration ---

DEFAULTS = {
    "split": [0.8, 0.2],
    "folds": 5,
    "classifier": "dt",
    "knn_k": 3,
    "distance": "euclidean",
    "t": 0.95,
    # If set (e.g. 0.95), BAP only converges when accuracy >= t and each class's
    # largest HB (by count of in-HB training points of that class) covers at least
    # this fraction of that class's training points (HB classifiers only).
    "hb_coverage_t": None,
    "direction": "forward",
    "splits": 1,
    "n": 10,
    "m": 5,
    "sampling": "stratified",
    "seed": 42,
    "output_dir": "results",
    "seed_modulus": 2**31,
    "split_seed_multiplier": 7919,
    "exp_seed_split_offset": 10000,
    "exp_seed_iter_offset": 1000,
}


@dataclass
class Config:
    """BAP configuration. Load from TOML or CLI flags."""

    train: str = ""
    test: str = ""
    testing: str = "split"  # "fixed" | "split" | "cv"
    split: list[float] = field(default_factory=lambda: DEFAULTS["split"].copy())
    folds: int = DEFAULTS["folds"]
    classifier: str = DEFAULTS["classifier"]  # "dt" | "knn" | "svm"
    parameters: dict[str, Any] = field(default_factory=dict)
    distance: str = DEFAULTS["distance"]
    t: float = DEFAULTS["t"]
    hb_coverage_t: float | None = DEFAULTS["hb_coverage_t"]
    direction: str = DEFAULTS["direction"]  # "forward" | "backward"
    splits: int = DEFAULTS["splits"]
    n: int = DEFAULTS["n"]
    m: int = DEFAULTS["m"]
    sampling: str = DEFAULTS["sampling"]  # "random" | "stratified"
    seed: int = DEFAULTS["seed"]
    output_dir: str = DEFAULTS["output_dir"]

    @classmethod
    def from_toml(cls, path: str) -> "Config":
        """Load config from TOML file. Requires tomli (Python <3.11) or tomllib (3.11+)."""
        try:
            import tomllib
        except ImportError:
            try:
                import tomli as tomllib
            except ImportError:
                raise ImportError("TOML support requires Python 3.11+ or 'pip install tomli'")

        with open(path, "rb") as f:
            data = tomllib.load(f)

        cfg = cls()
        if "train" in data:
            cfg.train = str(data["train"])
        if "test" in data:
            cfg.test = str(data["test"])
        if "output_dir" in data:
            cfg.output_dir = str(data["output_dir"])
        if "testing" in data:
            t = data["testing"]
            if isinstance(t, dict):
                if "fixed" in t:
                    cfg.testing = "fixed"
                    f = t["fixed"]
                    cfg.test = str(f.get("test", cfg.test) if isinstance(f, dict) else cfg.test)
                elif "split" in t:
                    cfg.testing = "split"
                    s = t["split"]
                    vals = s.get("split", s) if isinstance(s, dict) else s
                    cfg.split = [float(x) for x in (vals if isinstance(vals, (list, tuple)) else DEFAULTS["split"])]
                elif "cv" in t:
                    cfg.testing = "cv"
                    c = t["cv"]
                    cfg.folds = int(c.get("folds", DEFAULTS["folds"]) if isinstance(c, dict) else t.get("folds", DEFAULTS["folds"]))
            else:
                cfg.testing = str(t)
        if "classifier" in data:
            cfg.classifier = str(data["classifier"]).lower()
        if "parameters" in data:
            cfg.parameters = {str(k): v for k, v in data["parameters"].items()}
        if "distance" in data:
            cfg.distance = str(data["distance"])
        if "goal" in data and isinstance(data["goal"], dict):
            g = data["goal"]
            if "t" in g:
                cfg.t = float(g["t"])
            if "hb_coverage_t" in g and g["hb_coverage_t"] is not None:
                cfg.hb_coverage_t = float(g["hb_coverage_t"])
        if "direction" in data:
            d = data["direction"]
            cfg.direction = "backward" if (isinstance(d, dict) and d.get("backward")) else "forward"
        if "splits" in data:
            cfg.splits = int(data["splits"])
        if "n" in data:
            cfg.n = int(data["n"])
        if "m" in data:
            cfg.m = int(data["m"])
        if "sampling" in data:
            s = data["sampling"]
            cfg.sampling = "stratified" if (isinstance(s, dict) and s.get("stratified")) else "random"
        if "seed" in data:
            cfg.seed = int(data["seed"])
        if "output_dir" in data:
            cfg.output_dir = str(data["output_dir"])
        return cfg

    def to_dict(self) -> dict[str, Any]:
        """Serialize for config output."""
        return {
            "train": self.train,
            "test": self.test,
            "testing": self.testing,
            "split": self.split,
            "folds": self.folds,
            "classifier": self.classifier,
            "parameters": self.parameters,
            "distance": self.distance,
            "t": self.t,
            "hb_coverage_t": self.hb_coverage_t,
            "direction": self.direction,
            "splits": self.splits,
            "n": self.n,
            "m": self.m,
            "sampling": self.sampling,
            "seed": self.seed,
            "output_dir": self.output_dir,
        }


# --- Data loading ---

# Benchmark / toolkit CSV convention (e.g. fisher_iris.csv): feature columns + trailing `class`.
EXPORT_CLASS_COL = "class"


def detect_class_column(df: pd.DataFrame) -> str:
    """Detect class/label column (case-insensitive name match)."""
    for col in df.columns:
        if str(col).lower() in ("class", "label", "target"):
            return str(col)
    raise ValueError("No 'class', 'label', or 'target' column found in CSV")


def load_csv(path: str) -> tuple[pd.DataFrame, pd.Series]:
    """Load CSV and return (X, y)."""
    df = pd.read_csv(path)
    class_col = detect_class_column(df)
    X = df.drop(columns=[class_col])
    y = df[class_col].astype(str)
    if X.isnull().any().any():
        X = X.fillna(X.mean())
    return X, y


def dataframe_tabular_export(
    X: pd.DataFrame,
    y: pd.Series,
    feature_order: list[str],
) -> pd.DataFrame:
    """
    Export cases in tabular benchmark format: one column per attribute, then `class`
    (matches e.g. Java_Tabular_Vis_Toolkit fisher_iris.csv). Column order is fixed;
    readers may ignore attribute order as long as names match.
    """
    out = X.loc[:, feature_order].copy()
    out[EXPORT_CLASS_COL] = y.astype(str).values
    return out[list(feature_order) + [EXPORT_CLASS_COL]]


def hyperblock_edges_tabular_export(
    edge_rows: list[dict[str, Any]],
    feature_names: list[str],
) -> pd.DataFrame:
    """
    Convert internal hyperblock rows (hb_id, class, attr_min, attr_max) into tabular CSV rows:
    two rows per HB — bottom row holds axis-aligned minimums, top row holds maximums.
    HB index and edge (bottom/top) are encoded only in the `class` cell, e.g.
    `Setosa__HB0__bottom` / `Setosa__HB0__top` (no separate id column).
    """
    rows: list[dict[str, Any]] = []
    for row in edge_rows:
        hid = int(row["hb_id"])
        label = str(row["class"])
        safe = label.replace("__", "_")  # keep delimiter pattern unambiguous
        cls_bottom = f"{safe}__HB{hid}__bottom"
        cls_top = f"{safe}__HB{hid}__top"
        r_bot: dict[str, Any] = {EXPORT_CLASS_COL: cls_bottom}
        r_top: dict[str, Any] = {EXPORT_CLASS_COL: cls_top}
        for fn in feature_names:
            kmin, kmax = f"{fn}_min", f"{fn}_max"
            if kmin not in row or kmax not in row:
                raise KeyError(f"hyperblock row missing {kmin}/{kmax} for feature {fn!r}")
            r_bot[fn] = float(row[kmin])
            r_top[fn] = float(row[kmax])
        rows.append(r_bot)
        rows.append(r_top)
    df = pd.DataFrame(rows)
    return df[list(feature_names) + [EXPORT_CLASS_COL]]


def normalize_features(X_train: pd.DataFrame, X_test: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Min-max normalize. Uses train stats for both."""
    mn = X_train.min()
    mx = X_train.max()
    rng = mx - mn
    rng = rng.replace(0, 1)
    X_train_norm = (X_train - mn) / rng
    X_test_norm = (X_test - mn) / rng
    return X_train_norm, X_test_norm


def split_data(
    X: pd.DataFrame,
    y: pd.Series,
    train_ratio: float,
    test_ratio: float,
    seed: int,
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    """Stratified train/test split."""
    n = len(X)
    train_size = int(round(train_ratio * n))
    test_size = n - train_size
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, train_size=train_size, test_size=test_size, stratify=y, random_state=seed
    )
    return X_train, y_train, X_test, y_test


def load_data(
    config: Config, split_seed: int, fold_index: int | None = None
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    """Load and prepare train/test data per config. For CV, fold_index is 0..folds-1."""
    if config.testing == "fixed" and config.test:
        X_train, y_train = load_csv(config.train)
        X_test, y_test = load_csv(config.test)
        X_test = X_test.fillna(X_train.mean())
    elif config.testing == "cv" and fold_index is not None:
        X, y = load_csv(config.train)
        kf = StratifiedKFold(n_splits=config.folds, shuffle=True, random_state=split_seed)
        train_idx, test_idx = list(kf.split(X, y))[fold_index]
        X_train = X.iloc[train_idx].reset_index(drop=True)
        y_train = y.iloc[train_idx].reset_index(drop=True)
        X_test = X.iloc[test_idx].reset_index(drop=True)
        y_test = y.iloc[test_idx].reset_index(drop=True)
    else:
        X, y = load_csv(config.train)
        tr, te = config.split[0], config.split[1]
        X_train, y_train, X_test, y_test = split_data(X, y, tr, te, split_seed)

    X_train, X_test = normalize_features(X_train, X_test)
    return X_train, y_train, X_test, y_test


# --- Classifiers ---


def make_classifier(
    name: str,
    seed: int,
    distance: str = "euclidean",
    parameters: dict[str, Any] | None = None,
) -> Any:
    """Create classifier instance."""
    params = dict(parameters or {})
    if name == "dt":
        return DecisionTreeClassifier(random_state=seed, **params)
    if name == "knn":
        k = params.get("k", params.get("n_neighbors", DEFAULTS["knn_k"]))
        extra = {k2: v for k2, v in params.items() if k2 not in ("k", "n_neighbors", "distance")}
        return KNeighborsClassifier(n_neighbors=k, metric=distance, **extra)
    if name == "svm":
        return SVC(kernel="rbf", random_state=seed, **params)
    if name == "hb_vis" or name == "hb_viscanvas":
        return HyperblockClassifierVisCanvas(random_state=seed)
    if name == "hb_dv":
        return HyperblockClassifierDV(random_state=seed)
    raise ValueError(f"Unknown classifier: {name}. Use dt, knn, svm, hb_vis, or hb_dv.")


def compute_hb_largest_per_class_coverage(
    clf: Any,
    X_sub: pd.DataFrame,
    y_sub: pd.Series,
) -> tuple[float, dict[str, float], dict[str, int], int]:
    """
    For each class c, among hyperblocks labeled c, take the one that contains the
    most training points whose label is c. Return:
    - min over classes of (that count / n_c),
    - per-class fractions,
    - per-class point counts in those largest HBs,
    - total number of hyperblocks.

    Interpreting "top-sized HB per class": largest by in-training-set cardinality
    for points of that class (geometric containment in axis-aligned HB).
    """
    if not hasattr(clf, "hyperblocks_"):
        return 1.0, {}, {}, 0
    hbs = clf.hyperblocks_
    n_hb = len(hbs)
    if not hbs:
        return 0.0, {}, {}, 0

    Xa = np.asarray(X_sub.values if hasattr(X_sub, "values") else X_sub, dtype=float)
    y_arr = np.asarray(y_sub.astype(str))
    classes = np.unique(y_arr)
    per_class_frac: dict[str, float] = {}
    per_class_largest_n: dict[str, int] = {}

    for c in classes:
        mask_c = y_arr == c
        n_c = int(np.sum(mask_c))
        if n_c == 0:
            continue
        X_c = Xa[mask_c]
        best = 0
        for hb in hbs:
            if str(hb.class_) != str(c):
                continue
            inside = np.all((X_c >= hb.mins) & (X_c <= hb.maxs), axis=1)
            best = max(best, int(np.sum(inside)))
        per_class_largest_n[str(c)] = best
        per_class_frac[str(c)] = best / n_c if n_c else 0.0

    if not per_class_frac:
        return 0.0, {}, {}, n_hb
    min_cov = float(min(per_class_frac.values()))
    return min_cov, per_class_frac, per_class_largest_n, n_hb


# --- Sampling ---


def sample_indices(
    available: set[int],
    y: pd.Series,
    n: int,
    method: str,
    seed: int,
) -> list[int]:
    """Sample n indices from available using method."""
    avail_list = list(available)
    if len(avail_list) <= n:
        return avail_list

    if method == "random":
        rng = np.random.default_rng(seed)
        return rng.choice(avail_list, size=n, replace=False).tolist()

    if method == "stratified":
        avail_y = y.iloc[avail_list].values
        unique = np.unique(avail_y)
        if len(unique) < 2 or n < len(unique):
            rng = np.random.default_rng(seed)
            return rng.choice(avail_list, size=n, replace=False).tolist()
        dummy_X = np.arange(len(avail_list)).reshape(-1, 1)
        sss = StratifiedShuffleSplit(n_splits=1, test_size=n, random_state=seed)
        _, sel = next(sss.split(dummy_X, avail_y))
        return [avail_list[i] for i in sel]

    rng = np.random.default_rng(seed)
    return rng.choice(avail_list, size=n, replace=False).tolist()


# --- BAP Core ---


def run_single_iteration(
    exp_id: int,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    config: Config,
    seed: int,
    out_dir: str,
) -> dict[str, Any] | None:
    """
    Run one BAP iteration. Returns result dict if converged, None if failed.
    Pure function except file writes for converged casesets.
    """
    rng = np.random.default_rng(seed)
    clf_seed = int(rng.integers(0, DEFAULTS["seed_modulus"]))

    direction = config.direction
    m = config.m
    t = config.t
    classifier_name = config.classifier
    sampling = config.sampling

    n_train = len(X_train)
    if direction == "forward":
        case_indices: set[int] = set()
    else:
        case_indices = set(range(n_train))

    iteration = 0
    accuracy = 0.0

    while True:
        if direction == "forward":
            available = set(range(n_train)) - case_indices
            if not available:
                return None
            take = min(m, len(available))
            selected = sample_indices(available, y_train, take, sampling, seed + iteration)
            case_indices = case_indices | set(selected)
        else:
            available = case_indices
            if len(available) < 2:
                return None
            take = min(m, len(available) - 1)
            if take < 1:
                return None
            selected = sample_indices(available, y_train, take, sampling, seed + iteration)
            case_indices = case_indices - set(selected)

        idx_list = sorted(case_indices)
        X_sub = X_train.iloc[idx_list]
        y_sub = y_train.iloc[idx_list]

        if len(y_sub.unique()) < 2:
            iteration += 1
            continue

        clf = make_classifier(
            classifier_name,
            clf_seed + iteration,
            config.distance,
            {"k": config.parameters.get("k", DEFAULTS["knn_k"]), **config.parameters},
        )
        clf.fit(X_sub, y_sub)
        y_pred = clf.predict(X_test)
        accuracy = float(accuracy_score(y_test, y_pred))

        iteration += 1

        min_hb_class_cov = 1.0
        hb_coverage_detail = ""
        hb_weighted_coverage = 1.0
        hb_count = 0
        if hasattr(clf, "hyperblocks_"):
            min_hb_class_cov, per_frac, per_n, hb_count = compute_hb_largest_per_class_coverage(
                clf, X_sub, y_sub
            )
            if per_frac:
                y_arr = np.asarray(y_sub.astype(str))
                hb_weighted_coverage = sum(per_n.values()) / len(y_sub) if len(y_sub) else 0.0
                hb_coverage_detail = "; ".join(
                    f"{c}:{per_frac[c]:.3f}({per_n[c]}/{int(np.sum(y_arr == c))})"
                    for c in sorted(per_frac.keys())
                )

        need_hb_cov = config.hb_coverage_t is not None and hasattr(clf, "hyperblocks_")
        hb_ok = not need_hb_cov or min_hb_class_cov >= float(config.hb_coverage_t)

        if accuracy >= t and hb_ok:
            feature_names = list(X_train.columns)
            conv_df = dataframe_tabular_export(X_sub, y_sub, feature_names)
            fname = os.path.join(out_dir, f"converged_exp_{exp_id}_seed{seed}.csv")
            conv_df.to_csv(fname, index=False)
            if hasattr(clf, "get_hyperblock_edges"):
                edges = clf.get_hyperblock_edges()
                if edges:
                    edges_df = hyperblock_edges_tabular_export(edges, feature_names)
                    edges_fname = os.path.join(out_dir, f"converged_exp_{exp_id}_seed{seed}_hyperblocks.csv")
                    edges_df.to_csv(edges_fname, index=False)
            from collections import Counter
            class_dist = Counter(y_sub.astype(str))
            class_dist_str = ", ".join(f"{c}:{n}" for c, n in sorted(class_dist.items()))
            return {
                "exp_id": exp_id,
                "seed": seed,
                "cases_needed": len(case_indices),
                "accuracy": accuracy,
                "iteration": iteration,
                "cases_pct": len(case_indices) / n_train * 100,
                "class_dist": class_dist_str,
                "hb_count": hb_count,
                "hb_min_class_coverage": min_hb_class_cov,
                "hb_weighted_coverage": hb_weighted_coverage,
                "hb_coverage_detail": hb_coverage_detail,
            }
        if direction == "forward" and len(case_indices) >= n_train:
            return None
        if direction == "backward" and len(case_indices) < 2:
            return None

    return None


def run_split(
    split_id: int,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    config: Config,
    base_seed: int,
    out_dir: str,
    n_iterations: int | None = None,
) -> list[dict[str, Any]]:
    """Run n iterations for one split. Returns list of converged results."""
    n = n_iterations if n_iterations is not None else config.n
    results = []
    for i in range(n):
        exp_seed = (base_seed + split_id * DEFAULTS["exp_seed_split_offset"] + i * DEFAULTS["exp_seed_iter_offset"]) % DEFAULTS["seed_modulus"]
        r = run_single_iteration(
            i + 1, X_train, y_train, X_test, y_test, config, exp_seed, out_dir
        )
        if r is not None:
            r["split"] = split_id
            results.append(r)
    return results


def compute_statistics(results: list[dict[str, Any]], total_runs: int) -> list[dict[str, Any]]:
    """Compute statistics per magnitude. Simplified: single block."""
    if not results:
        return []
    cases = [r["cases_needed"] for r in results]
    accs = [r["accuracy"] for r in results]
    total_train = results[0].get("total_train") or 1
    mean_cases = float(np.mean(cases))
    min_c = int(np.min(cases))
    max_c = int(np.max(cases))
    min_result = next(r for r in results if r["cases_needed"] == min_c)
    max_result = next(r for r in results if r["cases_needed"] == max_c)
    stats = {
        "total_runs": total_runs,
        "converged": len(results),
        "convergence_rate": len(results) / total_runs if total_runs else 0,
        "mean_cases": mean_cases,
        "std_cases": float(np.std(cases)),
        "min_cases": min_c,
        "min_cases_class_dist": min_result.get("class_dist", ""),
        "max_cases": max_c,
        "max_cases_class_dist": max_result.get("class_dist", ""),
        "min_cases_pct": float(np.min([r["cases_pct"] for r in results])),
        "max_cases_pct": float(np.max([r["cases_pct"] for r in results])),
        "mean_accuracy": float(np.mean(accs)),
        "mean_sureness": 1.0 - (mean_cases / total_train),
    }
    if results and "hb_min_class_coverage" in results[0]:
        stats["mean_hb_count"] = float(np.mean([r["hb_count"] for r in results]))
        stats["mean_hb_min_class_coverage"] = float(
            np.mean([r["hb_min_class_coverage"] for r in results])
        )
        stats["mean_hb_weighted_coverage"] = float(
            np.mean([r["hb_weighted_coverage"] for r in results])
        )
    return [stats]


def save_config(config: Config, out_path: str) -> None:
    """Write config to txt file. Shows only the testing detail relevant to current mode."""
    lines = ["BAP Configuration", "=" * 40]
    lines.append(f"train: {config.train}")
    lines.append(f"testing: {config.testing}")
    if config.testing == "fixed":
        lines.append(f"test: {config.test}")
    elif config.testing == "split":
        lines.append(f"split: {config.split[0]}:{config.split[1]}")
    elif config.testing == "cv":
        lines.append(f"folds: {config.folds}")
    lines.extend([
        f"classifier: {config.classifier}",
        f"parameters: {config.parameters}",
        f"distance: {config.distance}",
        f"t: {config.t}",
        f"hb_coverage_t: {config.hb_coverage_t}",
        f"direction: {config.direction}",
        f"splits: {config.splits}",
        f"n: {config.n}",
        f"m: {config.m}",
        f"sampling: {config.sampling}",
        f"seed: {config.seed}",
        f"output_dir: {config.output_dir}",
    ])
    Path(out_path).write_text("\n".join(lines), encoding="utf-8")


def save_statistics(stats: list[dict[str, Any]], out_path: str) -> None:
    """Write statistics CSV."""
    if not stats:
        return
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=stats[0].keys())
        w.writeheader()
        w.writerows(stats)


# --- CLI & Main ---


def parse_args() -> tuple[Config, bool]:
    """Parse CLI and optionally load TOML. Returns (config, used_toml)."""
    parser = argparse.ArgumentParser(description="Bidirectional Active Processing")
    parser.add_argument("-c", "--config", help="TOML config file (overrides flags)")
    parser.add_argument("--train", required=False, help="Training CSV (or single dataset)")
    parser.add_argument("--test", help="Test CSV (for testing.fixed)")
    parser.add_argument("--testing", choices=["fixed", "split", "cv"], default="split")
    parser.add_argument("--split", type=str, default=",".join(map(str, DEFAULTS["split"])), help="Train,test ratio e.g. 0.8,0.2")
    parser.add_argument("--folds", type=int, default=DEFAULTS["folds"], help="Folds for CV")
    parser.add_argument("--classifier", choices=["dt", "knn", "svm", "hb_vis", "hb_dv"], default=DEFAULTS["classifier"])
    parser.add_argument("--k", type=int, default=DEFAULTS["knn_k"], help="K for KNN")
    parser.add_argument("--distance", default=DEFAULTS["distance"])
    parser.add_argument("-t", "--threshold", type=float, default=DEFAULTS["t"], dest="t")
    parser.add_argument(
        "--hb-coverage-t",
        type=float,
        default=None,
        dest="hb_coverage_t",
        help="If set, require each class's largest HB to cover at least this fraction of "
        "that class's training points (HB classifiers only), in addition to -t.",
    )
    parser.add_argument("--direction", choices=["forward", "backward"], default=DEFAULTS["direction"])
    parser.add_argument("--splits", type=int, default=DEFAULTS["splits"])
    parser.add_argument("-n", "--iterations", type=int, default=DEFAULTS["n"], dest="n")
    parser.add_argument("-m", type=int, default=DEFAULTS["m"], help="Cases per iteration")
    parser.add_argument("--sampling", choices=["random", "stratified"], default=DEFAULTS["sampling"])
    parser.add_argument("--seed", type=int, default=DEFAULTS["seed"])
    parser.add_argument("-o", "--output-dir", default=DEFAULTS["output_dir"], dest="output_dir")

    args = parser.parse_args()

    if args.config:
        cfg = Config.from_toml(args.config)
        if args.train and not cfg.train:
            cfg.train = args.train
        if args.test and not cfg.test:
            cfg.test = args.test
        # CLI overrides for common options
        if "-n" in sys.argv or "--iterations" in sys.argv:
            cfg.n = args.n
        if "-m" in sys.argv:
            cfg.m = args.m
        if "--classifier" in sys.argv:
            cfg.classifier = args.classifier
        if "--hb-coverage-t" in sys.argv:
            cfg.hb_coverage_t = args.hb_coverage_t
        return cfg, True

    cfg = Config()
    cfg.train = args.train or ""
    cfg.test = args.test or ""
    cfg.testing = args.testing
    cfg.split = [float(x.strip()) for x in args.split.split(",")]
    cfg.folds = args.folds
    cfg.classifier = args.classifier
    cfg.parameters = {"k": args.k}
    cfg.distance = args.distance
    cfg.t = getattr(args, "t", DEFAULTS["t"])
    cfg.hb_coverage_t = getattr(args, "hb_coverage_t", DEFAULTS["hb_coverage_t"])
    cfg.direction = args.direction
    cfg.splits = args.splits
    cfg.n = getattr(args, "n", DEFAULTS["n"])
    cfg.m = args.m
    cfg.sampling = args.sampling
    cfg.seed = args.seed
    cfg.output_dir = args.output_dir
    return cfg, False


def main() -> None:
    """Main entry point."""
    config, _ = parse_args()

    if not config.train or not os.path.isfile(config.train):
        print("Error: --train must point to an existing CSV.", file=sys.stderr)
        sys.exit(1)
    if config.testing == "fixed" and (not config.test or not os.path.isfile(config.test)):
        print("Error: --testing fixed requires --test to an existing CSV.", file=sys.stderr)
        sys.exit(1)
    if config.testing == "cv":
        config.splits = config.folds
        # n is total; divide across folds so total_runs = n
        div, rem = divmod(config.n, config.folds)
        n_per_split = [div + (1 if i < rem else 0) for i in range(config.folds)]
    else:
        n_per_split = [config.n] * config.splits

    np.random.seed(config.seed)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_root = os.path.join(config.output_dir, f"bap_{timestamp}")
    os.makedirs(out_root, exist_ok=True)

    save_config(config, os.path.join(out_root, "config.txt"))

    all_results: list[dict[str, Any]] = []
    total_runs = 0

    for split_id in range(1, config.splits + 1):
        split_seed = (config.seed + split_id * DEFAULTS["split_seed_multiplier"]) % DEFAULTS["seed_modulus"]
        fold_idx = (split_id - 1) if config.testing == "cv" else None
        X_train, y_train, X_test, y_test = load_data(config, split_seed, fold_idx)

        split_dir = os.path.join(out_root, f"split_{split_id}")
        os.makedirs(split_dir, exist_ok=True)

        n_this_split = n_per_split[split_id - 1]
        split_results = run_split(
            split_id, X_train, y_train, X_test, y_test, config, config.seed, split_dir, n_this_split
        )
        total_train = len(X_train)
        for r in split_results:
            r["total_train"] = total_train
        all_results.extend(split_results)
        total_runs += n_this_split

    stats = compute_statistics(all_results, total_runs)
    if stats:
        save_statistics(stats, os.path.join(out_root, "statistics.csv"))

    print(f"Results: {len(all_results)}/{total_runs} converged -> {out_root}")
    if all_results and stats:
        cases = [r["cases_needed"] for r in all_results]
        print(f"  Mean cases: {np.mean(cases):.1f} ± {np.std(cases):.1f}")
        s = stats[0]
        min_dist = s.get("min_cases_class_dist", "")
        max_dist = s.get("max_cases_class_dist", "")
        print(f"  Min: {s['min_cases']}" + (f" [{min_dist}]" if min_dist else ""))
        print(f"  Max: {s['max_cases']}" + (f" [{max_dist}]" if max_dist else ""))
        if "mean_hb_min_class_coverage" in s:
            print(
                f"  HB: mean min-class coverage {s['mean_hb_min_class_coverage']:.3f}, "
                f"weighted {s['mean_hb_weighted_coverage']:.3f}, "
                f"mean HB count {s['mean_hb_count']:.1f}"
            )


if __name__ == "__main__":
    main()
