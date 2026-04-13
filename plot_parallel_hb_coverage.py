#!/usr/bin/env python3
"""
Parallel-coordinates figures from BAP converged exports:
  (A) Per target class: only cases of that class inside at least one HB labeled that class, plus HB edges.
  (B) HB edges for that class + only cases of that class not inside any HB labeled that class.

Emits up to three figure sets per best converged run:
  - subset: best-coverage converged case CSV + its hyperblocks.
  - full150: all Iris rows in train min–max space + those same hyperblocks.
  - merged: if split_* contains merged_all_converged_cases.csv and
    merged_all_converged_hb_dv_hyperblocks.csv, union deduped cases + HBs refit on that union.
  - stage2pool / stage2subset: pass --stage2-results <bap_dir_on_filtered_csv>; uses config.txt
    train path, same split seed, best converged stage-2 HBs on full filtered pool vs best subset.

Figure titles: class, In/Out HB as k/n of that class in frame, min–max scope (Full vs Subset), HB edges.

Picks the converged run with best HB coverage (min per-class largest-HB coverage, then weighted).

Uses a second min–max pass over the union of plotted coordinates + HB corners for axis scaling.
Class colors are fixed across all figures.

Also writes `{stem}_coverage_statistics.txt` and `{stem}_hyperblock_statistics.csv` (coverage, case counts, per-HB geometry).
"""

from __future__ import annotations

import argparse
import csv
import re
from datetime import datetime, timezone
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

from hyperblock.hb_dv import Hyperblock
from bap import (
    DEFAULTS,
    EXPORT_CLASS_COL,
    compute_hb_largest_per_class_coverage,
    load_csv,
    split_data,
)


def read_train_csv_from_bap_config_txt(config_path: Path) -> Path:
    """Parse `train:` path from a BAP `config.txt` next to split folders."""
    for line in config_path.read_text(encoding="utf-8").splitlines():
        s = line.strip()
        if s.startswith("train:"):
            return Path(s.split(":", 1)[1].strip())
    raise ValueError(f"No train: line in {config_path}")


def tabular_full_in_train_space(
    tabular_path: Path,
    train_ratio: float,
    test_ratio: float,
    split_seed: int,
) -> tuple[pd.DataFrame, pd.Series]:
    """
    All rows of a tabular CSV, min–max scaled with the same train-column min/max as BAP
    (stratified train split, then transform every row).
    """
    X, y = load_csv(str(tabular_path))
    X_train, _y_train, _X_test, _y_test = split_data(X, y, train_ratio, test_ratio, split_seed)
    mn = X_train.min()
    mx = X_train.max()
    rng = (mx - mn).replace(0, 1)
    X_norm = (X - mn) / rng
    return X_norm, y


_HB_CLASS_RE = re.compile(r"^(.+)__HB(\d+)__(bottom|top)$")


def parse_hyperblocks_csv(hb_path: Path) -> list[Hyperblock]:
    df = pd.read_csv(hb_path)
    class_col = EXPORT_CLASS_COL
    feat_cols = [c for c in df.columns if c != class_col]
    by_hb: dict[tuple[int, str], dict[str, pd.Series]] = {}
    for _, row in df.iterrows():
        m = _HB_CLASS_RE.match(str(row[class_col]))
        if not m:
            continue
        base_cls, hid, kind = m.group(1), int(m.group(2)), m.group(3)
        key = (hid, base_cls)
        by_hb.setdefault(key, {})[kind] = row
    hbs: list[Hyperblock] = []
    for (hid, base_cls), parts in sorted(by_hb.items()):
        if "bottom" not in parts or "top" not in parts:
            continue
        bot = parts["bottom"]
        top = parts["top"]
        mins = np.array([float(bot[c]) for c in feat_cols], dtype=float)
        maxs = np.array([float(top[c]) for c in feat_cols], dtype=float)
        hbs.append(Hyperblock(mins, maxs, base_cls))
    return hbs


class _HBClf:
    __slots__ = ("hyperblocks_",)

    def __init__(self, hbs: list[Hyperblock]) -> None:
        self.hyperblocks_ = hbs


def load_cases_csv(path: Path) -> tuple[pd.DataFrame, pd.Series]:
    df = pd.read_csv(path)
    y = df[EXPORT_CLASS_COL].astype(str)
    X = df.drop(columns=[EXPORT_CLASS_COL])
    return X, y


def inside_any_hb_of_label(hbs: list[Hyperblock], x: np.ndarray, label: str) -> bool:
    for hb in hbs:
        if str(hb.class_) != str(label):
            continue
        if hb.contains(x):
            return True
    return False


def inside_any_hyperblock(hbs: list[Hyperblock], x: np.ndarray) -> bool:
    return any(hb.contains(x) for hb in hbs)


def _per_hb_point_counts(
    X: pd.DataFrame,
    y: pd.Series,
    hbs: list[Hyperblock],
) -> list[dict[str, int | str]]:
    feat_cols = list(X.columns)
    Xa = np.asarray(X[feat_cols].values, dtype=float)
    ys = np.asarray(y.astype(str))
    rows: list[dict[str, int | str]] = []
    for j, hb in enumerate(hbs):
        inside_mask = np.array([hb.contains(Xa[i]) for i in range(len(Xa))], dtype=bool)
        n_inside = int(np.sum(inside_mask))
        same = inside_mask & (ys == str(hb.class_))
        n_match = int(np.sum(same))
        n_mismatch = n_inside - n_match
        rows.append(
            {
                "hb_index": j,
                "hb_class": str(hb.class_),
                "n_points_inside": n_inside,
                "n_points_same_class": n_match,
                "n_points_other_class": n_mismatch,
            }
        )
    return rows


def _scope_summary(
    X: pd.DataFrame,
    y: pd.Series,
    hbs: list[Hyperblock],
    scope_name: str,
) -> dict[str, object]:
    feat_cols = list(X.columns)
    Xa = np.asarray(X[feat_cols].values, dtype=float)
    ys = np.asarray(y.astype(str))
    classes = sorted(np.unique(ys).tolist())
    n = len(X)
    clf = _HBClf(hbs)
    min_cov, per_frac, per_n, hb_count = compute_hb_largest_per_class_coverage(clf, X, y)
    y_series = y.astype(str)
    w_cov = sum(per_n.values()) / n if n else 0.0

    n_any_hb = int(sum(1 for i in range(n) if inside_any_hyperblock(hbs, Xa[i])))

    by_class: dict[str, dict[str, float | int]] = {}
    for c in classes:
        mask = ys == c
        idx = np.where(mask)[0]
        n_c = int(len(idx))
        n_in = int(sum(1 for i in idx if inside_any_hb_of_label(hbs, Xa[i], c)))
        n_out = n_c - n_in
        by_class[c] = {
            "n": n_c,
            "n_in_own_label_hb": n_in,
            "n_out_own_label_hb": n_out,
            "frac_in_own_label_hb": (n_in / n_c) if n_c else 0.0,
        }

    class_dist = {c: int(np.sum(ys == c)) for c in classes}
    return {
        "scope_name": scope_name,
        "n_rows": n,
        "n_hyperblocks": hb_count,
        "hb_min_class_coverage": float(min_cov),
        "hb_weighted_coverage": float(w_cov),
        "per_class_largest_hb_frac": {k: float(v) for k, v in per_frac.items()},
        "per_class_largest_hb_count": {k: int(v) for k, v in per_n.items()},
        "n_points_in_any_hb": n_any_hb,
        "frac_points_in_any_hb": (n_any_hb / n) if n else 0.0,
        "by_class": by_class,
        "class_dist": class_dist,
    }


def write_coverage_statistics(
    out_dir: Path,
    stem: str,
    hbs: list[Hyperblock],
    X_sub: pd.DataFrame,
    y_sub: pd.Series,
    X_full: pd.DataFrame,
    y_full: pd.Series,
    extra_meta: dict[str, object],
    hb_csv_path: Path,
    *,
    subset_scope_name: str = "Best subset (converged case CSV)",
    full_scope_name: str = "Full Iris (train min–max space)",
) -> tuple[Path, Path]:
    """Write human-readable summary + per-HB CSV (subset and full-150 counts)."""
    feat_cols = list(X_sub.columns)
    sub_counts = _per_hb_point_counts(X_sub, y_sub, hbs)
    full_counts = _per_hb_point_counts(X_full, y_full, hbs)
    sub_sum = _scope_summary(X_sub, y_sub, hbs, subset_scope_name)
    full_sum = _scope_summary(X_full, y_full, hbs, full_scope_name)

    csv_path = out_dir / f"{stem}_hyperblock_statistics.csv"
    fieldnames = (
        ["hb_index", "hb_class"]
        + [f"{c}_min" for c in feat_cols]
        + [f"{c}_max" for c in feat_cols]
        + [
            "subset_n_inside",
            "subset_n_same_class",
            "subset_n_other_class",
            "full150_n_inside",
            "full150_n_same_class",
            "full150_n_other_class",
        ]
    )
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for j, hb in enumerate(hbs):
            row: dict[str, object] = {
                "hb_index": j,
                "hb_class": hb.class_,
            }
            for d, c in enumerate(feat_cols):
                row[f"{c}_min"] = float(hb.mins[d])
                row[f"{c}_max"] = float(hb.maxs[d])
            row["subset_n_inside"] = sub_counts[j]["n_points_inside"]
            row["subset_n_same_class"] = sub_counts[j]["n_points_same_class"]
            row["subset_n_other_class"] = sub_counts[j]["n_points_other_class"]
            row["full150_n_inside"] = full_counts[j]["n_points_inside"]
            row["full150_n_same_class"] = full_counts[j]["n_points_same_class"]
            row["full150_n_other_class"] = full_counts[j]["n_points_other_class"]
            w.writerow(row)

    def fmt_scope(s: dict[str, object]) -> list[str]:
        lines = [
            f"  Rows: {s['n_rows']}",
            f"  Hyperblocks: {s['n_hyperblocks']}",
            f"  BAP min (largest HB per class): {s['hb_min_class_coverage']:.6f}",
            f"  BAP weighted coverage: {s['hb_weighted_coverage']:.6f}",
            f"  Points inside any HB: {s['n_points_in_any_hb']} ({100 * float(s['frac_points_in_any_hb']):.2f}% of rows)",
            "  Largest-HB fraction per class (training definition): "
            + ", ".join(f"{k}={v:.4f}" for k, v in sorted(s["per_class_largest_hb_frac"].items())),
            "  Class row counts: " + ", ".join(f"{k}={v}" for k, v in sorted(s["class_dist"].items())),
            "  Own-label HB containment (in / total for that class):",
        ]
        for c in sorted(s["by_class"].keys()):
            bc = s["by_class"][c]
            lines.append(
                f"    {c}: {bc['n_in_own_label_hb']}/{bc['n']} in "
                f"({100 * float(bc['frac_in_own_label_hb']):.2f}%), "
                f"outside {bc['n_out_own_label_hb']}"
            )
        return lines

    txt_path = out_dir / f"{stem}_coverage_statistics.txt"
    lines_out = [
        f"HB coverage statistics — {stem}",
        f"Generated (UTC): {datetime.now(timezone.utc).isoformat(timespec='seconds')}",
        "",
        "## Metadata",
    ]
    for k, v in sorted(extra_meta.items(), key=lambda kv: kv[0]):
        lines_out.append(f"  {k}: {v}")
    lines_out.extend(
        [
            f"  hyperblocks_source: {hb_csv_path}",
            "",
            "## Hyperblock count",
            f"  {len(hbs)} axis-aligned HBs (export order: hb_index 0..{len(hbs) - 1})",
            "",
            "## Scope summaries",
            "",
            "### " + str(sub_sum["scope_name"]),
            *fmt_scope(sub_sum),
            "",
            "### " + str(full_sum["scope_name"]),
            *fmt_scope(full_sum),
            "",
            "## Per-hyperblock point counts",
            "  Columns: see companion CSV:",
            f"    {csv_path.name}",
            "  For each HB: how many data points from each scope fall inside the box;",
            "  same_class = label matches HB label; other_class = inside box but different label.",
            "",
            "## Feature columns (parallel coordinates)",
            "  " + ", ".join(feat_cols),
        ]
    )
    txt_path.write_text("\n".join(lines_out) + "\n", encoding="utf-8")
    return txt_path, csv_path


def score_run(cases_path: Path, hbs: list[Hyperblock], n_train_total: int) -> tuple[float, float, float, int]:
    X, y = load_cases_csv(cases_path)
    clf = _HBClf(hbs)
    min_cov, _, per_n, _ = compute_hb_largest_per_class_coverage(clf, X, y)
    y_arr = np.asarray(y.astype(str))
    w_cov = sum(per_n.values()) / len(y_arr) if len(y_arr) else 0.0
    cases_pct = len(y_arr) / n_train_total * 100 if n_train_total else 0.0
    return float(min_cov), float(w_cov), float(cases_pct), len(y_arr)


def pick_best_run(split_dir: Path, n_train_total: int) -> tuple[Path, list[Hyperblock], dict[str, float | int]]:
    best: tuple[tuple, Path, list[Hyperblock]] | None = None
    for cases_path in sorted(split_dir.glob("converged_exp_*_seed*.csv")):
        if cases_path.name.endswith("_hyperblocks.csv"):
            continue
        hb_path = cases_path.with_name(cases_path.stem + "_hyperblocks.csv")
        if not hb_path.is_file():
            continue
        hbs = parse_hyperblocks_csv(hb_path)
        if not hbs:
            continue
        min_cov, w_cov, cases_pct, n_cases = score_run(cases_path, hbs, n_train_total)
        key = (min_cov, w_cov, cases_pct, n_cases)
        if best is None or key > best[0]:
            best = (key, cases_path, hbs)
    if best is None:
        raise FileNotFoundError(f"No converged case/hyperblock pairs under {split_dir}")
    (_k, cases_path, hbs) = best
    min_cov, w_cov, cases_pct, n_cases = score_run(cases_path, hbs, n_train_total)
    meta = {
        "cases_path": str(cases_path),
        "hb_min_class_coverage": min_cov,
        "hb_weighted_coverage": w_cov,
        "cases_pct": cases_pct,
        "n_cases": n_cases,
        "n_hbs": len(hbs),
    }
    return cases_path, hbs, meta


def ordered_classes(y: pd.Series) -> list[str]:
    return sorted(y.unique().tolist())


def parse_split_id(split_dir: Path) -> int:
    name = split_dir.name
    if name.startswith("split_"):
        return int(name.split("_", 1)[1])
    return 1


def split_seed_for_bap(config_seed: int, split_id: int) -> int:
    return (config_seed + split_id * DEFAULTS["split_seed_multiplier"]) % DEFAULTS["seed_modulus"]


def iris_full_in_bap_train_space(
    iris_path: Path,
    train_ratio: float,
    test_ratio: float,
    split_seed: int,
) -> tuple[pd.DataFrame, pd.Series]:
    """Iris-specific wrapper for `tabular_full_in_train_space`."""
    return tabular_full_in_train_space(iris_path, train_ratio, test_ratio, split_seed)


def build_color_map(classes: list[str]) -> dict[str, tuple]:
    palette = plt.cm.tab10(np.linspace(0, 0.9, max(len(classes), 3)))
    return {c: tuple(palette[i % len(palette)]) for i, c in enumerate(classes)}


def minmax_for_arrays(
    feat_cols: list[str],
    case_X: pd.DataFrame,
    hbs: list[Hyperblock],
) -> tuple[MinMaxScaler, list[str]]:
    rows = [case_X[feat_cols].to_numpy()]
    for hb in hbs:
        rows.append(hb.mins.reshape(1, -1))
        rows.append(hb.maxs.reshape(1, -1))
    all_rows = np.vstack(rows)
    scaler = MinMaxScaler()
    scaler.fit(all_rows)
    return scaler, feat_cols


def plot_parallel(
    ax: plt.Axes,
    feat_cols: list[str],
    scaler: MinMaxScaler,
    lines: list[tuple[np.ndarray, str, float] | tuple[np.ndarray, str, float, float]],
    title: str,
    color_by_class: dict[str, tuple],
    linewidth: float = 1.2,
    alpha: float = 0.35,
) -> None:
    xs = np.arange(len(feat_cols))
    ax.set_xticks(xs)
    ax.set_xticklabels(feat_cols, rotation=25, ha="right")
    ax.set_ylim(-0.02, 1.02)
    ax.set_ylabel("min–max normalized")
    for item in lines:
        if len(item) == 4:
            vals, cls, lw, a = item
        else:
            vals, cls, lw = item
            a = alpha
        if vals.size == 0:
            continue
        v = scaler.transform(vals.reshape(1, -1)).ravel()
        col = color_by_class.get(cls, (0.4, 0.4, 0.4))
        ax.plot(xs, v, color=col, linewidth=lw, alpha=a, solid_capstyle="round")
    ax.set_title(title, fontsize=10)
    ax.grid(True, axis="y", alpha=0.3)


_SCOPE_TITLE = {
    "full150": "Full Data",
    "subset": "Best subset",
    "merged": "Merged union",
    "stage2pool": "Stage-2 pool",
    "stage2subset": "Stage-2 best subset",
}


def figure_title(
    *,
    inside: bool,
    scope_key: str,
    n_part: int,
    n_total: int,
    class_name: str,
) -> str:
    """k/n = in-HB or out-of-HB count over that class in the plotted frame."""
    scope = _SCOPE_TITLE.get(scope_key, scope_key)
    tag = "In HB" if inside else "Out HB"
    return f"{class_name} · {tag} {n_part}/{n_total} · Min–Max {scope} · HB Edges"


def draw_class_figures(
    X: pd.DataFrame,
    y: pd.Series,
    hbs: list[Hyperblock],
    out_dir: Path,
    stem: str,
    scope_key: str,
) -> None:
    feat_cols = list(X.columns)
    classes = ordered_classes(y)
    color_by_class = build_color_map(classes)

    scaler, _ = minmax_for_arrays(feat_cols, X, hbs)
    Xa = np.asarray(X.values, dtype=float)

    y_str = y.astype(str)
    for target in classes:
        is_target = (y_str == str(target)).to_numpy()
        inside_hb = np.array(
            [inside_any_hb_of_label(hbs, Xa[i], target) for i in range(len(Xa))],
            dtype=bool,
        )
        inside_this_class = is_target & inside_hb
        outside_this_class = is_target & ~inside_hb
        n_class = int(np.sum(is_target))
        n_in = int(np.sum(inside_this_class))
        n_out = int(np.sum(outside_this_class))

        # (A) Target-class cases inside target HBs + same-class HB edge polylines (edges last, on top)
        fig, ax = plt.subplots(figsize=(8, 5))
        lines_a: list[tuple[np.ndarray, str, float] | tuple[np.ndarray, str, float, float]] = []
        for i in np.where(inside_this_class)[0]:
            lines_a.append((Xa[i], str(y.iloc[i]), 1.2, 0.35))
        for hb in hbs:
            if str(hb.class_) != str(target):
                continue
            base = str(hb.class_)
            lines_a.append((hb.mins, base, 2.8, 0.95))
            lines_a.append((hb.maxs, base, 2.8, 0.95))
        plot_parallel(
            ax,
            feat_cols,
            scaler,
            lines_a,
            figure_title(
                inside=True,
                scope_key=scope_key,
                n_part=n_in,
                n_total=max(n_class, 1),
                class_name=str(target),
            ),
            color_by_class,
        )
        h_l = [ax.plot([], [], color=color_by_class[c], label=c, linewidth=3)[0] for c in classes]
        ax.legend(handles=h_l, loc="upper right", framealpha=0.9)
        fig.tight_layout()
        fig.savefig(out_dir / f"{stem}_{scope_key}_{target}_inside.png", dpi=160)
        plt.close(fig)

        # (B) Target-class cases outside target HBs, then same HB edge style as inside (on top)
        fig, ax = plt.subplots(figsize=(8, 5))
        lines_b: list[tuple[np.ndarray, str, float] | tuple[np.ndarray, str, float, float]] = []
        for i in np.where(outside_this_class)[0]:
            lines_b.append((Xa[i], str(y.iloc[i]), 1.2, 0.35))
        for hb in hbs:
            if str(hb.class_) != str(target):
                continue
            base = str(hb.class_)
            lines_b.append((hb.mins, base, 2.8, 0.95))
            lines_b.append((hb.maxs, base, 2.8, 0.95))
        plot_parallel(
            ax,
            feat_cols,
            scaler,
            lines_b,
            figure_title(
                inside=False,
                scope_key=scope_key,
                n_part=n_out,
                n_total=max(n_class, 1),
                class_name=str(target),
            ),
            color_by_class,
        )
        h_l = [ax.plot([], [], color=color_by_class[c], label=c, linewidth=3)[0] for c in classes]
        ax.legend(handles=h_l, loc="upper right", framealpha=0.9)
        fig.tight_layout()
        fig.savefig(out_dir / f"{stem}_{scope_key}_{target}_outside.png", dpi=160)
        plt.close(fig)


def draw_stage2_figures(
    *,
    stage2_results: Path,
    split_id: int,
    train_ratio: float,
    test_ratio: float,
    config_seed: int,
    out_root: Path,
) -> None:
    """Filtered pool + best stage-2 HBs (same normalization as BAP stage 2)."""
    split_seed = split_seed_for_bap(config_seed, split_id)
    s2_split = stage2_results / f"split_{split_id}"
    cfg_txt = stage2_results / "config.txt"
    if not s2_split.is_dir():
        print(f"Stage 2 split dir missing: {s2_split}")
        return
    if not cfg_txt.is_file():
        print(f"Stage 2 config missing: {cfg_txt}")
        return
    try:
        train_csv = read_train_csv_from_bap_config_txt(cfg_txt)
    except ValueError as e:
        print(f"Stage 2 config parse error: {e}")
        return
    if not train_csv.is_file():
        print(f"Stage 2 train CSV not found: {train_csv}")
        return

    n_full = len(pd.read_csv(train_csv))
    n_train = int(round(train_ratio * n_full))
    try:
        cases_s2, hbs_s2, _meta_s2 = pick_best_run(s2_split, n_train_total=n_train)
    except FileNotFoundError:
        print(f"No converged stage-2 runs under {s2_split}")
        return

    stem_s2 = f"split_{split_id}_stage2_{cases_s2.stem}"
    X_pool, y_pool = tabular_full_in_train_space(train_csv, train_ratio, test_ratio, split_seed)
    draw_class_figures(X_pool, y_pool, hbs_s2, out_root, stem_s2, scope_key="stage2pool")

    X_sub, y_sub = load_cases_csv(cases_s2)
    draw_class_figures(X_sub, y_sub, hbs_s2, out_root, stem_s2, scope_key="stage2subset")

    print(
        f"Wrote stage-2 figures ({stem_s2}_stage2pool_* and {stem_s2}_stage2subset_*) -> {out_root}"
    )


def main() -> None:
    p = argparse.ArgumentParser(description="Parallel-coordinates HB coverage figures from BAP results.")
    p.add_argument(
        "results_dir",
        type=Path,
        help="Folder containing split_*/converged_exp_*.csv (e.g. results/bap_20260330_175212)",
    )
    p.add_argument(
        "--iris",
        type=Path,
        default=Path(__file__).resolve().parent.parent.parent
        / "machine learning datasets"
        / "default"
        / "fisher_iris.csv",
        help="Original Iris CSV (for train size / split 0.8)",
    )
    p.add_argument(
        "--train-ratio",
        type=float,
        default=0.8,
        help="Train fraction (for n_train count in scoring metadata; default matches BAP config).",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=42,
        help="BAP config seed (used with split_* id for stratified split; default matches BAP).",
    )
    p.add_argument(
        "--stage2-results",
        type=Path,
        default=None,
        help="Second BAP output dir (e.g. results/bap_* on iris_excluding_hb_union). "
        "Draws stage-2 pool + best subset with best stage-2 hyperblocks.",
    )
    args = p.parse_args()

    full = pd.read_csv(args.iris)
    n = len(full)
    n_train = int(round(args.train_ratio * n))
    test_ratio = 1.0 - args.train_ratio

    split_dirs = sorted(args.results_dir.glob("split_*"))
    if not split_dirs:
        split_dirs = [args.results_dir]

    out_root = args.results_dir / "parallel_coords_figures"
    out_root.mkdir(parents=True, exist_ok=True)

    for sd in split_dirs:
        cases_path, hbs, meta = pick_best_run(sd, n_train_total=n_train)
        stem = f"{sd.name}_{cases_path.stem}_bestcov"
        split_id = parse_split_id(sd)
        split_seed = split_seed_for_bap(args.seed, split_id)
        X_sub, y_sub = load_cases_csv(cases_path)
        X_full, y_full = iris_full_in_bap_train_space(
            args.iris, args.train_ratio, test_ratio, split_seed
        )

        draw_class_figures(X_sub, y_sub, hbs, out_root, stem, scope_key="subset")
        draw_class_figures(X_full, y_full, hbs, out_root, stem, scope_key="full150")

        merged_cases = sd / "merged_all_converged_cases.csv"
        merged_hb = sd / "merged_all_converged_hb_dv_hyperblocks.csv"
        if merged_cases.is_file() and merged_hb.is_file():
            X_m, y_m = load_cases_csv(merged_cases)
            hbs_m = parse_hyperblocks_csv(merged_hb)
            if hbs_m:
                stem_m = f"{sd.name}_merged_all_converged_hb_dv"
                draw_class_figures(X_m, y_m, hbs_m, out_root, stem_m, scope_key="merged")
                print(f"Wrote merged union figures ({stem_m}_merged_*) -> {out_root}")
            else:
                print(f"Skipping merged figures: no HBs parsed from {merged_hb}")
        else:
            print(
                "Skipping merged figures (add merge_converged_fit_hb_dv.py outputs to split dir)."
            )

        hb_csv_path = cases_path.with_name(cases_path.stem + "_hyperblocks.csv")
        extra_meta: dict[str, object] = {
            "bap_seed": args.seed,
            "iris_csv": str(args.iris.resolve()),
            "n_iris_rows": n,
            "n_train_rows_bap": n_train,
            "split_dir": str(sd.resolve()),
            "split_id": split_id,
            "stratified_split_seed": split_seed,
            "train_ratio": args.train_ratio,
        }
        for k, v in meta.items():
            extra_meta[str(k)] = v
        txt_stats, csv_stats = write_coverage_statistics(
            out_root,
            stem,
            hbs,
            X_sub,
            y_sub,
            X_full,
            y_full,
            extra_meta,
            hb_csv_path,
        )

        meta_path = out_root / f"{stem}_meta.txt"
        meta_path.write_text(
            "\n".join(f"{k}: {v}" for k, v in meta.items())
            + f"\nsplit_dir: {sd}\n"
            f"split_id: {split_id}\n"
            f"bap_seed: {args.seed}\n"
            f"stratified_split_seed: {split_seed}\n"
            f"n_train_rows: {n_train}\n"
            f"n_iris_full: {n}\n"
            f"coverage_statistics: {txt_stats.name}\n"
            f"hyperblock_statistics_csv: {csv_stats.name}\n",
            encoding="utf-8",
        )
        print(f"Wrote subset + full150 figures for {stem} -> {out_root}")
        print(f"Wrote {txt_stats.name} and {csv_stats.name}")
        print(meta)

        if args.stage2_results is not None:
            draw_stage2_figures(
                stage2_results=args.stage2_results.resolve(),
                split_id=split_id,
                train_ratio=args.train_ratio,
                test_ratio=test_ratio,
                config_seed=args.seed,
                out_root=out_root,
            )


if __name__ == "__main__":
    main()
