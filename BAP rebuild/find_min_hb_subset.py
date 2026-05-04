#!/usr/bin/env python3
"""
Find minimum-cardinality subset of hyperblocks that achieves the best
own-class coverage on a case CSV.

Reads BAP-style hyperblocks + cases CSVs, normalizes cases with the same
min/max scope used to fit the hyperblocks (default: all-data, matching
results/iris_all_data_hb_stage2_*/normalization.txt), and solves a per-class
exact set-cover by brute force over the (small) number of class HBs.

Outputs a summary to stdout and optionally writes a filtered hyperblocks CSV
containing only the selected HBs.
"""

from __future__ import annotations

import argparse
import re
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

from hb_dv import Hyperblock

EXPORT_CLASS_COL = "class"
_HB_CLASS_RE = re.compile(r"^(.+)__HB(\d+)__(bottom|top)$")


def parse_hyperblocks_csv(hb_path: Path) -> list[tuple[int, Hyperblock]]:
    """Return list of (hb_id, Hyperblock) preserving the original HB index."""
    df = pd.read_csv(hb_path)
    feat_cols = [c for c in df.columns if c != EXPORT_CLASS_COL]
    by_hb: dict[tuple[int, str], dict[str, pd.Series]] = {}
    for _, row in df.iterrows():
        m = _HB_CLASS_RE.match(str(row[EXPORT_CLASS_COL]))
        if not m:
            continue
        base_cls, hid, kind = m.group(1), int(m.group(2)), m.group(3)
        by_hb.setdefault((hid, base_cls), {})[kind] = row
    out: list[tuple[int, Hyperblock]] = []
    for (hid, cls), parts in sorted(by_hb.items()):
        if "bottom" in parts and "top" in parts:
            mins = np.array([float(parts["bottom"][c]) for c in feat_cols], dtype=float)
            maxs = np.array([float(parts["top"][c]) for c in feat_cols], dtype=float)
            out.append((hid, Hyperblock(mins, maxs, cls)))
    return out, feat_cols


def normalize(X: pd.DataFrame, scope: str) -> pd.DataFrame:
    if scope == "all":
        mn, mx = X.min(), X.max()
    else:
        raise ValueError(f"Only scope='all' is implemented, got {scope}")
    rng = (mx - mn).replace(0, 1)
    return (X - mn) / rng


def coverage_matrix(hbs: list[Hyperblock], X: np.ndarray) -> np.ndarray:
    """B[i, j] = True if HB i contains point j."""
    n_hb, n_pts = len(hbs), X.shape[0]
    B = np.zeros((n_hb, n_pts), dtype=bool)
    for i, hb in enumerate(hbs):
        in_box = np.all((X >= hb.mins) & (X <= hb.maxs), axis=1)
        B[i] = in_box
    return B


def min_set_cover(B: np.ndarray) -> tuple[list[int], int]:
    """Exact min-cardinality set cover on small problems via brute force.

    B[i, j] = True if set i covers element j. Returns (selected indices,
    covered count). If full cover impossible, returns the smallest subset
    achieving max coverage.
    """
    n_sets, n_elems = B.shape
    target_universe = np.any(B, axis=0)
    target = int(target_universe.sum())
    sets = [np.flatnonzero(B[i]) for i in range(n_sets)]

    for k in range(1, n_sets + 1):
        best_cov = -1
        best_choice: tuple[int, ...] | None = None
        for combo in combinations(range(n_sets), k):
            covered = np.zeros(n_elems, dtype=bool)
            for i in combo:
                covered[sets[i]] = True
            c = int(covered.sum())
            if c > best_cov:
                best_cov = c
                best_choice = combo
                if c == target:
                    break
        if best_cov == target:
            return list(best_choice), best_cov
    return [], 0


def best_per_size(B: np.ndarray) -> list[tuple[int, int, tuple[int, ...]]]:
    """For each k=1..n_sets, return (k, max_covered, best_combo)."""
    n_sets, n_elems = B.shape
    out: list[tuple[int, int, tuple[int, ...]]] = []
    sets = [np.flatnonzero(B[i]) for i in range(n_sets)]
    for k in range(1, n_sets + 1):
        best_cov = -1
        best_choice: tuple[int, ...] = ()
        for combo in combinations(range(n_sets), k):
            covered = np.zeros(n_elems, dtype=bool)
            for i in combo:
                covered[sets[i]] = True
            c = int(covered.sum())
            if c > best_cov:
                best_cov, best_choice = c, combo
        out.append((k, best_cov, best_choice))
    return out


def main() -> None:
    here = Path(__file__).resolve().parent
    default_root = here / "results" / "iris_all_data_hb_stage2_20260407_003515"
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cases", type=Path, default=default_root / "all_data_hb_dv_cases.csv")
    p.add_argument(
        "--hyperblocks",
        type=Path,
        default=default_root / "all_data_hb_dv_hyperblocks.csv",
    )
    p.add_argument("--scope", choices=("all",), default="all")
    p.add_argument(
        "--mode",
        choices=("own-class", "any-class"),
        default="own-class",
        help="own-class: each point must lie in an HB of its own class. "
        "any-class: each point must lie in any HB.",
    )
    p.add_argument(
        "--out-hbs",
        type=Path,
        default=None,
        help="Optional filtered hyperblocks CSV to write.",
    )
    args = p.parse_args()

    hbs_with_id, feat_cols = parse_hyperblocks_csv(args.hyperblocks)
    if not hbs_with_id:
        raise SystemExit(f"No hyperblocks parsed from {args.hyperblocks}")
    df = pd.read_csv(args.cases)
    y = df[EXPORT_CLASS_COL].astype(str).to_numpy()
    X = normalize(df[feat_cols], args.scope)
    Xa = np.asarray(X.values, dtype=float)

    print(f"Cases:        {args.cases}")
    print(f"Hyperblocks:  {args.hyperblocks}")
    print(f"Norm scope:   {args.scope}    Coverage rule: {args.mode}")
    print(f"#HBs: {len(hbs_with_id)}    #points: {len(y)}    features: {feat_cols}")
    print()

    if args.mode == "own-class":
        classes = sorted(set(hb.class_ for _, hb in hbs_with_id))
        selected_global: list[int] = []
        rows = []
        per_class_pareto: dict[str, list[tuple[int, int, list[int]]]] = {}
        for cls in classes:
            cls_idx = np.flatnonzero(y == cls)
            cls_hb = [(hid, hb) for hid, hb in hbs_with_id if hb.class_ == cls]
            local_hbs = [hb for _, hb in cls_hb]
            B = coverage_matrix(local_hbs, Xa[cls_idx])
            pareto = best_per_size(B)
            per_class_pareto[cls] = [
                (k, cov, [cls_hb[i][0] for i in combo]) for k, cov, combo in pareto
            ]
            sel_local, covered = min_set_cover(B)
            chosen_ids = [cls_hb[i][0] for i in sel_local]
            selected_global.extend(chosen_ids)
            rows.append(
                {
                    "class": cls,
                    "n_points": len(cls_idx),
                    "n_hbs_class": len(cls_hb),
                    "min_subset_size": len(sel_local),
                    "covered": covered,
                    "coverage_pct": 100.0 * covered / max(1, len(cls_idx)),
                    "chosen_hb_ids": chosen_ids,
                }
            )
        print("Per-class Pareto (size k -> max own-class points covered):")
        for cls in classes:
            n_c = sum(1 for v in y if v == cls)
            print(f"  {cls} ({n_c} points):")
            prev = -1
            for k, cov, ids in per_class_pareto[cls]:
                tag = "  *" if cov > prev else ""
                print(
                    f"    k={k}  covered={cov}/{n_c}  "
                    f"({100*cov/n_c:.2f}%)  HBs={ids}{tag}"
                )
                prev = max(prev, cov)
        print()
        print("Per-class minimum HB subset (max-coverage at smallest k):")
        for r in rows:
            print(
                f"  {r['class']:<11} pts={r['n_points']:>3}  HBs={r['n_hbs_class']:>2}"
                f"  min_subset={r['min_subset_size']}  covered={r['covered']}/{r['n_points']}"
                f"  ({r['coverage_pct']:.2f}%)  chosen=HB{sorted(r['chosen_hb_ids'])}"
            )
        total_pts = sum(r["n_points"] for r in rows)
        total_cov = sum(r["covered"] for r in rows)
        print()
        print(f"Total selected HBs: {len(selected_global)} / {len(hbs_with_id)}")
        print(f"Selected HB ids:    {sorted(selected_global)}")
        print(
            f"Overall coverage:   {total_cov}/{total_pts} "
            f"({100.0 * total_cov / max(1, total_pts):.2f}%)"
        )
    else:
        local_hbs = [hb for _, hb in hbs_with_id]
        B = coverage_matrix(local_hbs, Xa)
        sel, covered = min_set_cover(B)
        chosen_ids = [hbs_with_id[i][0] for i in sel]
        print("Any-class union cover:")
        print(f"  min_subset={len(sel)}  covered={covered}/{len(y)}")
        print(f"  chosen=HB{sorted(chosen_ids)}")
        selected_global = chosen_ids

    if args.out_hbs:
        df_hb = pd.read_csv(args.hyperblocks)
        keep_labels = set()
        for hid, hb in hbs_with_id:
            if hid in selected_global:
                keep_labels.add(f"{hb.class_}__HB{hid}__bottom")
                keep_labels.add(f"{hb.class_}__HB{hid}__top")
        out = df_hb[df_hb[EXPORT_CLASS_COL].isin(keep_labels)].reset_index(drop=True)
        args.out_hbs.parent.mkdir(parents=True, exist_ok=True)
        out.to_csv(args.out_hbs, index=False)
        print(f"\nWrote filtered hyperblocks: {args.out_hbs.resolve()}")


if __name__ == "__main__":
    main()
