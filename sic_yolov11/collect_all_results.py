#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import re
import json
import argparse
from typing import Any, Dict, List, Tuple, Optional

import pandas as pd


# =========================
# Model sort order (customize as needed)
# =========================
MODEL_ORDER = {
    "11n": 0,
    "11s": 1,
    "11m": 2,
    "11l": 3,
    "11x": 4,
    "11p2": 5,
}

DEFAULT_METRICS_NAME = "metrics.json"

DEFAULT_TOTAL_DIR = "/home/k900/Documents/ultralytics/runs/detect/predict500"

# =========================
# Utils
# =========================
def flatten_dict(d: Dict[str, Any], prefix: str = "", sep: str = ".") -> Dict[str, Any]:
    """Flatten nested dict into one-level dict with dot-keys."""
    out = {}
    for k, v in d.items():
        key = f"{prefix}{sep}{k}" if prefix else str(k)
        if isinstance(v, dict):
            out.update(flatten_dict(v, prefix=key, sep=sep))
        else:
            out[key] = v
    return out


def safe_read_json(path: str) -> Optional[Dict[str, Any]]:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def parse_run_name(run_name: str) -> Tuple[int, float, int, str]:
    """
    Parse run folder name like:
        epoch100_conf0.1_11n
        epoch500_conf0.3_11p2
    Sorting key:
        (epoch, conf, model_rank, model_str)

    If parsing fails, puts it at the end.
    """
    epoch = 10**12
    conf = 10**12
    model = "unknown"

    m_epoch = re.search(r"epoch(\d+)", run_name)
    m_conf = re.search(r"conf([\d.]+)", run_name)
    # model suffix must be at end: _11n / _11x / _11p2 / _11s ...
    m_model = re.search(r"_(11[a-z0-9]+)$", run_name)

    if m_epoch:
        epoch = int(m_epoch.group(1))
    if m_conf:
        conf = float(m_conf.group(1))
    if m_model:
        model = m_model.group(1)

    model_rank = MODEL_ORDER.get(model, 999)
    return (epoch, conf, model_rank, model)


def find_run_dirs(total_dir: str) -> List[str]:
    """
    Find subdirectories under total_dir that match pattern epoch*_conf*_* (recommended).
    If you still have old names epoch*_conf* without model suffix, they will also be included,
    but model will be 'unknown' and sorted after known models for same epoch/conf.
    """
    runs = []
    if not os.path.isdir(total_dir):
        return runs

    for name in os.listdir(total_dir):
        full = os.path.join(total_dir, name)
        if not os.path.isdir(full):
            continue
        # accept:
        # epoch100_conf0.1_11n
        # epoch100_conf0.1 (legacy)
        if re.search(r"epoch\d+_conf[\d.]+", name):
            runs.append(name)
    return runs


def main():
    ap = argparse.ArgumentParser(description="Collect metrics.json from epoch/conf/model runs and build all_results.csv")
    ap.add_argument("--total_dir", default =DEFAULT_TOTAL_DIR , help="Directory containing run subfolders (epoch*_conf*_*).")
    ap.add_argument("--out_csv", default="/home/k900/Documents/ultralytics/runs/detect/predict500/all_results_recalc_percent.csv", help="Output CSV path. Default: <total_dir>/all_results.csv")
    ap.add_argument("--metrics_name", default=DEFAULT_METRICS_NAME, help="metrics json filename (default: metrics.json)")
    ap.add_argument("--recursive", action="store_true",
                    help="Search metrics.json recursively under each run dir (slower). Default: only run_dir/metrics.json")
    ap.add_argument("--print_missing", action="store_true", help="Print runs that are missing metrics.json")
    args = ap.parse_args()

    total_dir = args.total_dir
    out_csv = args.out_csv or os.path.join(total_dir, "all_results.csv")

    run_names = find_run_dirs(total_dir)
    if not run_names:
        raise SystemExit(f"[ERROR] No run dirs found in: {total_dir}")

    # sort by epoch -> conf -> model_rank
    run_names = sorted(run_names, key=parse_run_name)

    rows: List[Dict[str, Any]] = []
    missing: List[str] = []

    for run_name in run_names:
        run_dir = os.path.join(total_dir, run_name)

        metrics_path = os.path.join(run_dir, args.metrics_name)
        found_path = None

        if os.path.isfile(metrics_path):
            found_path = metrics_path
        elif args.recursive:
            # find first metrics.json under run_dir
            for root, _, files in os.walk(run_dir):
                if args.metrics_name in files:
                    found_path = os.path.join(root, args.metrics_name)
                    break

        if not found_path:
            missing.append(run_name)
            continue

        data = safe_read_json(found_path)
        if data is None:
            missing.append(run_name)
            continue

        flat = flatten_dict(data)

        epoch, conf, model_rank, model = parse_run_name(run_name)

        row = {
            "run_name": run_name,
            "epoch_conf_model": run_name,     # same as run_name (for clarity)
            "epoch": epoch if epoch != 10**12 else None,
            "conf": conf if conf != 10**12 else None,
            "model": model,
            "model_rank": model_rank,
            "metrics_json": found_path,
        }
        # merge all flattened metrics
        row.update(flat)
        rows.append(row)

    if not rows:
        raise SystemExit("[ERROR] No valid metrics.json parsed. Check paths/filenames.")

    df = pd.DataFrame(rows)

    # final sort to guarantee stable order in CSV
    # (epoch, conf, model_rank, run_name)
    df = df.sort_values(by=["epoch", "conf", "model_rank", "run_name"], ascending=[True, True, True, True])

    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    df.to_csv(out_csv, index=False, encoding="utf-8-sig")
    print(f"[OK] Saved: {out_csv}")
    print(f"[INFO] Rows: {len(df)}")

    if args.print_missing and missing:
        print("\n[WARN] Missing or invalid metrics.json in these run dirs:")
        for m in missing:
            print("  -", m)


if __name__ == "__main__":
    main()

