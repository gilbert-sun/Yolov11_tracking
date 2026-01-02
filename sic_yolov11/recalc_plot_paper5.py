#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import re
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# =========================================================
# Model type order (3rd sorting pattern)
# =========================================================
MODEL_ORDER = {
    "11n": 0,
    "11s": 1,
    "11m": 2,
    "11l": 3,
    "11x": 4,
    "11p2": 5,
}

# =========================================================
# Super Parameters (fixed table)
# =========================================================
SUPER_PARAMS = [
    ("model", "yolov11"),
    ("cls", "tsd:0"),
    ("epoch", "(0~2000)"),
    ("conf", "(0.1~0.9)"),
    ("IoU", "(0.1)"),
    ("batch", "(12:2~64)"),
    ("imgsz", "(640: 64~1280)"),
    ("multi_scale", "True"),
]

# =========================================================
# Plot metrics
# =========================================================
PLOT_METRICS = [
    ("accuracy", "Accuracy (%)"),
    ("precision", "Precision (%)"),
    ("recall", "Recall (%)"),
    ("f1", "F1 (%)"),
]

# =========================================================
# Helpers
# =========================================================
def parse_epoch_conf_model(key: str):
    """
    Parse:
      epoch100_conf0.1_11n
      ep100_conf0.1_11x
      epoch100_conf0.1 (legacy -> model unknown)
    """
    epoch, conf, model = None, None, "unknown"

    m_epoch = re.search(r"(?:epoch|ep)(\d+)", key)
    m_conf = re.search(r"conf([\d.]+)", key)
    m_model = re.search(r"_(11[a-z0-9]+)$", key)

    if m_epoch:
        epoch = int(m_epoch.group(1))
    if m_conf:
        conf = float(m_conf.group(1))
    if m_model:
        model = m_model.group(1)

    model_rank = MODEL_ORDER.get(model, 999)
    return epoch, conf, model, model_rank


def distinct_colors(n: int):
    # enough colors for ~60 bars; repeats if more
    cmaps = [plt.cm.tab20, plt.cm.tab20b, plt.cm.tab20c]
    colors = []
    for cmap in cmaps:
        colors.extend([cmap(i) for i in np.linspace(0, 1, 20)])
    if n <= len(colors):
        return colors[:n]
    return [colors[i % len(colors)] for i in range(n)]


def is_yolo_label_line(line: str) -> bool:
    parts = line.strip().split()
    if len(parts) < 5:
        return False
    if not parts[0].lstrip("-").isdigit():
        return False
    for t in parts[1:5]:
        try:
            float(t)
        except:
            return False
    return True


def count_classes_in_label_file(label_path: str, num_classes: int = 4):
    counts = [0] * num_classes
    try:
        with open(label_path, "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                s = line.strip()
                if not s:
                    continue
                parts = s.split()
                if parts and parts[0].lstrip("-").isdigit():
                    cid = int(parts[0])
                    if 0 <= cid < num_classes:
                        counts[cid] += 1
    except FileNotFoundError:
        return counts
    return counts


def parse_gt_txt_class_stats(gt_txt_path: str, num_classes: int = 4):
    """
    Supports:
    A) gt.txt is a list of label-file paths (common for --gt_list)
    B) gt.txt itself is a YOLO label file
    """
    if not os.path.exists(gt_txt_path):
        raise FileNotFoundError(f"gt.txt not found: {gt_txt_path}")

    first_nonempty = None
    with open(gt_txt_path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            s = line.strip()
            if s:
                first_nonempty = s
                break

    counts = [0] * num_classes
    if first_nonempty is None:
        return counts

    if is_yolo_label_line(first_nonempty):
        return count_classes_in_label_file(gt_txt_path, num_classes=num_classes)

    with open(gt_txt_path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            p = line.strip()
            if not p:
                continue
            if not os.path.isabs(p):
                p = os.path.abspath(p)
            c = count_classes_in_label_file(p, num_classes=num_classes)
            counts = [counts[i] + c[i] for i in range(num_classes)]
    return counts


# =========================================================
# Tables on the right
# =========================================================
def add_tables(ax, df_all, metric, topk, gt_counts, best_summary):
    """
    Right side tables in order:
    1) Super Parameter
    2) Top-K epoch_conf_model + TP/FP/FN/TN (sorted by metric desc, then epoch/conf/model)
    3) GT class stats table (class 0..3)
    4) Top-1 summary (epoch_conf_model, Acc, Prec, Rec, F1)
    """

    x0, w = 1.02, 0.42
    super_bbox = [x0, 0.70, w, 0.27]
    topk_bbox = [x0, 0.40, w, 0.28]
    gt_bbox = [x0, 0.22, w, 0.16]
    best_bbox = [x0, 0.04, w, 0.16]

    # (1) Super Params
    sp = ax.table(
        cellText=[[k, v] for k, v in SUPER_PARAMS],
        colLabels=["Super Parameter", "Value"],
        cellLoc="left",
        colLoc="left",
        bbox=super_bbox,
    )
    sp.auto_set_font_size(False)
    sp.set_fontsize(5)

    # (2) Top-K
    if metric in df_all.columns:
        tmp = df_all.copy()
        tmp["_m"] = pd.to_numeric(tmp[metric], errors="coerce")
        topk_df = tmp.sort_values(
            by=["_m", "epoch", "conf", "model_rank"],
            ascending=[False, True, True, True],
        ).head(int(topk)).copy()

        # build table (epoch_conf_model is string; numeric columns int)
        tcols = ["epoch_conf_model", "tp_total", "fp_total", "fn_total", "tn_total_used"]
        tbl = topk_df[tcols].copy()
        tbl["epoch_conf_model"] = tbl["epoch_conf_model"].astype(str)
        for c in ["tp_total", "fp_total", "fn_total", "tn_total_used"]:
            tbl[c] = pd.to_numeric(tbl[c], errors="coerce").fillna(0).astype(int)

        t = ax.table(
            cellText=tbl.values.tolist(),
            colLabels=["ep_cf_mod", "TP", "FP", "FN", "TN"],
            cellLoc="center",
            colLoc="center",
            bbox=topk_bbox,
        )
        t.auto_set_font_size(False)
        t.set_fontsize(3)

    # (3) GT stats
    if gt_counts is not None and len(gt_counts) >= 4:
        gt_rows = [[f"class {i}", int(gt_counts[i])] for i in range(4)]
        gtt = ax.table(
            cellText=gt_rows,
            colLabels=["GT classes (from gt.txt)", "count"],
            cellLoc="center",
            colLoc="center",
            bbox=gt_bbox,
        )
        gtt.auto_set_font_size(False)
        gtt.set_fontsize(5)

    # (4) Top-1 summary
    if best_summary is not None:
        def fmt(v):
            try:
                if np.isnan(v):
                    return "NaN"
                return f"{float(v):.2f}"
            except:
                return str(v)

        row = [[
            best_summary["epoch_conf_model"],
            fmt(best_summary["accuracy"]),
            fmt(best_summary["precision"]),
            fmt(best_summary["recall"]),
            fmt(best_summary["f1"]),
        ]]
        bt = ax.table(
            cellText=row,
            colLabels=["T1_ep_cf_mod", "Accuracy", "Precision", "Recall", "F1"],
            cellLoc="center",
            colLoc="center",
            bbox=best_bbox,
        )
        bt.auto_set_font_size(False)
        bt.set_fontsize(3)


# =========================================================
# Main
# =========================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="/home/k900/Documents/ultralytics/runs/detect/predict500/all_results_recalc_percent.csv" , help="Path to all_results.csv (col includes epoch/conf/model key)")
    ap.add_argument("--gt_txt", default="/home/k900/Documents/ultralytics/gt.txt", help="Path to gt.txt (label list or label file)")
    ap.add_argument("--out_dir", default="/home/k900/Documents/ultralytics/runs/detect/predict500/paper_plots3", help="Output plot directory")
    ap.add_argument("--dpi", type=int, default=600, help="PNG dpi")
    ap.add_argument("--sort", choices=["none", "metric"], default="none",
                    help="none: epoch/conf/model order; metric: metric desc then tie by epoch/conf/model")
    ap.add_argument("--topk", type=int, default=10, help="Top-K rows to show in Top-K table")
    ap.add_argument("--best_metric", choices=["accuracy", "precision", "recall", "f1"], default="f1",
                    help="Metric used to select global Top-1 summary row")
    ap.add_argument("--no_long_legend", action="store_true", help="Disable legend (recommended)")
    ap.add_argument("--legend_inside", action="store_true", help="Put legend inside chart (upper right)")
    ap.add_argument("--export_pdf",  choices=["none", "true"], default="none", help="Export PDF too")
    args = ap.parse_args()

    df = pd.read_csv(args.csv)

    # -------- Determine key column -> epoch_conf_model (唯一 key) --------
    # Priority: epoch_conf_model > run_name > epoch_conf
    if "epoch_conf_model" in df.columns:
        df["epoch_conf_model"] = df["epoch_conf_model"].astype(str)
    elif "run_name" in df.columns:
        df["epoch_conf_model"] = df["run_name"].astype(str)
    elif "epoch_conf" in df.columns:
        df["epoch_conf_model"] = df["epoch_conf"].astype(str)
    else:
        raise ValueError("CSV must contain one of: epoch_conf_model, run_name, epoch_conf")

    # -------- Parse epoch/conf/model_rank from epoch_conf_model --------
    parsed = df["epoch_conf_model"].apply(parse_epoch_conf_model)
    df["epoch"] = parsed.apply(lambda x: x[0])
    df["conf"] = parsed.apply(lambda x: x[1])
    df["model"] = parsed.apply(lambda x: x[2])
    df["model_rank"] = parsed.apply(lambda x: x[3])

    # -------- Ensure TP/FP/FN/TN columns exist --------
    # Accept various possible column names
    def pick_col(cands):
        for c in cands:
            if c in df.columns:
                return c
        return None

    tp_col = pick_col([
    "tp_total", "TP_total", "TP_Total", "tp",
    "micro.tp_total", "micro.TP_total", "micro.TP_Total", "micro.tp",
    ])
    fp_col = pick_col([
    "fp_total", "FP_total", "FP_Total", "fp",
    "micro.fp_total", "micro.FP_total", "micro.FP_Total", "micro.fp",
    ])
    fn_col = pick_col([
    "fn_total", "FN_total", "FN_Total", "fn",
    "micro.fn_total", "micro.FN_total", "micro.FN_Total", "micro.fn",
    ])
    tn_col = pick_col([
    "tn_total", "TN_total", "TN_Total", "tn", "tn_total_used",
    "micro.tn_total", "micro.TN_total", "micro.TN_Total", "micro.tn",
    ])


    if tp_col is None or fp_col is None or fn_col is None:
        cand = [c for c in df.columns if re.search(r"(tp|fp|fn|tn)", c, re.IGNORECASE)]
        raise ValueError(
                f"Missing TP/FP/FN columns. Found: tp={tp_col}, fp={fp_col}, fn={fn_col}\n"
                f"Columns containing tp/fp/fn/tn (for debugging):\n{cand}"
        )
        #raise ValueError(f"Missing TP/FP/FN columns. Found: tp={tp_col}, fp={fp_col}, fn={fn_col}")

    # Standardize to tp_total/fp_total/fn_total/tn_total_used
    df["tp_total"] = pd.to_numeric(df[tp_col], errors="coerce").fillna(0).astype(float)
    df["fp_total"] = pd.to_numeric(df[fp_col], errors="coerce").fillna(0).astype(float)
    df["fn_total"] = pd.to_numeric(df[fn_col], errors="coerce").fillna(0).astype(float)

    if tn_col is None:
        df["tn_total_used"] = 0.0
    else:
        df["tn_total_used"] = pd.to_numeric(df[tn_col], errors="coerce").fillna(0).astype(float)

    # -------- Recompute metrics (vectorized, safe), percent --------
    tp = df["tp_total"].to_numpy()
    fp = df["fp_total"].to_numpy()
    fn = df["fn_total"].to_numpy()
    tn = df["tn_total_used"].to_numpy()

    df["precision"] = np.where((tp + fp) > 0, tp / (tp + fp), np.nan) * 100.0
    df["recall"]    = np.where((tp + fn) > 0, tp / (tp + fn), np.nan) * 100.0
    df["f1"]        = np.where((2*tp + fp + fn) > 0, (2*tp) / (2*tp + fp + fn), np.nan) * 100.0
    df["accuracy"]  = np.where((tp + tn + fp + fn) > 0, (tp + tn) / (tp + tn + fp + fn), np.nan) * 100.0

    # -------- GT stats (class 0..3) --------
    try:
        gt_counts = parse_gt_txt_class_stats(args.gt_txt, num_classes=4)
        print(f"[OK] GT class counts: {gt_counts}")
    except Exception as e:
        gt_counts = None
        print(f"[WARN] GT stats parse failed: {e}")

    # -------- Global Top-1 summary (best_metric desc, tie by epoch/conf/model_rank) --------
    best_metric = args.best_metric
    tmpb = df.copy()
    tmpb["_b"] = pd.to_numeric(tmpb[best_metric], errors="coerce")
    tmpb = tmpb.sort_values(
        by=["_b", "epoch", "conf", "model_rank"],
        ascending=[False, True, True, True],
    )
    if len(tmpb) > 0:
        br = tmpb.iloc[0]
        best_summary = {
            "epoch_conf_model": br["epoch_conf_model"],
            "accuracy": float(br["accuracy"]),
            "precision": float(br["precision"]),
            "recall": float(br["recall"]),
            "f1": float(br["f1"]),
        }
        print(f"[OK] Top-1 by {best_metric}: {best_summary}")
    else:
        best_summary = None
        print("[WARN] No rows to pick Top-1")

    # -------- Output dir --------
    os.makedirs(args.out_dir, exist_ok=True)

    # -------- Plot 4 charts --------
    for metric, title in PLOT_METRICS:
        plot_df = df.copy()
        plot_df[metric] = pd.to_numeric(plot_df[metric], errors="coerce")

        if args.sort == "metric":
            plot_df = plot_df.sort_values(
                by=[metric, "epoch", "conf", "model_rank"],
                ascending=[False, True, True, True],
            )
        else:
            plot_df = plot_df.sort_values(
                by=["epoch", "conf", "model_rank"],
                ascending=[True, True, True],
            )

        labels = plot_df["epoch_conf_model"].astype(str).tolist()
        values = plot_df[metric].to_numpy(dtype=float)

        n = len(labels)
        x = np.arange(n)
        colors = distinct_colors(n)

        plt.figure(figsize=(22, 6), facecolor="white")
        ax = plt.gca()
        ax.set_facecolor("white")

        bars = ax.bar(x, values, color=colors, edgecolor="black", linewidth=0.4)

        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=5)
        ax.set_xlabel("epoch_conf_model", fontsize=5)
        ax.set_ylabel(title)
        ax.set_title(f"{title} per epoch/conf/model")
        ax.grid(axis="y", alpha=0.25)

        # Right-side tables (use df sorted by metric for Top-K table consistency)
        add_tables(
            ax=ax,
            df_all=df,
            metric=metric,
            topk=args.topk,
            gt_counts=gt_counts,
            best_summary=best_summary,
        )

        # Legend (optional)
        if not args.no_long_legend:
            if args.legend_inside:
                ax.legend(
                    handles=bars, labels=labels, title="model",
                    loc="upper right",
                    fontsize=3.5,
                    frameon=True, framealpha=0.85,
                    facecolor="white", edgecolor="black",
                )
            else:
                ax.legend(
                    handles=bars, labels=labels, title="model",
                    loc="center left", bbox_to_anchor=(1.02, 0.5),
                    frameon=True,
                )

        # reserve space on right for tables
        plt.tight_layout(rect=[0.0, 0.0, 0.70, 1.0])

        base = f"{metric}"
        if args.sort == "metric":
            base += "_sorted"
        base += "_with_tables"

        png_path = os.path.join(args.out_dir, f"{base}.png")
        plt.savefig(png_path, dpi=args.dpi, facecolor="white", bbox_inches="tight")
        print(f"[OK] Saved: {png_path}")

        if args.export_pdf == 'true':
            pdf_path = os.path.join(args.out_dir, f"{base}.pdf")
            plt.savefig(pdf_path, facecolor="white", bbox_inches="tight")
            print(f"[OK] Saved: {pdf_path}")

        plt.close()

    print("[DONE] All plots generated.")


if __name__ == "__main__":
    main()

