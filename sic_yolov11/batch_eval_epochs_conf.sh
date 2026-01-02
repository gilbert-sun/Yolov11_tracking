#!/usr/bin/env bash
set -euo pipefail

# ====== User config ======
SOURCE_DIR="/media/k900/PlextorSSD0/111925_new_forprediction_all_cls0"

# 權重資料夾：內含 epoch100.pt, epoch200.pt, ...
WEIGHTS_DIR="/home/k900/Documents/ultralytics/runs/detect/train_Nmodel/weights"

# 兩支 python 程式位置（依你實際放置位置調整）
RUNNER_PY="/home/k900/Documents/ultralytics/run_yolo_and_make_lists.py"
EVAL_PY="/home/k900/Documents/ultralytics/eval_yolo_metrics.py"

# 統一總輸出目錄 (Total_Dir)
TOTAL_DIR="/home/k900/Documents/ultralytics/runs/detect/predict400"

# 評估 IoU
EVAL_IOU="0.1"

# 若你只想評估 class 0（需要時取消註解）
# CLASS_FILTER_ARGS=(--class_filter 0)
CLASS_FILTER_ARGS=(--class_filter 0)

# device 可選：例如 "0" 或 "cpu"；不想指定就留空字串
DEVICE=CPU

# imgsz 可選，不想指定就留空
IMGSZ=""

# epochs & confs（照你指定：100 起，每 400 間隔）
EPOCHS=(200)
#(600 800 1000 1200) 
#(300 600 700 800) #(100 500 900 1300 1700)
CONFS=(0.1 0.3 0.5 0.7)
# 0.9)
# =========================

mkdir -p "${TOTAL_DIR}"

echo "Total output dir: ${TOTAL_DIR}"
echo "Source dir      : ${SOURCE_DIR}"
echo "Weights dir     : ${WEIGHTS_DIR}"
echo "Runner py       : ${RUNNER_PY}"
echo "Eval py         : ${EVAL_PY}"
echo

for epoch in "${EPOCHS[@]}"; do
  MODEL_PATH="${WEIGHTS_DIR}/epoch${epoch}.pt"

  if [[ ! -f "${MODEL_PATH}" ]]; then
    echo "[WARN] Model not found, skip: ${MODEL_PATH}"
    continue
  fi

  for conf in "${CONFS[@]}"; do
    SUBDIR="${TOTAL_DIR}/epoch${epoch}_conf${conf}"
    mkdir -p "${SUBDIR}"

    GT_LIST="${SUBDIR}/gt${epoch}.txt"
    PRED_LIST="${SUBDIR}/predict${epoch}.txt"

    echo "============================================================"
    echo "[RUN] epoch=${epoch} conf=${conf}"
    echo "Model  : ${MODEL_PATH}"
    echo "Subdir : ${SUBDIR}"
    echo "GT     : ${GT_LIST}"
    echo "Pred   : ${PRED_LIST}"
    echo "============================================================"

    # ---- (1) run prediction + make gt.txt/predict.txt ----
    # 將 Ultralytics 的輸出也放在 SUBDIR 下（project=subdir, name=predict）
    RUN_ARGS=(python3 "${RUNNER_PY}"
      --model "${MODEL_PATH}"
      --source "${SOURCE_DIR}"
      --conf "${conf}"
      --out_gt "${GT_LIST}"
      --out_pred "${PRED_LIST}"
      --project "${SUBDIR}"
      --name "predict"
    )

    if [[ -n "${DEVICE}" ]]; then
      RUN_ARGS+=(--device "${DEVICE}")
    fi
    if [[ -n "${IMGSZ}" ]]; then
      RUN_ARGS+=(--imgsz "${IMGSZ}")
    fi

    echo "[1/2] Running prediction + make lists..."
    "${RUN_ARGS[@]}"

    # ---- (2) eval (save all outputs into the same SUBDIR) ----
    echo "[2/2] Evaluating..."
    python3 "${EVAL_PY}" \
      --gt_list "${GT_LIST}" \
      --pred_list "${PRED_LIST}" \
      --iou "${EVAL_IOU}" \
      --out_dir "${SUBDIR}" \
      "${CLASS_FILTER_ARGS[@]}"

    echo "[DONE] epoch=${epoch} conf=${conf} -> ${SUBDIR}"
    echo
  done
done

echo "All done. Results in: ${TOTAL_DIR}"

