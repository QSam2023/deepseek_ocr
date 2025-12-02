#!/bin/bash

# Define base directory (assuming script is run from eval_result)
BASE_DIR=$(pwd)

echo "============================================================"
echo "Running Table OCR Evaluation"
echo "============================================================"
if [ -f "table_ocr_eval/pred_table.json" ]; then
    python3 table_ocr_eval/eval_table_ocr.py \
        table_ocr_eval/table_ocr_label.json \
        table_ocr_eval/pred_table.json
else
    echo "Prediction file table_ocr_eval/pred_table.json not found."
fi
echo ""

echo "============================================================"
echo "Running Stamp Classification Evaluation"
echo "============================================================"
if [ -f "stamp_cls_eval/pred_stamp_cls.json" ]; then
    python3 stamp_cls_eval/eval_stamp_cls.py \
        stamp_cls_eval/stamp_cls_label.json \
        stamp_cls_eval/pred_stamp_cls.json
else
    echo "Prediction file stamp_cls_eval/pred_stamp_cls.json not found."
fi
echo ""

echo "============================================================"
echo "Running Stamp OCR Evaluation"
echo "============================================================"
if [ -f "stamp_ocr_eval/pred_stamp_ocr.json" ]; then
    python3 stamp_ocr_eval/eval_stamp_ocr.py \
        stamp_ocr_eval/stamp_ocr_label.json \
        stamp_ocr_eval/pred_stamp_ocr.json
else
    echo "Prediction file stamp_ocr_eval/pred_stamp_ocr.json not found."
fi
echo ""
