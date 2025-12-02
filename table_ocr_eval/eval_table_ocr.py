import json
import argparse
import os
import sys
from typing import Dict, Any, Tuple, List

# 添加项目根目录到path，以便导入eval_utils
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from eval_utils import load_json, compare_json, build_image_map, parse_data_structure

def evaluate(gt_path: str, pred_path: str):
    print(f"Loading Ground Truth from: {gt_path}")
    print(f"Loading Predictions from: {pred_path}")

    gt_data = load_json(gt_path)
    pred_data = load_json(pred_path)

    # Parse data structure (handle {"results": [...]} or just [...])
    gt_list = parse_data_structure(gt_data)
    pred_list = parse_data_structure(pred_data)

    # Create dictionaries keyed by image_name for easy lookup
    gt_map = build_image_map(gt_list)
    pred_map = build_image_map(pred_list)

    total_fields_all = 0
    matched_fields_all = 0
    total_pred_fields_all = 0
    
    image_count = 0
    perfect_images = 0

    print("-" * 60)
    print(f"{'Image Name':<30} | {'Accuracy':<10} | {'Missing':<8} | {'Extra':<8}")
    print("-" * 60)

    for image_name, gt_item in gt_map.items():
        if image_name not in pred_map:
            print(f"{image_name:<30} | {'MISSING':<10} | {'-':<8} | {'-':<8}")
            continue
        
        image_count += 1
        pred_item = pred_map[image_name]
        
        # We compare the 'result' field specifically
        gt_result = gt_item.get('result', {})
        pred_result = pred_item.get('result', {})

        t_gt, match, t_pred = compare_json(gt_result, pred_result)
        
        total_fields_all += t_gt
        matched_fields_all += match
        total_pred_fields_all += t_pred

        accuracy = (match / t_gt * 100) if t_gt > 0 else 0.0
        missing = t_gt - match # This is a simplification; strictly missing keys + wrong values
        extra = max(0, t_pred - t_gt) # Rough estimate of extra fields

        if accuracy == 100.0 and t_gt == t_pred:
            perfect_images += 1

        print(f"{image_name:<30} | {accuracy:6.2f}%    | {missing:<8} | {extra:<8}")

    print("-" * 60)
    
    overall_accuracy = (matched_fields_all / total_fields_all * 100) if total_fields_all > 0 else 0.0
    perfect_rate = (perfect_images / image_count * 100) if image_count > 0 else 0.0

    print(f"Total Images Evaluated: {image_count}")
    print(f"Overall Field-Level Accuracy: {overall_accuracy:.2f}%")
    print(f"Perfect Image Match Rate: {perfect_rate:.2f}%")
    print(f"Total GT Fields: {total_fields_all}")
    print(f"Total Matched Fields: {matched_fields_all}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate Table OCR JSON results.")
    parser.add_argument("gt_file", help="Path to ground truth JSON file")
    parser.add_argument("pred_file", help="Path to prediction JSON file")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.gt_file):
        print(f"Error: Ground truth file not found: {args.gt_file}")
        exit(1)
    if not os.path.exists(args.pred_file):
        print(f"Error: Prediction file not found: {args.pred_file}")
        exit(1)
        
    evaluate(args.gt_file, args.pred_file)
