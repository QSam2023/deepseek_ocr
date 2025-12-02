import json
import argparse
import os
import sys
from typing import Dict, Any, List

# 添加项目根目录到path，以便导入eval_utils
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from eval_utils import load_json, normalize_value, build_image_map, parse_data_structure

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

    total_images = 0
    correct_classifications = 0
    correct_extractions = 0
    total_positive_samples = 0

    print("-" * 80)
    print(f"{'Image Name':<30} | {'Status':<15} | {'GT Type':<15} | {'Pred Type':<15}")
    print("-" * 80)

    for image_name, gt_item in gt_map.items():
        if image_name not in pred_map:
            print(f"{image_name:<30} | {'MISSING':<15} | {'-':<15} | {'-':<15}")
            continue
        
        total_images += 1
        pred_item = pred_map[image_name]
        
        # Check "公章信息" field
        gt_info = gt_item.get("公章信息", "无公章信息")
        pred_info = pred_item.get("公章信息", "无公章信息")
        
        # Determine type (Has Stamp vs No Stamp)
        gt_has_stamp = gt_info != "无公章信息"
        pred_has_stamp = pred_info != "无公章信息"
        
        gt_type = "Has Stamp" if gt_has_stamp else "No Stamp"
        pred_type = "Has Stamp" if pred_has_stamp else "No Stamp"
        
        status = "CORRECT"
        
        if gt_has_stamp == pred_has_stamp:
            correct_classifications += 1
            if gt_has_stamp:
                total_positive_samples += 1
                # For positive samples, check content match
                if normalize_value(gt_info) == normalize_value(pred_info):
                    correct_extractions += 1
                else:
                    status = "CONTENT MISMATCH"
        else:
            status = "TYPE MISMATCH"
            if gt_has_stamp:
                total_positive_samples += 1

        print(f"{image_name:<30} | {status:<15} | {gt_type:<15} | {pred_type:<15}")

    print("-" * 80)
    
    cls_accuracy = (correct_classifications / total_images * 100) if total_images > 0 else 0.0
    extraction_accuracy = (correct_extractions / total_positive_samples * 100) if total_positive_samples > 0 else 0.0

    print(f"Total Images: {total_images}")
    print(f"Classification Accuracy: {cls_accuracy:.2f}%")
    print(f"Extraction Accuracy (Positive Samples): {extraction_accuracy:.2f}% ({correct_extractions}/{total_positive_samples})")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate Stamp Classification/Extraction results.")
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
