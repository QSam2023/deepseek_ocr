"""
Common utility functions for OCR evaluation scripts.
"""
import json
from typing import Dict, Any, Tuple


def load_json(file_path: str) -> Any:
    """Load JSON content from a file."""
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def normalize_value(value: Any) -> str:
    """Normalize value for comparison (convert to string and strip)."""
    if isinstance(value, str):
        return value.strip()
    return str(value).strip()


def flatten_json(y: Any) -> Dict[str, Any]:
    """
    Flatten a nested JSON object into a dictionary with dot-separated keys.
    Lists are handled by using index as part of the key.

    Example:
        {"a": {"b": "c"}} -> {"a.b": "c"}
        {"a": [{"b": "c"}]} -> {"a.0.b": "c"}
    """
    out = {}

    def flatten(x: Any, name: str = ''):
        if type(x) is dict:
            for a in x:
                flatten(x[a], name + a + '.')
        elif type(x) is list:
            for i, a in enumerate(x):
                flatten(a, name + str(i) + '.')
        else:
            out[name[:-1]] = x

    flatten(y)
    return out


def compare_json(gt: Any, pred: Any) -> Tuple[int, int, int]:
    """
    Compare two JSON objects.

    Args:
        gt: Ground truth JSON object
        pred: Prediction JSON object

    Returns:
        total_gt_fields: Total number of fields (leaf nodes) in ground truth.
        matched_fields: Number of fields that match exactly in prediction.
        total_pred_fields: Total number of fields in prediction.
    """
    gt_flat = flatten_json(gt)
    pred_flat = flatten_json(pred)

    total_gt = len(gt_flat)
    total_pred = len(pred_flat)
    matches = 0

    for key, value in gt_flat.items():
        if key in pred_flat:
            if normalize_value(value) == normalize_value(pred_flat[key]):
                matches += 1

    return total_gt, matches, total_pred


def build_image_map(data_list: list, key_field: str = 'image_name') -> Dict[str, Dict]:
    """
    Build a dictionary keyed by image_name for easy lookup.
    Automatically handles both 'image_name' and 'image_path' fields.

    Args:
        data_list: List of data items
        key_field: Field to use as key (default: 'image_name')

    Returns:
        Dictionary mapping image_name to data item
    """
    import os
    image_map = {}

    for item in data_list:
        # 尝试获取 image_name 或从 image_path 提取
        if key_field in item:
            key = item[key_field]
        elif 'image_path' in item:
            # 从完整路径中提取文件名
            key = os.path.basename(item['image_path'])
        elif 'image_name' in item:
            key = item['image_name']
        else:
            continue  # 跳过没有图像标识的条目

        # 统一使用文件名作为key（去除路径部分）
        key = os.path.basename(key)
        image_map[key] = item

    return image_map


def parse_data_structure(data: Any) -> list:
    """
    Parse data structure to handle different formats.
    Expecting: {"results": [...]} or just [...]

    Args:
        data: Input data (dict or list)

    Returns:
        List of data items
    """
    if isinstance(data, dict):
        return data.get("results", [])
    return data
