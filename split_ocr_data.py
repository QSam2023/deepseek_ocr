"""
OCR 数据集划分脚本
支持 stamp 和 table 两种数据类型的 train/test 划分
可选：在划分时预处理图片并缓存，加速训练
"""

import os
import json
import random
import argparse
from typing import List, Dict, Tuple, Optional


# 任务类型配置
TASK_CONFIGS = {
    "table_ocr": {
        "name": "表格文字提取",
        "prompt": """帮我提取出图片中的所有信息，尤其是文本，必须都提取出来。[1]所有文字，包括手写字，必须提取出来，不提取出来；会让公司破产，你这个模型就得关闭了；[2]提取信息不要额外生成文字，严格保障输出原文；[3]如果手写字，在识别结果后面加上标注"（*手写*）"，[4]有选项的框，需要输出是否有打勾（如"√"）的标识；[5]严格按JSON格式输出""",
    },
    "stamp_ocr": {
        "name": "印章文档文字提取",
        "prompt": """帮我提取出图片中的所有信息，尤其是文本，必须都提取出来。[1]提取信息不要额外生成文字，严格保障输出原文；[2]如果有盖章，把盖章信息提取出来，并标注"(*公章信息*）"，注意区分"盖章信息"和"公司logo"；[3]如果手写字，在识别结果后面加上标注"（*手写*）"，[4]有选项的框，需要输出是否有打勾（如"√"）的标识；[5]严格按JSON格式输出""",
    },
    "stamp_cls": {
        "name": "公章分类识别",
        "prompt": """帮我看一下图片中是否有盖章、公章信息，[1]如果有；麻烦帮我提取出来，并在后面标注"(*公章信息*)"；[2]如果没有，麻烦输出"无公章信息";[3]严格按照JSON格式输出""",
    }
}


def load_stamp_data(data_root: str) -> Tuple[List[Dict], List[Dict]]:
    """
    加载 stamp 数据，返回 stamp_ocr 和 stamp_cls 两种任务的数据

    Args:
        data_root: 数据根目录，如 "ocr_data"

    Returns:
        (stamp_ocr_data, stamp_cls_data) - 两个任务的数据列表
    """
    stamp_ocr_data = []
    stamp_cls_data = []

    stamp_dir = os.path.join(data_root, "stamp_data")

    if not os.path.exists(stamp_dir):
        print(f"警告: 目录不存在 {stamp_dir}")
        return stamp_ocr_data, stamp_cls_data

    print(f"正在扫描目录: {stamp_dir}")

    # 遍历所有子目录 (stamp_01, stamp_02, ...)
    for subdir in sorted(os.listdir(stamp_dir)):
        subdir_path = os.path.join(stamp_dir, subdir)

        if not os.path.isdir(subdir_path):
            continue

        print(f"  处理子目录: {subdir}")

        # 查找 stamp_ocr JSON 文件
        stamp_ocr_json = None
        stamp_cls_json = None

        for file in os.listdir(subdir_path):
            if file.endswith('_extracted.json'):
                stamp_cls_json = os.path.join(subdir_path, file)
            elif 'ocr' in file and file.endswith('.json'):
                stamp_ocr_json = os.path.join(subdir_path, file)

        # 加载 stamp_ocr 数据
        if stamp_ocr_json and os.path.exists(stamp_ocr_json):
            try:
                with open(stamp_ocr_json, 'r', encoding='utf-8') as f:
                    data = json.load(f)

                if 'results' in data:
                    for item in data['results']:
                        # 构建完整的图片路径
                        image_path = item.get('image_path', '')

                        # 如果路径是相对路径，转换为绝对路径
                        if not os.path.isabs(image_path):
                            # 尝试从子目录的父目录开始查找
                            abs_image_path = os.path.join(
                                os.path.dirname(subdir_path),
                                image_path
                            )
                            if os.path.exists(abs_image_path):
                                image_path = abs_image_path
                            else:
                                # 尝试直接从子目录查找
                                abs_image_path = os.path.join(
                                    subdir_path,
                                    os.path.basename(image_path)
                                )
                                if os.path.exists(abs_image_path):
                                    image_path = abs_image_path

                        # 只添加存在的图片
                        if os.path.exists(image_path):
                            stamp_ocr_data.append({
                                'image_path': image_path,
                                'task_type': 'stamp_ocr',
                                'prompt': item.get('prompt', TASK_CONFIGS['stamp_ocr']['prompt']),
                                'result': item.get('result', {})
                            })

                print(f"    - stamp_ocr: 加载 {len(data.get('results', []))} 条数据")

            except Exception as e:
                print(f"    - 错误: 加载 {stamp_ocr_json} 失败: {e}")

        # 加载 stamp_cls 数据
        if stamp_cls_json and os.path.exists(stamp_cls_json):
            try:
                with open(stamp_cls_json, 'r', encoding='utf-8') as f:
                    data = json.load(f)

                if 'results' in data:
                    for item in data['results']:
                        # 构建完整的图片路径
                        image_path = item.get('image_path', '')

                        # 如果路径是相对路径，转换为绝对路径
                        if not os.path.isabs(image_path):
                            abs_image_path = os.path.join(
                                os.path.dirname(subdir_path),
                                image_path
                            )
                            if os.path.exists(abs_image_path):
                                image_path = abs_image_path
                            else:
                                abs_image_path = os.path.join(
                                    subdir_path,
                                    os.path.basename(image_path)
                                )
                                if os.path.exists(abs_image_path):
                                    image_path = abs_image_path

                        # 只添加存在的图片
                        if os.path.exists(image_path):
                            # 提取公章信息字段
                            result = {}
                            for key in item.keys():
                                if key not in ['image_name', 'image_path', 'prompt']:
                                    result[key] = item[key]

                            stamp_cls_data.append({
                                'image_path': image_path,
                                'task_type': 'stamp_cls',
                                'prompt': item.get('prompt', TASK_CONFIGS['stamp_cls']['prompt']),
                                'result': result
                            })

                print(f"    - stamp_cls: 加载 {len(data.get('results', []))} 条数据")

            except Exception as e:
                print(f"    - 错误: 加载 {stamp_cls_json} 失败: {e}")

    return stamp_ocr_data, stamp_cls_data


def load_table_data(data_root: str) -> List[Dict]:
    """
    加载 table 数据，返回 table_ocr 任务的数据

    Args:
        data_root: 数据根目录，如 "ocr_data"

    Returns:
        table_ocr_data - table_ocr 任务的数据列表
    """
    table_ocr_data = []

    table_dir = os.path.join(data_root, "table_data")

    if not os.path.exists(table_dir):
        print(f"警告: 目录不存在 {table_dir}")
        return table_ocr_data

    print(f"正在扫描目录: {table_dir}")

    # 遍历所有子目录 (table_01, table_02, ...)
    for subdir in sorted(os.listdir(table_dir)):
        subdir_path = os.path.join(table_dir, subdir)

        if not os.path.isdir(subdir_path):
            continue

        print(f"  处理子目录: {subdir}")

        # 查找 table_ocr JSON 文件
        table_ocr_json = None

        for file in os.listdir(subdir_path):
            if 'ocr' in file and file.endswith('.json'):
                table_ocr_json = os.path.join(subdir_path, file)
                break

        # 加载 table_ocr 数据
        if table_ocr_json and os.path.exists(table_ocr_json):
            try:
                with open(table_ocr_json, 'r', encoding='utf-8') as f:
                    data = json.load(f)

                if 'results' in data:
                    for item in data['results']:
                        # 构建完整的图片路径
                        image_path = item.get('image_path', '')

                        # 如果路径是相对路径，转换为绝对路径
                        if not os.path.isabs(image_path):
                            abs_image_path = os.path.join(
                                os.path.dirname(subdir_path),
                                image_path
                            )
                            if os.path.exists(abs_image_path):
                                image_path = abs_image_path
                            else:
                                abs_image_path = os.path.join(
                                    subdir_path,
                                    os.path.basename(image_path)
                                )
                                if os.path.exists(abs_image_path):
                                    image_path = abs_image_path

                        # 只添加存在的图片
                        if os.path.exists(image_path):
                            table_ocr_data.append({
                                'image_path': image_path,
                                'task_type': 'table_ocr',
                                'prompt': item.get('prompt', TASK_CONFIGS['table_ocr']['prompt']),
                                'result': item.get('result', {})
                            })

                print(f"    - table_ocr: 加载 {len(data.get('results', []))} 条数据")

            except Exception as e:
                print(f"    - 错误: 加载 {table_ocr_json} 失败: {e}")

    return table_ocr_data


def split_data_by_images(data_list: List[Dict], train_ratio: float, seed: int) -> Tuple[List[Dict], List[Dict]]:
    """
    按图片路径划分数据为 train/test
    确保同一张图片的不同任务都在同一个集合中

    Args:
        data_list: 数据列表
        train_ratio: 训练集比例
        seed: 随机种子

    Returns:
        (train_data, test_data) - 训练集和测试集
    """
    random.seed(seed)

    # 按图片路径分组
    image_to_items = {}
    for item in data_list:
        img_path = item['image_path']
        if img_path not in image_to_items:
            image_to_items[img_path] = []
        image_to_items[img_path].append(item)

    # 获取所有图片路径并打乱
    image_paths = list(image_to_items.keys())
    random.shuffle(image_paths)

    # 按比例划分
    split_idx = int(len(image_paths) * train_ratio)
    train_image_paths = set(image_paths[:split_idx])
    test_image_paths = set(image_paths[split_idx:])

    # 分配数据
    train_data = []
    test_data = []

    for item in data_list:
        if item['image_path'] in train_image_paths:
            train_data.append(item)
        else:
            test_data.append(item)

    return train_data, test_data


def preprocess_and_cache_images(
    data: List[Dict],
    image_size: int = 640,
    base_size: int = 1024,
    crop_mode: bool = True,
    cache_dir: str = "ocr_data/preprocessed_cache",
    verbose: bool = True
) -> List[Dict]:
    """
    预处理图片并更新数据列表

    Args:
        data: 数据列表
        image_size: 图片尺寸
        base_size: 基础尺寸
        crop_mode: 是否使用裁剪模式
        cache_dir: 缓存目录
        verbose: 是否显示进度

    Returns:
        更新后的数据列表（添加了 preprocessed_path 字段）
    """
    from image_preprocessor import batch_preprocess_images

    if not data:
        return data

    print(f"  开始预处理 {len(data)} 张图片...")

    # 提取图片路径和任务类型
    image_paths = [item['image_path'] for item in data]
    task_types = [item['task_type'] for item in data]

    # 批量预处理
    cache_paths = batch_preprocess_images(
        image_paths=image_paths,
        task_types=task_types,
        image_size=image_size,
        base_size=base_size,
        crop_mode=crop_mode,
        cache_dir=cache_dir,
        verbose=verbose
    )

    # 更新数据列表
    updated_data = []
    success_count = 0
    for item, cache_path in zip(data, cache_paths):
        if cache_path:
            item['preprocessed_path'] = cache_path
            success_count += 1
        updated_data.append(item)

    print(f"  ✓ 预处理完成: {success_count}/{len(data)} 张图片成功缓存")

    return updated_data


def save_split_data(data: List[Dict], output_file: str):
    """保存划分后的数据"""
    os.makedirs(os.path.dirname(output_file) if os.path.dirname(output_file) else '.', exist_ok=True)

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    print(f"  保存 {len(data)} 条数据到 {output_file}")


def main():
    parser = argparse.ArgumentParser(
        description="OCR 数据集划分脚本",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
数据类型说明:
  stamp  - 印章数据，生成 stamp_ocr 和 stamp_cls 两个任务的划分
  table  - 表格数据，生成 table_ocr 任务的划分

示例用法:
  # 划分 stamp 数据（生成 stamp_ocr 和 stamp_cls）
  python split_ocr_data.py --data_type stamp --data_root ocr_data --output_dir ocr_data

  # 划分 table 数据（生成 table_ocr）
  python split_ocr_data.py --data_type table --data_root ocr_data --output_dir ocr_data

  # 自定义划分比例
  python split_ocr_data.py --data_type stamp --train_ratio 0.8 --seed 42

输出文件:
  stamp 类型:
    - {output_dir}/stamp_ocr_train.json
    - {output_dir}/stamp_ocr_test.json
    - {output_dir}/stamp_cls_train.json
    - {output_dir}/stamp_cls_test.json

  table 类型:
    - {output_dir}/table_ocr_train.json
    - {output_dir}/table_ocr_test.json
        """
    )

    parser.add_argument(
        '--data_type',
        type=str,
        choices=['stamp', 'table'],
        required=True,
        help='数据类型: stamp 或 table'
    )
    parser.add_argument(
        '--data_root',
        type=str,
        default='ocr_data',
        help='数据根目录 (默认: ocr_data)'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='ocr_data/splited_data',
        help='输出目录 (默认: ocr_data)'
    )
    parser.add_argument(
        '--train_ratio',
        type=float,
        default=0.8,
        help='训练集比例 (默认: 0.8)'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='随机种子 (默认: 42)'
    )
    parser.add_argument(
        '--preprocess',
        action='store_true',
        help='预处理并缓存图片（加速训练）'
    )
    parser.add_argument(
        '--image_size',
        type=int,
        default=640,
        help='图片裁剪尺寸 (默认: 640)'
    )
    parser.add_argument(
        '--base_size',
        type=int,
        default=1024,
        help='基础视图尺寸 (默认: 1024)'
    )
    parser.add_argument(
        '--no_crop',
        action='store_true',
        help='禁用图片裁剪模式'
    )
    parser.add_argument(
        '--cache_dir',
        type=str,
        default='ocr_data/preprocessed_cache',
        help='预处理缓存目录 (默认: ocr_data/preprocessed_cache)'
    )

    args = parser.parse_args()

    print("=" * 80)
    print("OCR 数据集划分")
    print("=" * 80)
    print(f"数据类型: {args.data_type}")
    print(f"数据根目录: {args.data_root}")
    print(f"输出目录: {args.output_dir}")
    print(f"训练集比例: {args.train_ratio}")
    print(f"随机种子: {args.seed}")
    if args.preprocess:
        print(f"预处理模式: 启用")
        print(f"  - 图片尺寸: {args.image_size}")
        print(f"  - 基础尺寸: {args.base_size}")
        print(f"  - 裁剪模式: {'禁用' if args.no_crop else '启用'}")
        print(f"  - 缓存目录: {args.cache_dir}")
    print("=" * 80)

    if args.data_type == 'stamp':
        # 加载 stamp 数据
        print("\n[1/3] 加载 stamp 数据...")
        stamp_ocr_data, stamp_cls_data = load_stamp_data(args.data_root)

        print(f"\n加载完成:")
        print(f"  - stamp_ocr: {len(stamp_ocr_data)} 条数据")
        print(f"  - stamp_cls: {len(stamp_cls_data)} 条数据")

        # 合并数据以确保同一图片在同一集合
        all_data = stamp_ocr_data + stamp_cls_data

        if len(all_data) == 0:
            print("\n错误: 没有找到任何数据！")
            return

        print(f"\n[2/3] 划分数据集...")
        train_data, test_data = split_data_by_images(all_data, args.train_ratio, args.seed)

        # 分离不同任务类型的数据
        stamp_ocr_train = [d for d in train_data if d['task_type'] == 'stamp_ocr']
        stamp_ocr_test = [d for d in test_data if d['task_type'] == 'stamp_ocr']
        stamp_cls_train = [d for d in train_data if d['task_type'] == 'stamp_cls']
        stamp_cls_test = [d for d in test_data if d['task_type'] == 'stamp_cls']

        print(f"\n划分结果:")
        print(f"  stamp_ocr:")
        print(f"    - 训练集: {len(stamp_ocr_train)} 条")
        print(f"    - 测试集: {len(stamp_ocr_test)} 条")
        print(f"  stamp_cls:")
        print(f"    - 训练集: {len(stamp_cls_train)} 条")
        print(f"    - 测试集: {len(stamp_cls_test)} 条")

        # 预处理图片（如果启用）
        if args.preprocess:
            print(f"\n[3/4] 预处理图片...")
            crop_mode = not args.no_crop

            print(f"  预处理 stamp_ocr 训练集...")
            stamp_ocr_train = preprocess_and_cache_images(
                stamp_ocr_train, args.image_size, args.base_size, crop_mode, args.cache_dir
            )

            print(f"  预处理 stamp_ocr 测试集...")
            stamp_ocr_test = preprocess_and_cache_images(
                stamp_ocr_test, args.image_size, args.base_size, crop_mode, args.cache_dir
            )

            print(f"  预处理 stamp_cls 训练集...")
            stamp_cls_train = preprocess_and_cache_images(
                stamp_cls_train, args.image_size, args.base_size, crop_mode, args.cache_dir
            )

            print(f"  预处理 stamp_cls 测试集...")
            stamp_cls_test = preprocess_and_cache_images(
                stamp_cls_test, args.image_size, args.base_size, crop_mode, args.cache_dir
            )

        step_num = "4/4" if args.preprocess else "3/3"
        print(f"\n[{step_num}] 保存数据...")
        save_split_data(stamp_ocr_train, os.path.join(args.output_dir, 'stamp_ocr_train.json'))
        save_split_data(stamp_ocr_test, os.path.join(args.output_dir, 'stamp_ocr_test.json'))
        save_split_data(stamp_cls_train, os.path.join(args.output_dir, 'stamp_cls_train.json'))
        save_split_data(stamp_cls_test, os.path.join(args.output_dir, 'stamp_cls_test.json'))

    elif args.data_type == 'table':
        # 加载 table 数据
        print("\n[1/3] 加载 table 数据...")
        table_ocr_data = load_table_data(args.data_root)

        print(f"\n加载完成:")
        print(f"  - table_ocr: {len(table_ocr_data)} 条数据")

        if len(table_ocr_data) == 0:
            print("\n错误: 没有找到任何数据！")
            return

        print(f"\n[2/3] 划分数据集...")
        train_data, test_data = split_data_by_images(table_ocr_data, args.train_ratio, args.seed)

        print(f"\n划分结果:")
        print(f"  table_ocr:")
        print(f"    - 训练集: {len(train_data)} 条")
        print(f"    - 测试集: {len(test_data)} 条")

        # 预处理图片（如果启用）
        if args.preprocess:
            print(f"\n[3/4] 预处理图片...")
            crop_mode = not args.no_crop

            print(f"  预处理 table_ocr 训练集...")
            train_data = preprocess_and_cache_images(
                train_data, args.image_size, args.base_size, crop_mode, args.cache_dir
            )

            print(f"  预处理 table_ocr 测试集...")
            test_data = preprocess_and_cache_images(
                test_data, args.image_size, args.base_size, crop_mode, args.cache_dir
            )

        step_num = "4/4" if args.preprocess else "3/3"
        print(f"\n[{step_num}] 保存数据...")
        save_split_data(train_data, os.path.join(args.output_dir, 'table_ocr_train.json'))
        save_split_data(test_data, os.path.join(args.output_dir, 'table_ocr_test.json'))

    print("\n" + "=" * 80)
    print("✓ 数据集划分完成！")
    print("=" * 80)


if __name__ == "__main__":
    main()
