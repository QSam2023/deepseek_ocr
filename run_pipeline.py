"""
OCR Pipeline 主脚本
整合完整流程: 数据划分 -> 批量推理 -> 自动评估
"""

import os
import sys
import argparse
import subprocess
from pathlib import Path


def run_command(cmd: list, description: str):
    """运行命令并处理输出"""
    print(f"\n{'=' * 80}")
    print(f"{description}")
    print('=' * 80)
    print(f"命令: {' '.join(cmd)}")
    print()

    result = subprocess.run(cmd, capture_output=False)

    if result.returncode != 0:
        print(f"✗ 错误: {description} 失败 (返回码: {result.returncode})")
        return False

    print(f"\n✓ {description} 完成")
    return True


def step_1_split_data(data_type: str, data_root: str, output_dir: str):
    """步骤1: 数据划分"""
    print(f"\n{'#' * 80}")
    print("步骤 1/3: 数据划分")
    print('#' * 80)

    if data_type in ['all', 'table']:
        cmd = [
            sys.executable, 'split_ocr_data.py',
            '--data_type', 'table',
            '--data_root', data_root,
            '--output_dir', output_dir
        ]
        if not run_command(cmd, "划分 Table 数据"):
            return False

    if data_type in ['all', 'stamp']:
        cmd = [
            sys.executable, 'split_ocr_data.py',
            '--data_type', 'stamp',
            '--data_root', data_root,
            '--output_dir', output_dir
        ]
        if not run_command(cmd, "划分 Stamp 数据"):
            return False

    return True


def step_2_batch_inference(data_type: str, split_data_dir: str, output_dir: str, resume: bool):
    """步骤2: 批量推理"""
    print(f"\n{'#' * 80}")
    print("步骤 2/3: 批量推理 (调用Cloud API)")
    print('#' * 80)

    cmd = [
        sys.executable, 'batch_inference.py',
        '--data_type', data_type,
        '--split_data_dir', split_data_dir,
        '--output_dir', output_dir
    ]

    if not resume:
        cmd.append('--no-resume')

    if not run_command(cmd, "批量推理"):
        return False

    return True


def step_3_evaluate(data_type: str, split_data_dir: str, result_dir: str):
    """步骤3: 评估"""
    print(f"\n{'#' * 80}")
    print("步骤 3/3: 评估")
    print('#' * 80)

    # 确定要评估的任务
    if data_type == 'all':
        tasks = [
            ('table_ocr', 'table_ocr_eval/eval_table_ocr.py'),
            ('stamp_ocr', 'stamp_ocr_eval/eval_stamp_ocr.py'),
            ('stamp_cls', 'stamp_cls_eval/eval_stamp_cls.py')
        ]
    elif data_type == 'table':
        tasks = [('table_ocr', 'table_ocr_eval/eval_table_ocr.py')]
    else:  # stamp
        tasks = [
            ('stamp_ocr', 'stamp_ocr_eval/eval_stamp_ocr.py'),
            ('stamp_cls', 'stamp_cls_eval/eval_stamp_cls.py')
        ]

    all_success = True

    for task_type, eval_script in tasks:
        # 构建文件路径
        gt_file = os.path.join(split_data_dir, f"{task_type}_test.json")
        pred_file = os.path.join(result_dir, "test", task_type, f"{task_type}_predictions.json")

        # 检查文件是否存在
        if not os.path.exists(gt_file):
            print(f"⚠ 跳过 {task_type}: 测试集文件不存在 ({gt_file})")
            continue

        if not os.path.exists(pred_file):
            print(f"⚠ 跳过 {task_type}: 预测结果文件不存在 ({pred_file})")
            continue

        # 运行评估
        cmd = [sys.executable, eval_script, gt_file, pred_file]
        if not run_command(cmd, f"评估 {task_type}"):
            all_success = False

    return all_success


def main():
    parser = argparse.ArgumentParser(
        description="OCR Pipeline 主脚本 - 整合数据划分、推理、评估",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法:
  # 运行完整pipeline（所有数据类型）
  python run_pipeline.py --data_type all

  # 只处理table数据
  python run_pipeline.py --data_type table

  # 只处理stamp数据
  python run_pipeline.py --data_type stamp

  # 跳过数据划分步骤（使用现有的split结果）
  python run_pipeline.py --data_type all --skip_split

  # 跳过API推理步骤（只运行评估）
  python run_pipeline.py --data_type all --skip_split --skip_inference

  # 从头开始推理（不使用断点续传）
  python run_pipeline.py --data_type all --no-resume

流程说明:
  1. 数据划分: 调用 split_ocr_data.py 划分训练集和测试集
  2. 批量推理: 调用 batch_inference.py 对测试集进行API推理
  3. 自动评估: 调用各个eval脚本评估模型性能

输出结构:
  ocr_data/splited_data/        - 划分后的数据
    ├── table_ocr_train.json
    ├── table_ocr_test.json
    ├── stamp_ocr_train.json
    ├── stamp_ocr_test.json
    ├── stamp_cls_train.json
    └── stamp_cls_test.json

  cloud_result/test/            - 测试集推理结果
    ├── table_ocr/
    │   └── table_ocr_predictions.json
    ├── stamp_ocr/
    │   └── stamp_ocr_predictions.json
    └── stamp_cls/
        └── stamp_cls_predictions.json
        """
    )

    parser.add_argument(
        '--data_type',
        type=str,
        choices=['all', 'table', 'stamp'],
        default='all',
        help='数据类型 (all: 所有任务, table: table_ocr, stamp: stamp_ocr + stamp_cls)'
    )
    parser.add_argument(
        '--data_root',
        type=str,
        default='ocr_data',
        help='原始数据根目录 (默认: ocr_data)'
    )
    parser.add_argument(
        '--split_data_dir',
        type=str,
        default='ocr_data/splited_data',
        help='划分后的数据目录 (默认: ocr_data/splited_data)'
    )
    parser.add_argument(
        '--result_dir',
        type=str,
        default='cloud_result',
        help='推理结果根目录 (默认: cloud_result)'
    )
    parser.add_argument(
        '--skip_split',
        action='store_true',
        help='跳过数据划分步骤（使用现有的split结果）'
    )
    parser.add_argument(
        '--skip_inference',
        action='store_true',
        help='跳过API推理步骤（只运行评估）'
    )
    parser.add_argument(
        '--no-resume',
        action='store_true',
        help='不使用断点续传，从头开始推理'
    )

    args = parser.parse_args()

    print(f"\n{'#' * 80}")
    print("OCR Pipeline - 完整流程")
    print('#' * 80)
    print(f"数据类型: {args.data_type}")
    print(f"数据根目录: {args.data_root}")
    print(f"划分数据目录: {args.split_data_dir}")
    print(f"结果目录: {args.result_dir}")
    print(f"跳过数据划分: {args.skip_split}")
    print(f"跳过推理: {args.skip_inference}")
    print(f"断点续传: {not args.no_resume}")
    print('#' * 80)

    # 步骤1: 数据划分
    if not args.skip_split:
        if not step_1_split_data(args.data_type, args.data_root, args.split_data_dir):
            print("\n✗ Pipeline 失败: 数据划分步骤出错")
            sys.exit(1)
    else:
        print("\n跳过步骤 1/3: 数据划分")

    # 步骤2: 批量推理
    if not args.skip_inference:
        # 检查环境变量
        if not os.environ.get("GOOGLE_AI_STUDIO_KEY"):
            print("\n✗ 错误: 未设置环境变量 GOOGLE_AI_STUDIO_KEY")
            print("请先设置: export GOOGLE_AI_STUDIO_KEY='your_api_key'")
            sys.exit(1)

        resume = not args.no_resume
        if not step_2_batch_inference(args.data_type, args.split_data_dir, args.result_dir, resume):
            print("\n✗ Pipeline 失败: 批量推理步骤出错")
            sys.exit(1)
    else:
        print("\n跳过步骤 2/3: 批量推理")

    # 步骤3: 评估
    if not step_3_evaluate(args.data_type, args.split_data_dir, args.result_dir):
        print("\n✗ Pipeline 失败: 评估步骤出错")
        sys.exit(1)

    # 完成
    print(f"\n{'#' * 80}")
    print("✓ OCR Pipeline 全部完成！")
    print('#' * 80)
    print(f"\n结果位置:")
    print(f"  - 划分数据: {args.split_data_dir}")
    print(f"  - 推理结果: {os.path.join(args.result_dir, 'test')}")
    print(f"\n{'#' * 80}\n")


if __name__ == "__main__":
    main()
