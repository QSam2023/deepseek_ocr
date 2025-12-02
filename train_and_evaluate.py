#!/usr/bin/env python3
"""
完整的训练和评估流程脚本
1. 检查基础模型
2. 划分数据集
3. 训练前评估（使用基础模型）
4. 训练模型
5. 训练后评估（使用 LoRA 模型）
6. 对比结果
"""

import os
import sys
import json
import yaml
import time
import argparse
import subprocess
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional


class Color:
    """终端颜色"""
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


def print_section(title: str, color: str = Color.HEADER):
    """打印章节标题"""
    print(f"\n{color}{'=' * 80}")
    print(f"{title}")
    print(f"{'=' * 80}{Color.ENDC}\n")


def print_step(step: str, total: int, current: int, description: str):
    """打印步骤信息"""
    print(f"\n{Color.BOLD}{Color.OKCYAN}[步骤 {current}/{total}] {step}{Color.ENDC}")
    print(f"{Color.OKBLUE}{description}{Color.ENDC}")
    print("-" * 80)


def run_command(cmd: List[str], description: str, check: bool = True) -> bool:
    """
    运行命令并处理输出

    Args:
        cmd: 命令列表
        description: 命令描述
        check: 是否检查返回码

    Returns:
        是否成功
    """
    print(f"\n{Color.OKBLUE}运行: {' '.join(cmd)}{Color.ENDC}\n")

    result = subprocess.run(cmd, capture_output=False)

    if check and result.returncode != 0:
        print(f"\n{Color.FAIL}✗ 错误: {description} 失败 (返回码: {result.returncode}){Color.ENDC}")
        return False

    print(f"\n{Color.OKGREEN}✓ {description} 完成{Color.ENDC}")
    return True


def step_1_check_model(model_dir: str, model_id: str, auto_download: bool) -> bool:
    """步骤 1: 检查基础模型"""
    print_step("检查基础模型", 6, 1, f"检查模型目录: {model_dir}")

    cmd = [sys.executable, 'check_origin_model.py', '--model_dir', model_dir, '--model_id', model_id]

    if auto_download:
        cmd.append('--auto-download')

    return run_command(cmd, "模型检查")


def step_2_split_data(data_type: str, data_root: str, output_dir: str,
                      train_ratio: float, seed: int) -> bool:
    """步骤 2: 划分数据集"""
    print_step("划分数据集", 6, 2, f"数据类型: {data_type}, 训练比例: {train_ratio}")

    # 确定要划分的数据类型
    types_to_split = []
    if data_type == 'all':
        types_to_split = ['table', 'stamp']
    elif data_type == 'table':
        types_to_split = ['table']
    elif data_type == 'stamp':
        types_to_split = ['stamp']

    # 划分数据
    for dtype in types_to_split:
        print(f"\n{Color.OKBLUE}划分 {dtype} 数据...{Color.ENDC}")
        cmd = [
            sys.executable, 'split_ocr_data.py',
            '--data_type', dtype,
            '--data_root', data_root,
            '--output_dir', output_dir,
            '--train_ratio', str(train_ratio),
            '--seed', str(seed)
        ]

        if not run_command(cmd, f"划分 {dtype} 数据"):
            return False

    return True


def step_3_baseline_inference(data_type: str, model_path: str, split_data_dir: str,
                               output_dir: str) -> bool:
    """步骤 3: 训练前评估（基线）"""
    print_step("训练前评估", 6, 3, f"使用基础模型: {model_path}")

    cmd = [
        sys.executable, 'batch_inference.py',
        '--data_type', data_type,
        '--inference_mode', 'local',
        '--model_path', model_path,
        '--split_data_dir', split_data_dir,
        '--output_dir', output_dir,
        '--no-resume'  # 从头开始
    ]

    return run_command(cmd, "训练前推理")


def step_4_train_model(config_path: str, overrides: Dict[str, Any]) -> bool:
    """步骤 4: 训练模型"""
    print_step("训练模型", 6, 4, f"配置文件: {config_path}")

    cmd = [sys.executable, 'train_model.py', '--config', config_path]

    # 添加覆盖参数
    if 'data_type' in overrides:
        cmd.extend(['--data_type', overrides['data_type']])
    if 'max_steps' in overrides:
        cmd.extend(['--max_steps', str(overrides['max_steps'])])
    if 'num_train_epochs' in overrides:
        cmd.extend(['--num_train_epochs', str(overrides['num_train_epochs'])])
    if 'learning_rate' in overrides:
        cmd.extend(['--learning_rate', str(overrides['learning_rate'])])
    if 'output_dir' in overrides:
        cmd.extend(['--output_dir', overrides['output_dir']])

    return run_command(cmd, "模型训练")


def step_5_lora_inference(data_type: str, lora_path: str, base_model_path: str,
                          split_data_dir: str, output_dir: str) -> bool:
    """步骤 5: 训练后评估（LoRA）"""
    print_step("训练后评估", 6, 5, f"使用 LoRA 模型: {lora_path}")

    cmd = [
        sys.executable, 'batch_inference.py',
        '--data_type', data_type,
        '--inference_mode', 'local',
        '--model_path', lora_path,
        '--base_model_path', base_model_path,
        '--split_data_dir', split_data_dir,
        '--output_dir', output_dir,
        '--no-resume'  # 从头开始
    ]

    return run_command(cmd, "训练后推理")


def step_6_evaluate_and_compare(data_type: str, split_data_dir: str,
                                 baseline_dir: str, lora_dir: str) -> Dict[str, Any]:
    """步骤 6: 评估并对比结果"""
    print_step("评估并对比结果", 6, 6, "运行评估脚本并对比性能")

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

    results = {}

    for task_type, eval_script in tasks:
        print(f"\n{Color.OKBLUE}{'=' * 80}")
        print(f"评估任务: {task_type}")
        print(f"{'=' * 80}{Color.ENDC}\n")

        gt_file = os.path.join(split_data_dir, f"{task_type}_test.json")

        if not os.path.exists(gt_file):
            print(f"{Color.WARNING}⚠ 跳过 {task_type}: 测试集文件不存在{Color.ENDC}")
            continue

        # 评估基线模型
        print(f"\n{Color.OKCYAN}评估基线模型 (训练前):{Color.ENDC}")
        baseline_pred_file = os.path.join(baseline_dir, "test", task_type, f"{task_type}_predictions.json")

        if os.path.exists(baseline_pred_file):
            cmd = [sys.executable, eval_script, gt_file, baseline_pred_file]
            run_command(cmd, f"基线模型评估 ({task_type})", check=False)
        else:
            print(f"{Color.WARNING}⚠ 基线预测文件不存在: {baseline_pred_file}{Color.ENDC}")

        # 评估 LoRA 模型
        print(f"\n{Color.OKCYAN}评估 LoRA 模型 (训练后):{Color.ENDC}")
        lora_pred_file = os.path.join(lora_dir, "test", task_type, f"{task_type}_predictions.json")

        if os.path.exists(lora_pred_file):
            cmd = [sys.executable, eval_script, gt_file, lora_pred_file]
            run_command(cmd, f"LoRA 模型评估 ({task_type})", check=False)
        else:
            print(f"{Color.WARNING}⚠ LoRA 预测文件不存在: {lora_pred_file}{Color.ENDC}")

        results[task_type] = {
            'baseline_pred': baseline_pred_file,
            'lora_pred': lora_pred_file,
            'gt': gt_file
        }

    return results


def save_experiment_summary(config: Dict[str, Any], results: Dict[str, Any],
                            start_time: float, output_file: str):
    """保存实验总结"""
    summary = {
        'experiment_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'duration_seconds': time.time() - start_time,
        'config': config,
        'results': results,
    }

    os.makedirs(os.path.dirname(output_file) if os.path.dirname(output_file) else '.', exist_ok=True)

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"\n{Color.OKGREEN}实验总结已保存: {output_file}{Color.ENDC}")


def main():
    parser = argparse.ArgumentParser(
        description="完整的训练和评估流程",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
工作流程:
  1. 检查基础模型（可自动下载）
  2. 划分数据集
  3. 训练前评估（使用基础模型）
  4. 训练模型
  5. 训练后评估（使用 LoRA 模型）
  6. 对比结果

示例用法:
  # 完整流程（默认参数）
  python train_and_evaluate.py

  # 自定义数据类型和训练步数
  python train_and_evaluate.py --data_type stamp --max_steps 100

  # 使用自定义配置文件
  python train_and_evaluate.py --train_config my_config.yaml

  # 跳过某些步骤
  python train_and_evaluate.py --skip_model_check --skip_data_split

  # 自动下载模型（无需确认）
  python train_and_evaluate.py --auto_download_model

输出目录:
  baseline_result/         - 训练前推理结果
  lora_result/             - 训练后推理结果
  lora_model/              - 训练的 LoRA 模型
  experiment_summary.json  - 实验总结
        """
    )

    # 基础配置
    parser.add_argument('--model_dir', type=str, default='./deepseek_ocr',
                        help='基础模型目录 (默认: ./deepseek_ocr)')
    parser.add_argument('--model_id', type=str, default='unsloth/DeepSeek-OCR',
                        help='Hugging Face 模型ID (默认: unsloth/DeepSeek-OCR)')
    parser.add_argument('--auto_download_model', action='store_true',
                        help='自动下载模型，无需确认')

    # 数据配置
    parser.add_argument('--data_type', type=str, choices=['all', 'table', 'stamp'], default='all',
                        help='数据类型 (默认: all)')
    parser.add_argument('--data_root', type=str, default='ocr_data',
                        help='数据根目录 (默认: ocr_data)')
    parser.add_argument('--split_data_dir', type=str, default='ocr_data/splited_data',
                        help='划分后的数据目录 (默认: ocr_data/splited_data)')
    parser.add_argument('--train_ratio', type=float, default=0.8,
                        help='训练集比例 (默认: 0.8)')
    parser.add_argument('--split_seed', type=int, default=42,
                        help='数据划分随机种子 (默认: 42)')

    # 训练配置
    parser.add_argument('--train_config', type=str, default='train_config.yaml',
                        help='训练配置文件 (默认: train_config.yaml)')
    parser.add_argument('--max_steps', type=int, default=None,
                        help='最大训练步数 (覆盖配置文件)')
    parser.add_argument('--num_train_epochs', type=int, default=None,
                        help='训练轮数 (覆盖配置文件)')
    parser.add_argument('--learning_rate', type=float, default=None,
                        help='学习率 (覆盖配置文件)')
    parser.add_argument('--train_output_dir', type=str, default='outputs',
                        help='训练输出目录 (默认: outputs)')

    # 推理配置
    parser.add_argument('--baseline_output_dir', type=str, default='baseline_result',
                        help='基线推理输出目录 (默认: baseline_result)')
    parser.add_argument('--lora_output_dir', type=str, default='lora_result',
                        help='LoRA 推理输出目录 (默认: lora_result)')
    parser.add_argument('--lora_model_path', type=str, default='lora_model',
                        help='LoRA 模型保存路径 (默认: lora_model)')

    # 流程控制
    parser.add_argument('--skip_model_check', action='store_true',
                        help='跳过模型检查步骤')
    parser.add_argument('--skip_data_split', action='store_true',
                        help='跳过数据划分步骤')
    parser.add_argument('--skip_baseline_inference', action='store_true',
                        help='跳过训练前评估步骤')
    parser.add_argument('--skip_training', action='store_true',
                        help='跳过训练步骤')
    parser.add_argument('--skip_lora_inference', action='store_true',
                        help='跳过训练后评估步骤')

    # 输出配置
    parser.add_argument('--summary_file', type=str, default='experiment_summary.json',
                        help='实验总结文件 (默认: experiment_summary.json)')

    args = parser.parse_args()

    # 记录开始时间
    start_time = time.time()

    # 打印流程标题
    print_section("🚀 DeepSeek OCR 完整训练和评估流程", Color.HEADER)
    print(f"{Color.OKBLUE}开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}{Color.ENDC}")
    print(f"{Color.OKBLUE}数据类型: {args.data_type}{Color.ENDC}")
    print(f"{Color.OKBLUE}基础模型: {args.model_dir}{Color.ENDC}")
    print(f"{Color.OKBLUE}LoRA 模型: {args.lora_model_path}{Color.ENDC}")

    try:
        # 步骤 1: 检查模型
        if not args.skip_model_check:
            if not step_1_check_model(args.model_dir, args.model_id, args.auto_download_model):
                raise RuntimeError("模型检查失败")
        else:
            print(f"\n{Color.WARNING}⚠ 跳过步骤 1: 模型检查{Color.ENDC}")

        # 步骤 2: 划分数据
        if not args.skip_data_split:
            if not step_2_split_data(args.data_type, args.data_root, args.split_data_dir,
                                     args.train_ratio, args.split_seed):
                raise RuntimeError("数据划分失败")
        else:
            print(f"\n{Color.WARNING}⚠ 跳过步骤 2: 数据划分{Color.ENDC}")

        # 步骤 3: 训练前评估
        if not args.skip_baseline_inference:
            if not step_3_baseline_inference(args.data_type, args.model_dir, args.split_data_dir,
                                             args.baseline_output_dir):
                raise RuntimeError("训练前评估失败")
        else:
            print(f"\n{Color.WARNING}⚠ 跳过步骤 3: 训练前评估{Color.ENDC}")

        # 步骤 4: 训练模型
        if not args.skip_training:
            overrides = {}
            if args.data_type:
                overrides['data_type'] = args.data_type
            if args.max_steps:
                overrides['max_steps'] = args.max_steps
            if args.num_train_epochs:
                overrides['num_train_epochs'] = args.num_train_epochs
            if args.learning_rate:
                overrides['learning_rate'] = args.learning_rate
            if args.train_output_dir:
                overrides['output_dir'] = args.train_output_dir

            if not step_4_train_model(args.train_config, overrides):
                raise RuntimeError("模型训练失败")
        else:
            print(f"\n{Color.WARNING}⚠ 跳过步骤 4: 模型训练{Color.ENDC}")

        # 步骤 5: 训练后评估
        if not args.skip_lora_inference:
            if not step_5_lora_inference(args.data_type, args.lora_model_path, args.model_dir,
                                         args.split_data_dir, args.lora_output_dir):
                raise RuntimeError("训练后评估失败")
        else:
            print(f"\n{Color.WARNING}⚠ 跳过步骤 5: 训练后评估{Color.ENDC}")

        # 步骤 6: 评估并对比
        results = step_6_evaluate_and_compare(args.data_type, args.split_data_dir,
                                              args.baseline_output_dir, args.lora_output_dir)

        # 保存实验总结
        config_summary = {
            'model_dir': args.model_dir,
            'data_type': args.data_type,
            'train_ratio': args.train_ratio,
            'lora_model_path': args.lora_model_path,
        }
        save_experiment_summary(config_summary, results, start_time, args.summary_file)

        # 完成
        duration = time.time() - start_time
        print_section("✅ 完整流程执行成功！", Color.OKGREEN)
        print(f"{Color.OKGREEN}总耗时: {duration:.2f} 秒 ({duration/60:.2f} 分钟){Color.ENDC}")
        print(f"\n{Color.OKBLUE}结果位置:{Color.ENDC}")
        print(f"  - 训练前结果: {args.baseline_output_dir}/test/")
        print(f"  - 训练后结果: {args.lora_output_dir}/test/")
        print(f"  - LoRA 模型: {args.lora_model_path}/")
        print(f"  - 实验总结: {args.summary_file}")
        print()

    except Exception as e:
        print(f"\n{Color.FAIL}{'=' * 80}")
        print(f"✗ 流程失败: {e}")
        print(f"{'=' * 80}{Color.ENDC}\n")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
