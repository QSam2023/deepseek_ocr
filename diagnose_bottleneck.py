#!/usr/bin/env python3
"""
性能瓶颈诊断工具
分析训练过程中的各个环节，找出真正的性能瓶颈
"""

import os
import sys
import time
import json
import yaml
import torch
import argparse
from pathlib import Path
from typing import Dict, List, Any
from PIL import Image
import numpy as np


class Color:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'


def print_section(title: str):
    """打印章节标题"""
    print(f"\n{Color.HEADER}{'=' * 80}")
    print(f"{title}")
    print(f"{'=' * 80}{Color.ENDC}\n")


def benchmark_data_loading(config_path: str, num_samples: int = 100):
    """基准测试数据加载速度"""
    print_section("1️⃣  数据加载性能测试")

    # 加载配置
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    # 加载训练数据
    split_data_dir = config['data']['split_data_dir']
    data_type = config['data']['data_type']

    train_files = []
    if data_type == 'all':
        train_files = ['table_ocr_train.json', 'stamp_ocr_train.json', 'stamp_cls_train.json']
    elif data_type == 'table':
        train_files = ['table_ocr_train.json']
    elif data_type == 'stamp':
        train_files = ['stamp_ocr_train.json', 'stamp_cls_train.json']

    all_train_data = []
    for train_file in train_files:
        file_path = os.path.join(split_data_dir, train_file)
        if os.path.exists(file_path):
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            all_train_data.extend(data)

    if not all_train_data:
        print(f"{Color.FAIL}错误: 没有找到训练数据{Color.ENDC}")
        return {}

    print(f"总数据量: {len(all_train_data)} 样本")
    print(f"测试样本数: {min(num_samples, len(all_train_data))} 样本\n")

    # 测试纯数据读取（不包含图像）
    print(f"{Color.OKCYAN}测试 1: JSON 数据读取{Color.ENDC}")
    start_time = time.time()
    for i in range(min(num_samples, len(all_train_data))):
        sample = all_train_data[i]
        _ = sample.get('image_path')
        _ = sample.get('prompt')
        _ = sample.get('result')
    json_time = time.time() - start_time
    print(f"  耗时: {json_time:.3f} 秒")
    print(f"  速度: {min(num_samples, len(all_train_data)) / json_time:.1f} 样本/秒")

    # 测试图像加载
    print(f"\n{Color.OKCYAN}测试 2: 图像文件加载{Color.ENDC}")
    image_load_times = []
    image_sizes = []

    for i in range(min(num_samples, len(all_train_data))):
        sample = all_train_data[i]
        image_path = sample.get('image_path')
        if image_path and os.path.exists(image_path):
            start = time.time()
            img = Image.open(image_path).convert('RGB')
            image_load_times.append(time.time() - start)
            image_sizes.append(img.size)

    if image_load_times:
        avg_load_time = np.mean(image_load_times)
        total_load_time = sum(image_load_times)
        print(f"  平均图像加载时间: {avg_load_time*1000:.2f} ms")
        print(f"  总耗时: {total_load_time:.3f} 秒")
        print(f"  速度: {len(image_load_times) / total_load_time:.1f} 图像/秒")
        print(f"  平均图像尺寸: {int(np.mean([s[0] for s in image_sizes]))}x{int(np.mean([s[1] for s in image_sizes]))}")

    # 测试图像预处理
    print(f"\n{Color.OKCYAN}测试 3: 图像预处理（resize + tensor）{Color.ENDC}")
    from data_collator import DeepSeekOCRDataCollator

    base_size = config['data_processing']['base_size']
    preprocess_times = []

    for i in range(min(20, len(all_train_data))):  # 只测试20个样本
        sample = all_train_data[i]
        image_path = sample.get('image_path')
        if image_path and os.path.exists(image_path):
            img = Image.open(image_path).convert('RGB')

            start = time.time()
            # 模拟预处理
            from PIL import ImageOps
            global_view = ImageOps.pad(img, (base_size, base_size))
            _ = torch.tensor(np.array(global_view)).permute(2, 0, 1).float()
            preprocess_times.append(time.time() - start)

    if preprocess_times:
        avg_preprocess = np.mean(preprocess_times)
        print(f"  平均预处理时间: {avg_preprocess*1000:.2f} ms")
        print(f"  预估速度: {1/avg_preprocess:.1f} 样本/秒")

    return {
        'json_read_speed': min(num_samples, len(all_train_data)) / json_time if json_time > 0 else 0,
        'image_load_time_ms': np.mean(image_load_times) * 1000 if image_load_times else 0,
        'image_load_speed': len(image_load_times) / sum(image_load_times) if image_load_times else 0,
        'preprocess_time_ms': np.mean(preprocess_times) * 1000 if preprocess_times else 0,
        'total_data_pipeline_time_ms': (
            (np.mean(image_load_times) + np.mean(preprocess_times)) * 1000
            if image_load_times and preprocess_times else 0
        )
    }


def benchmark_model_forward(config_path: str):
    """基准测试模型前向传播速度"""
    print_section("2️⃣  模型计算性能测试")

    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    try:
        from unsloth import FastVisionModel
        from transformers import AutoModel

        print("加载模型...")
        os.environ["UNSLOTH_WARN_UNINITIALIZED"] = '0'

        model_path = config['model']['model_path']

        model, tokenizer = FastVisionModel.from_pretrained(
            model_path,
            load_in_4bit=config['model']['load_in_4bit'],
            auto_model=AutoModel,
            trust_remote_code=True,
            unsloth_force_compile=config['model']['unsloth_force_compile'],
            use_gradient_checkpointing=config['model']['use_gradient_checkpointing'],
        )

        # 配置 LoRA
        lora_config = config['model']['lora']
        model = FastVisionModel.get_peft_model(
            model,
            target_modules=lora_config['target_modules'],
            r=int(lora_config['r']),
            lora_alpha=int(lora_config['lora_alpha']),
            lora_dropout=float(lora_config['lora_dropout']),
            bias=str(lora_config['bias']),
            random_state=int(lora_config['random_state']),
            use_rslora=bool(lora_config['use_rslora']),
        )

        FastVisionModel.for_training(model)
        print("✓ 模型加载完成\n")

        # 创建模拟输入
        batch_size = config['training']['per_device_train_batch_size']
        seq_length = 512  # 模拟序列长度

        print(f"{Color.OKCYAN}测试配置:{Color.ENDC}")
        print(f"  Batch Size: {batch_size}")
        print(f"  Sequence Length: {seq_length}")
        print(f"  LoRA Rank: {lora_config['r']}")

        # 创建模拟输入
        input_ids = torch.randint(0, 32000, (batch_size, seq_length)).cuda()
        attention_mask = torch.ones(batch_size, seq_length).cuda()

        # 预热
        print(f"\n{Color.OKCYAN}预热中...{Color.ENDC}")
        with torch.no_grad():
            for _ in range(3):
                _ = model(input_ids=input_ids, attention_mask=attention_mask)

        if torch.cuda.is_available():
            torch.cuda.synchronize()

        # 测试前向传播
        print(f"\n{Color.OKCYAN}测试: 前向传播{Color.ENDC}")
        forward_times = []
        for _ in range(10):
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            start = time.time()

            with torch.no_grad():
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)

            if torch.cuda.is_available():
                torch.cuda.synchronize()
            forward_times.append(time.time() - start)

        avg_forward = np.mean(forward_times)
        print(f"  平均前向时间: {avg_forward*1000:.2f} ms")
        print(f"  吞吐量: {batch_size / avg_forward:.1f} 样本/秒")

        # 测试前向+反向传播
        print(f"\n{Color.OKCYAN}测试: 前向+反向传播{Color.ENDC}")
        model.train()
        backward_times = []

        for _ in range(10):
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            start = time.time()

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            loss = outputs.logits.mean()  # 模拟损失
            loss.backward()

            if torch.cuda.is_available():
                torch.cuda.synchronize()
            backward_times.append(time.time() - start)

            # 清除梯度
            model.zero_grad()

        avg_backward = np.mean(backward_times)
        print(f"  平均前向+反向时间: {avg_backward*1000:.2f} ms")
        print(f"  吞吐量: {batch_size / avg_backward:.1f} 样本/秒")

        return {
            'forward_time_ms': avg_forward * 1000,
            'backward_time_ms': avg_backward * 1000,
            'throughput_samples_per_sec': batch_size / avg_backward
        }

    except Exception as e:
        print(f"{Color.FAIL}模型测试失败: {e}{Color.ENDC}")
        import traceback
        traceback.print_exc()
        return {}


def analyze_bottleneck(data_stats: Dict, model_stats: Dict):
    """分析瓶颈"""
    print_section("3️⃣  瓶颈分析")

    if not data_stats or not model_stats:
        print(f"{Color.WARNING}数据不完整，无法分析{Color.ENDC}")
        return

    data_time = data_stats.get('total_data_pipeline_time_ms', 0)
    model_time = model_stats.get('backward_time_ms', 0)

    print(f"{Color.BOLD}时间分解:{Color.ENDC}")
    print(f"  数据加载+预处理: {data_time:.2f} ms")
    print(f"  模型前向+反向:   {model_time:.2f} ms")
    print(f"  总时间 (理论):   {data_time + model_time:.2f} ms\n")

    total_time = data_time + model_time
    data_percent = (data_time / total_time * 100) if total_time > 0 else 0
    model_percent = (model_time / total_time * 100) if total_time > 0 else 0

    print(f"{Color.BOLD}时间占比:{Color.ENDC}")
    print(f"  数据处理: {data_percent:.1f}%")
    print(f"  模型计算: {model_percent:.1f}%\n")

    print(f"{Color.BOLD}瓶颈诊断:{Color.ENDC}")

    if data_percent > 40:
        print(f"  {Color.FAIL}🔴 数据加载是主要瓶颈 ({data_percent:.1f}%){Color.ENDC}")
        print(f"\n{Color.OKCYAN}建议优化:{Color.ENDC}")
        print("  1. 增加 dataloader_num_workers")
        print("  2. 启用 dataloader_prefetch_factor")
        print("  3. 使用更快的存储（SSD/NVMe）")
        print("  4. 预处理并缓存图像")
    elif model_percent > 60:
        print(f"  {Color.OKGREEN}🟢 模型计算是主要部分 ({model_percent:.1f}%) - 这是正常的{Color.ENDC}")
        print(f"\n{Color.OKCYAN}进一步优化建议:{Color.ENDC}")
        print("  1. 增加 batch size（如果显存允许）")
        print("  2. 使用混合精度训练（bf16/fp16）")
        print("  3. 启用 torch.compile（PyTorch 2.0+）")
        print("  4. 考虑减小模型大小或序列长度")
    else:
        print(f"  {Color.OKCYAN}🟡 数据和计算较为平衡{Color.ENDC}")
        print(f"\n{Color.OKCYAN}优化建议:{Color.ENDC}")
        print("  1. 同时优化数据加载和模型计算")
        print("  2. 增加 batch size")
        print("  3. 检查是否有其他隐藏瓶颈")


def check_gpu_utilization():
    """检查 GPU 利用率"""
    print_section("4️⃣  GPU 利用率检查")

    try:
        import subprocess
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=name,utilization.gpu,utilization.memory,memory.used,memory.total',
             '--format=csv,noheader,nounits'],
            capture_output=True,
            text=True,
            check=True
        )

        parts = result.stdout.strip().split(', ')
        gpu_name = parts[0]
        gpu_util = float(parts[1])
        mem_util = float(parts[2])
        mem_used = float(parts[3])
        mem_total = float(parts[4])

        print(f"{Color.BOLD}当前 GPU 状态:{Color.ENDC}")
        print(f"  GPU: {gpu_name}")
        print(f"  计算利用率: {gpu_util}%")
        print(f"  显存利用率: {mem_util}%")
        print(f"  显存使用: {mem_used}MB / {mem_total}MB\n")

        print(f"{Color.BOLD}利用率评估:{Color.ENDC}")
        if gpu_util < 30:
            print(f"  {Color.FAIL}🔴 GPU 利用率极低 ({gpu_util}%){Color.ENDC}")
            print("  可能原因: 数据加载瓶颈、batch size 太小")
        elif gpu_util < 50:
            print(f"  {Color.WARNING}🟡 GPU 利用率偏低 ({gpu_util}%){Color.ENDC}")
            print("  有优化空间")
        elif gpu_util < 70:
            print(f"  {Color.OKCYAN}🟢 GPU 利用率中等 ({gpu_util}%){Color.ENDC}")
            print("  较为合理")
        else:
            print(f"  {Color.OKGREEN}🟢 GPU 利用率良好 ({gpu_util}%){Color.ENDC}")
            print("  已充分利用")

        if mem_used / mem_total < 0.3:
            print(f"\n  {Color.WARNING}⚠️  显存利用率低 ({mem_used/mem_total*100:.1f}%){Color.ENDC}")
            print("  建议: 可以增大 batch size 或模型 rank")

    except Exception as e:
        print(f"{Color.WARNING}无法获取 GPU 信息: {e}{Color.ENDC}")


def main():
    parser = argparse.ArgumentParser(
        description="性能瓶颈深度诊断工具",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument('--config', type=str, default='train_config_optimized.yaml',
                        help='配置文件路径')
    parser.add_argument('--num-samples', type=int, default=100,
                        help='测试样本数量')
    parser.add_argument('--skip-model', action='store_true',
                        help='跳过模型测试（加快诊断）')

    args = parser.parse_args()

    print_section("🔍 DeepSeek OCR 性能瓶颈诊断")

    # 检查 GPU
    check_gpu_utilization()

    # 测试数据加载
    data_stats = benchmark_data_loading(args.config, args.num_samples)

    # 测试模型计算
    model_stats = {}
    if not args.skip_model:
        model_stats = benchmark_model_forward(args.config)

    # 分析瓶颈
    if data_stats and model_stats:
        analyze_bottleneck(data_stats, model_stats)

    print_section("✅ 诊断完成")
    print(f"{Color.OKBLUE}建议: 根据上述分析结果调整配置{Color.ENDC}\n")


if __name__ == "__main__":
    main()
