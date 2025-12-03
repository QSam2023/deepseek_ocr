"""
DeepSeek OCR 训练脚本
支持从配置文件加载参数，使用 unsloth 进行高效微调
"""

import os
import sys
import json
import yaml
import torch
import argparse
import subprocess
from pathlib import Path
from typing import List, Dict, Any
from PIL import Image

from transformers import AutoModel, Trainer, TrainingArguments
from unsloth import FastVisionModel, is_bf16_supported
from data_collator import DeepSeekOCRDataCollator


def load_config(config_path: str) -> Dict[str, Any]:
    """加载 YAML 配置文件"""
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config


def split_data_if_needed(config: Dict[str, Any]):
    """根据配置决定是否需要重新划分数据"""
    if config['data']['use_existing_split']:
        print("\n使用现有的数据划分")
        split_dir = config['data']['split_data_dir']
        if not os.path.exists(split_dir):
            raise FileNotFoundError(
                f"划分数据目录不存在: {split_dir}\n"
                f"请先运行数据划分或将 use_existing_split 设置为 false"
            )
        return

    print("\n重新划分数据集...")
    data_type = config['data']['data_type']
    data_root = config['data']['data_root']
    split_data_dir = config['data']['split_data_dir']
    train_ratio = config['data']['train_ratio']
    seed = config['data']['split_seed']

    # 确定要划分的数据类型
    types_to_split = []
    if data_type == 'all':
        types_to_split = ['table', 'stamp']
    elif data_type == 'table':
        types_to_split = ['table']
    elif data_type == 'stamp':
        types_to_split = ['stamp']

    # 调用 split_ocr_data.py
    for dtype in types_to_split:
        cmd = [
            sys.executable, 'split_ocr_data.py',
            '--data_type', dtype,
            '--data_root', data_root,
            '--output_dir', split_data_dir,
            '--train_ratio', str(train_ratio),
            '--seed', str(seed)
        ]

        print(f"\n划分 {dtype} 数据...")
        print(f"命令: {' '.join(cmd)}")
        result = subprocess.run(cmd, capture_output=False)

        if result.returncode != 0:
            raise RuntimeError(f"数据划分失败: {dtype}")

    print("\n数据划分完成")


def load_training_data(config: Dict[str, Any]) -> List[Dict]:
    """加载训练数据"""
    split_data_dir = config['data']['split_data_dir']
    data_type = config['data']['data_type']

    train_files = []
    if data_type == 'all':
        train_files = [
            'table_ocr_train.json',
            'stamp_ocr_train.json',
            'stamp_cls_train.json'
        ]
    elif data_type == 'table':
        train_files = ['table_ocr_train.json']
    elif data_type == 'stamp':
        train_files = ['stamp_ocr_train.json', 'stamp_cls_train.json']

    # 加载所有训练数据
    all_train_data = []
    for train_file in train_files:
        file_path = os.path.join(split_data_dir, train_file)
        if not os.path.exists(file_path):
            print(f"警告: 训练数据文件不存在 {file_path}，跳过")
            continue

        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        print(f"加载 {train_file}: {len(data)} 条数据")
        all_train_data.extend(data)

    if not all_train_data:
        raise ValueError("没有找到任何训练数据！")

    print(f"\n总计加载 {len(all_train_data)} 条训练数据")
    return all_train_data


def convert_to_conversation(sample: Dict) -> Dict:
    """
    将数据集样本转换为对话格式

    Args:
        sample: 包含 image_path, prompt, result 的字典

    Returns:
        包含 messages 的字典
    """
    # 读取图像
    image_path = sample['image_path']
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"图像文件不存在: {image_path}")

    image = Image.open(image_path).convert('RGB')

    # 构建对话
    prompt = sample.get('prompt', '')
    result = sample.get('result', {})

    # 统一将 result 转换为 JSON 格式
    # 如果是 string，包装成 {"result": "..."}
    if isinstance(result, dict):
        result_obj = result
    else:
        result_obj = {"result": str(result)}

    # 生成 JSON 字符串并用 markdown 代码块包裹
    result_json = json.dumps(result_obj, ensure_ascii=False, indent=2)
    result_text = f"```json\n{result_json}\n```"

    conversation = [
        {
            "role": "<|User|>",
            "content": f"<image>\n{prompt}",
            "images": [image]
        },
        {
            "role": "<|Assistant|>",
            "content": result_text
        },
    ]

    return {"messages": conversation}


def setup_model_and_tokenizer(config: Dict[str, Any]):
    """设置模型和 tokenizer"""
    model_config = config['model']

    print("\n加载模型和 tokenizer...")
    print(f"模型路径: {model_config['model_path']}")

    # 设置环境变量
    os.environ["UNSLOTH_WARN_UNINITIALIZED"] = '0'

    # 加载模型
    model, tokenizer = FastVisionModel.from_pretrained(
        model_config['model_path'],
        load_in_4bit=model_config['load_in_4bit'],
        auto_model=AutoModel,
        trust_remote_code=True,
        unsloth_force_compile=model_config['unsloth_force_compile'],
        use_gradient_checkpointing=model_config['use_gradient_checkpointing'],
    )

    print("模型加载完成")

    # 配置 LoRA
    lora_config = model_config['lora']
    print("\n配置 LoRA...")
    print(f"  r: {lora_config['r']}")
    print(f"  lora_alpha: {lora_config['lora_alpha']}")
    print(f"  target_modules: {lora_config['target_modules']}")

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

    # 启用训练模式
    FastVisionModel.for_training(model)
    print("LoRA 配置完成")

    return model, tokenizer


def create_data_collator(tokenizer, model, config: Dict[str, Any]):
    """创建数据整理器"""
    data_proc_config = config['data_processing']

    print("\n创建 DataCollator...")
    print(f"  image_size: {data_proc_config['image_size']}")
    print(f"  base_size: {data_proc_config['base_size']}")
    print(f"  crop_mode: {data_proc_config['crop_mode']}")
    print(f"  train_on_responses_only: {data_proc_config['train_on_responses_only']}")

    data_collator = DeepSeekOCRDataCollator(
        tokenizer=tokenizer,
        model=model,
        image_size=int(data_proc_config['image_size']),
        base_size=int(data_proc_config['base_size']),
        crop_mode=bool(data_proc_config['crop_mode']),
        train_on_responses_only=bool(data_proc_config['train_on_responses_only']),
    )

    return data_collator


def create_training_args(config: Dict[str, Any]) -> TrainingArguments:
    """创建训练参数"""
    train_config = config['training']

    print("\n配置训练参数...")
    print(f"  output_dir: {train_config['output_dir']}")
    print(f"  batch_size: {train_config['per_device_train_batch_size']}")
    print(f"  gradient_accumulation_steps: {train_config['gradient_accumulation_steps']}")
    print(f"  learning_rate: {train_config['learning_rate']}")

    # 计算有效 batch size
    effective_batch_size = (
        train_config['per_device_train_batch_size'] *
        train_config['gradient_accumulation_steps']
    )
    print(f"  有效 batch size: {effective_batch_size}")

    # 根据配置决定使用 max_steps 还是 num_train_epochs
    # 确保所有数值参数都是正确的类型
    training_args_dict = {
        "output_dir": str(train_config['output_dir']),
        "per_device_train_batch_size": int(train_config['per_device_train_batch_size']),
        "gradient_accumulation_steps": int(train_config['gradient_accumulation_steps']),
        "warmup_steps": int(train_config['warmup_steps']),
        "learning_rate": float(train_config['learning_rate']),
        "logging_steps": int(train_config['logging_steps']),
        "optim": str(train_config['optim']),
        "weight_decay": float(train_config['weight_decay']),
        "lr_scheduler_type": str(train_config['lr_scheduler_type']),
        "seed": int(train_config['seed']),
        "dataloader_num_workers": int(train_config['dataloader_num_workers']),
        "save_strategy": str(train_config['save_strategy']),
        "save_steps": int(train_config['save_steps']),
        "save_total_limit": int(train_config['save_total_limit']),
        "report_to": str(train_config['report_to']),
        "remove_unused_columns": False,  # 视觉微调必须设置
    }

    # 🚀 优化：混合精度训练配置
    # 优先使用配置文件中的设置，否则自动检测
    if 'bf16' in train_config:
        training_args_dict['bf16'] = bool(train_config['bf16'])
        print(f"  bf16 (配置文件): {train_config['bf16']}")
    else:
        training_args_dict['bf16'] = is_bf16_supported()
        print(f"  bf16 (自动检测): {is_bf16_supported()}")

    if 'fp16' in train_config:
        training_args_dict['fp16'] = bool(train_config['fp16'])
        print(f"  fp16 (配置文件): {train_config['fp16']}")
    else:
        training_args_dict['fp16'] = not is_bf16_supported()
        print(f"  fp16 (自动检测): {not is_bf16_supported()}")

    # 🚀 优化：TF32 加速（A100 支持）
    if 'tf32' in train_config:
        training_args_dict['tf32'] = bool(train_config['tf32'])
        print(f"  tf32: {train_config['tf32']}")

    # 🚀 优化：DataLoader 优化参数
    if 'dataloader_prefetch_factor' in train_config:
        # prefetch_factor 只在 num_workers > 0 时有效
        if train_config['dataloader_num_workers'] > 0:
            training_args_dict['dataloader_prefetch_factor'] = int(train_config['dataloader_prefetch_factor'])
            print(f"  dataloader_prefetch_factor: {train_config['dataloader_prefetch_factor']}")

    if 'dataloader_pin_memory' in train_config:
        training_args_dict['dataloader_pin_memory'] = bool(train_config['dataloader_pin_memory'])
        print(f"  dataloader_pin_memory: {train_config['dataloader_pin_memory']}")

    if 'dataloader_persistent_workers' in train_config:
        # persistent_workers 只在 num_workers > 0 时有效
        if train_config['dataloader_num_workers'] > 0:
            training_args_dict['dataloader_persistent_workers'] = bool(train_config['dataloader_persistent_workers'])
            print(f"  dataloader_persistent_workers: {train_config['dataloader_persistent_workers']}")

    # 🚀 优化：梯度裁剪
    if 'max_grad_norm' in train_config:
        training_args_dict['max_grad_norm'] = float(train_config['max_grad_norm'])
        print(f"  max_grad_norm: {train_config['max_grad_norm']}")

    # 添加 max_steps 或 num_train_epochs
    if 'num_train_epochs' in train_config and train_config['num_train_epochs'] is not None:
        training_args_dict['num_train_epochs'] = int(train_config['num_train_epochs'])
        print(f"  num_train_epochs: {train_config['num_train_epochs']}")
    else:
        training_args_dict['max_steps'] = int(train_config['max_steps'])
        print(f"  max_steps: {train_config['max_steps']}")

    return TrainingArguments(**training_args_dict)


def print_gpu_stats():
    """打印 GPU 内存使用情况"""
    if torch.cuda.is_available():
        gpu_stats = torch.cuda.get_device_properties(0)
        start_gpu_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
        max_memory = round(gpu_stats.total_memory / 1024 / 1024 / 1024, 3)
        print(f"\nGPU = {gpu_stats.name}. Max memory = {max_memory} GB.")
        print(f"{start_gpu_memory} GB of memory reserved.")
    else:
        print("\n警告: 未检测到 GPU，将使用 CPU 训练（速度会很慢）")


def save_model(model, tokenizer, config: Dict[str, Any]):
    """保存模型"""
    save_config = config['saving']

    # 保存 LoRA 模型
    lora_path = save_config['lora_model_path']
    print(f"\n保存 LoRA 模型到: {lora_path}")
    model.save_pretrained(lora_path)
    tokenizer.save_pretrained(lora_path)
    print("LoRA 模型保存完成")

    # 如果需要，保存合并后的完整模型
    if save_config['save_merged_model']:
        merged_path = save_config['merged_model_path']
        print(f"\n保存合并后的完整模型到: {merged_path}")
        model.save_pretrained_merged(merged_path, tokenizer, save_method="merged_16bit")
        print("完整模型保存完成")


def main():
    parser = argparse.ArgumentParser(
        description="DeepSeek OCR 训练脚本",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法:
  # 使用默认配置文件
  python train_model.py

  # 使用自定义配置文件
  python train_model.py --config my_train_config.yaml

  # 覆盖配置文件中的某些参数
  python train_model.py --data_type table --max_steps 100

配置文件说明:
  配置文件使用 YAML 格式，包含以下部分:
  - data: 数据配置（数据路径、划分参数等）
  - model: 模型配置（模型路径、LoRA 参数等）
  - data_processing: 数据处理配置（图像大小、裁剪模式等）
  - training: 训练配置（batch size、学习率、优化器等）
  - saving: 模型保存配置

  详细配置说明请参考 train_config.yaml
        """
    )

    parser.add_argument(
        '--config',
        type=str,
        default='train_config.yaml',
        help='配置文件路径 (默认: train_config.yaml)'
    )

    # 允许通过命令行覆盖某些关键参数
    parser.add_argument('--data_type', type=str, choices=['all', 'table', 'stamp'],
                        help='覆盖配置文件中的 data_type')
    parser.add_argument('--max_steps', type=int,
                        help='覆盖配置文件中的 max_steps')
    parser.add_argument('--num_train_epochs', type=int,
                        help='覆盖配置文件中的 num_train_epochs')
    parser.add_argument('--learning_rate', type=float,
                        help='覆盖配置文件中的 learning_rate')
    parser.add_argument('--output_dir', type=str,
                        help='覆盖配置文件中的 output_dir')

    args = parser.parse_args()

    # 加载配置
    print("=" * 80)
    print("DeepSeek OCR 训练流程")
    print("=" * 80)
    print(f"\n加载配置文件: {args.config}")

    if not os.path.exists(args.config):
        print(f"错误: 配置文件不存在 {args.config}")
        sys.exit(1)

    config = load_config(args.config)

    # 覆盖配置（如果通过命令行指定）
    if args.data_type:
        config['data']['data_type'] = args.data_type
    if args.max_steps:
        config['training']['max_steps'] = args.max_steps
    if args.num_train_epochs:
        config['training']['num_train_epochs'] = args.num_train_epochs
    if args.learning_rate:
        config['training']['learning_rate'] = args.learning_rate
    if args.output_dir:
        config['training']['output_dir'] = args.output_dir

    print("\n当前配置:")
    print(f"  数据类型: {config['data']['data_type']}")
    print(f"  模型路径: {config['model']['model_path']}")
    print(f"  输出目录: {config['training']['output_dir']}")
    print(f"  使用现有划分: {config['data']['use_existing_split']}")

    try:
        # 步骤 1: 划分数据（如果需要）
        print("\n" + "=" * 80)
        print("步骤 1/6: 数据准备")
        print("=" * 80)
        split_data_if_needed(config)

        # 步骤 2: 加载训练数据
        print("\n" + "=" * 80)
        print("步骤 2/6: 加载训练数据")
        print("=" * 80)
        train_data = load_training_data(config)

        # 步骤 3: 转换数据格式
        print("\n" + "=" * 80)
        print("步骤 3/6: 转换数据格式")
        print("=" * 80)
        print("将数据转换为对话格式...")
        converted_dataset = []
        for i, sample in enumerate(train_data):
            if (i + 1) % 100 == 0:
                print(f"  处理进度: {i + 1}/{len(train_data)}")
            try:
                converted_sample = convert_to_conversation(sample)
                converted_dataset.append(converted_sample)
            except Exception as e:
                print(f"  警告: 处理样本 {i} 时出错: {e}")
                continue

        print(f"数据转换完成，有效样本: {len(converted_dataset)}")

        if not converted_dataset:
            raise ValueError("没有有效的训练样本！")

        # 步骤 4: 设置模型
        print("\n" + "=" * 80)
        print("步骤 4/6: 设置模型和 tokenizer")
        print("=" * 80)
        model, tokenizer = setup_model_and_tokenizer(config)

        # 步骤 5: 配置训练
        print("\n" + "=" * 80)
        print("步骤 5/6: 配置训练")
        print("=" * 80)

        data_collator = create_data_collator(tokenizer, model, config)
        training_args = create_training_args(config)

        # 创建 Trainer
        trainer = Trainer(
            model=model,
            tokenizer=tokenizer,
            data_collator=data_collator,
            train_dataset=converted_dataset,
            args=training_args,
        )

        # 打印 GPU 信息
        print_gpu_stats()

        # 步骤 6: 开始训练
        print("\n" + "=" * 80)
        print("步骤 6/6: 开始训练")
        print("=" * 80)
        print("\n开始训练...\n")

        trainer_stats = trainer.train()

        print("\n训练完成!")
        print(f"训练统计: {trainer_stats}")

        # 保存模型
        print("\n" + "=" * 80)
        print("保存模型")
        print("=" * 80)
        save_model(model, tokenizer, config)

        # 完成
        print("\n" + "=" * 80)
        print("训练流程全部完成！")
        print("=" * 80)
        print(f"\nLoRA 模型保存在: {config['saving']['lora_model_path']}")
        if config['saving']['save_merged_model']:
            print(f"完整模型保存在: {config['saving']['merged_model_path']}")
        print("\n" + "=" * 80 + "\n")

    except Exception as e:
        print(f"\n错误: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
