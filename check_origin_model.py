#!/usr/bin/env python3
"""
检查模型目录结构和必要文件
支持自动下载缺失的模型
"""
import os
import json
import sys
import argparse

# 尝试导入 huggingface_hub
try:
    from huggingface_hub import snapshot_download
    HF_AVAILABLE = True
except ImportError:
    HF_AVAILABLE = False


def download_model(model_id: str, local_dir: str) -> bool:
    """
    从 Hugging Face Hub 下载模型

    Args:
        model_id: 模型ID (例如: unsloth/DeepSeek-OCR)
        local_dir: 本地保存目录

    Returns:
        下载是否成功
    """
    if not HF_AVAILABLE:
        print("❌ 错误: huggingface_hub 未安装")
        print("请安装: pip install huggingface_hub")
        return False

    try:
        print(f"\n{'=' * 60}")
        print(f"开始下载模型: {model_id}")
        print(f"保存目录: {local_dir}")
        print(f"{'=' * 60}")
        print("这可能需要较长时间，请耐心等待...\n")

        # 下载模型
        snapshot_download(
            repo_id=model_id,
            local_dir=local_dir,
            local_dir_use_symlinks=False  # 直接复制文件，不使用符号链接
        )

        print(f"\n{'=' * 60}")
        print("✅ 模型下载完成！")
        print(f"{'=' * 60}\n")
        return True

    except Exception as e:
        print(f"\n{'=' * 60}")
        print(f"❌ 模型下载失败: {e}")
        print(f"{'=' * 60}\n")
        return False


def check_model_directory(model_dir: str) -> tuple:
    """
    检查模型目录是否存在以及是否包含必要文件

    Args:
        model_dir: 模型目录路径

    Returns:
        (目录是否存在, 是否包含必要文件)
    """
    print("=" * 60)
    print("检查模型目录:", model_dir)
    print("=" * 60)

    # 检查目录是否存在
    if not os.path.exists(model_dir):
        print(f"❌ 目录不存在: {model_dir}")
        return False, False

    print(f"✅ 目录存在: {model_dir}")

    # 检查是否为空目录
    try:
        files = os.listdir(model_dir)
        if not files:
            print("❌ 目录为空！")
            return True, False
    except Exception as e:
        print(f"❌ 读取目录失败: {e}")
        return True, False

    # 检查必要文件
    required_files = ["config.json"]
    has_required = all(os.path.exists(os.path.join(model_dir, f)) for f in required_files)

    return True, has_required


def validate_model(model_dir: str) -> bool:
    """
    验证模型文件的完整性

    Args:
        model_dir: 模型目录路径

    Returns:
        验证是否通过
    """
    # 列出目录内容
    print("\n目录内容:")
    print("-" * 60)

    try:
        files = os.listdir(model_dir)
        if not files:
            print("❌ 目录为空！")
            return False
        else:
            for item in sorted(files):
                full_path = os.path.join(model_dir, item)
                if os.path.isdir(full_path):
                    print(f"📁 {item}/")
                else:
                    size = os.path.getsize(full_path)
                    size_mb = size / (1024 * 1024)
                    print(f"📄 {item} ({size_mb:.2f} MB)")
    except Exception as e:
        print(f"❌ 读取目录失败: {e}")
        return False

    # 检查必要文件
    print("\n" + "=" * 60)
    print("检查必要文件:")
    print("=" * 60)

    required_files = {
        "config.json": "模型配置文件",
        "tokenizer_config.json": "分词器配置",
        "processor_config.json": "预处理器配置",
    }

    missing_required = []
    for filename, description in required_files.items():
        file_path = os.path.join(model_dir, filename)
        if os.path.exists(file_path):
            print(f"✅ {filename} - {description}")
            # 尝试读取 config.json 内容
            if filename == "config.json":
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        config = json.load(f)
                    print(f"   模型类型: {config.get('model_type', 'unknown')}")
                    print(f"   架构: {config.get('architectures', 'unknown')}")
                except Exception as e:
                    print(f"   ⚠️  读取配置文件失败: {e}")
        else:
            print(f"❌ {filename} - {description} (缺失)")
            missing_required.append(filename)

    # 检查模型权重文件
    print("\n检查模型权重文件:")
    print("-" * 60)

    model_weight_patterns = [
        "model.safetensors",
        "pytorch_model.bin",
        "model.pt",
    ]

    has_weights = False
    for pattern in model_weight_patterns:
        file_path = os.path.join(model_dir, pattern)
        if os.path.exists(file_path):
            size = os.path.getsize(file_path)
            size_gb = size / (1024 * 1024 * 1024)
            print(f"✅ {pattern} ({size_gb:.2f} GB)")
            has_weights = True

    # 检查是否有分片文件
    sharded_files = [f for f in files if f.startswith("model-") or f.startswith("pytorch_model-")]
    if sharded_files:
        print(f"✅ 找到 {len(sharded_files)} 个分片文件")
        has_weights = True

    if not has_weights:
        print("❌ 未找到模型权重文件")

    # 诊断总结
    print("\n" + "=" * 60)
    print("诊断总结:")
    print("=" * 60)

    has_config = "config.json" not in missing_required

    if not has_config:
        print("🔴 主要问题: 缺少 config.json 文件")
        print("\n可能的原因:")
        print("1. 模型下载不完整")
        print("2. 下载到了错误的目录")
        print("3. 使用了错误的下载方法")
        return False
    elif not has_weights:
        print("🟡 警告: config.json 存在但缺少模型权重文件")
        return False
    else:
        print("🟢 所有必要文件看起来都存在")
        return True


def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description="检查并验证 DeepSeek-OCR 模型文件",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法:
  # 检查模型并在缺失时提示下载
  python check_origin_model.py

  # 自动下载模型（无需确认）
  python check_origin_model.py --auto-download

  # 指定自定义模型目录
  python check_origin_model.py --model_dir ./my_model

  # 指定自定义模型ID
  python check_origin_model.py --model_id unsloth/DeepSeek-OCR
        """
    )

    parser.add_argument(
        '--model_dir',
        type=str,
        default='./deepseek_ocr',
        help='模型目录路径 (默认: ./deepseek_ocr)'
    )
    parser.add_argument(
        '--model_id',
        type=str,
        default='unsloth/DeepSeek-OCR',
        help='Hugging Face 模型ID (默认: unsloth/DeepSeek-OCR)'
    )
    parser.add_argument(
        '--auto-download',
        action='store_true',
        help='自动下载模型，无需确认'
    )

    args = parser.parse_args()

    # 检查模型目录
    dir_exists, has_required = check_model_directory(args.model_dir)

    # 如果目录不存在或缺少必要文件
    if not dir_exists or not has_required:
        need_download = args.auto_download

        if not args.auto_download:
            print("\n" + "=" * 60)
            if not dir_exists:
                print(f"模型目录不存在: {args.model_dir}")
            else:
                print(f"模型目录缺少必要文件")

            response = input(f"\n是否从 Hugging Face 下载模型 ({args.model_id})? [y/N]: ")
            need_download = response.lower() in ['y', 'yes']

        if need_download:
            # 创建目录（如果不存在）
            os.makedirs(args.model_dir, exist_ok=True)

            # 下载模型
            if not download_model(args.model_id, args.model_dir):
                print("❌ 模型下载失败")
                sys.exit(1)
        else:
            print("\n跳过下载。请手动下载模型或使用 --auto-download 参数")
            sys.exit(1)

    # 验证模型文件
    print("\n" + "=" * 60)
    print("验证模型文件:")
    print("=" * 60)

    if validate_model(args.model_dir):
        print("\n" + "=" * 60)
        print("✅ 模型验证通过！可以正常使用")
        print("=" * 60)
        sys.exit(0)
    else:
        print("\n" + "=" * 60)
        print("❌ 模型验证失败，请检查上述错误")
        print("=" * 60)
        sys.exit(1)


if __name__ == "__main__":
    main()
