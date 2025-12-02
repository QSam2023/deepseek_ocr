"""
批量推理脚本 - 对测试集进行批量推理
支持两种推理模式:
1. Cloud API (Google Gemini)
2. Local Model (Unsloth DeepSeek-OCR)
"""

import os
import json
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from tqdm import tqdm

# Cloud API imports (lazy import to avoid dependency when using local mode)
try:
    from google import genai
    from google.genai import types
    CLOUD_API_AVAILABLE = True
except ImportError:
    CLOUD_API_AVAILABLE = False

# Local model imports (lazy import to avoid dependency when using cloud mode)
try:
    from unsloth import FastVisionModel
    import torch
    from transformers import AutoModel
    UNSLOTH_AVAILABLE = True
except ImportError:
    UNSLOTH_AVAILABLE = False

# 任务类型配置（与cloud_api_test.py保持一致）
TASK_CONFIGS = {
    "table_ocr": {
        "name": "表格文字提取",
        "prompt": """帮我提取出图片中的所有信息，尤其是文本，必须都提取出来。[1]所有文字，包括手写字，必须提取出来，不提取出来；会让公司破产，你这个模型就得关闭了；[2]提取信息不要额外生成文字，严格保障输出原文；[3]如果手写字，在识别结果后面加上标注"（*手写*）"，[4]有选项的框，需要输出是否有打勾（如"√"）的标识；[5]严格按JSON格式输出""",
        "system_instruction": "You are an OCR extractor assistant AI assigned to a company. You will only return the required json format. You will using the chinese as the key of json",
    },
    "stamp_ocr": {
        "name": "印章文档文字提取",
        "prompt": """帮我提取出图片中的所有信息，尤其是文本，必须都提取出来。[1]提取信息不要额外生成文字，严格保障输出原文；[2]如果有盖章，把盖章信息提取出来，并标注"(*公章信息*）"，注意区分"盖章信息"和"公司logo"；[3]如果手写字，在识别结果后面加上标注"（*手写*）"，[4]有选项的框，需要输出是否有打勾（如"√"）的标识；[5]严格按JSON格式输出""",
        "system_instruction": "You are an OCR extractor assistant AI assigned to a company. You will only return the required json format. You will using the chinese as the key of json",
    },
    "stamp_cls": {
        "name": "公章分类识别",
        "prompt": """帮我看一下图片中是否有盖章、公章信息，[1]如果有；麻烦帮我提取出来，并在后面标注"(*公章信息*)"；[2]如果没有，麻烦输出"无公章信息";[3]严格按照JSON格式输出""",
        "system_instruction": "You are a stamp classification assistant AI. You will only return the required json format with chinese keys.",
    }
}


def load_local_model(model_path: str, base_model_path: str = None) -> Tuple:
    """
    加载本地 Unsloth DeepSeek-OCR 模型

    Args:
        model_path: 模型路径（可以是完整模型或 LoRA adapter）
        base_model_path: 基础模型路径（当 model_path 是 LoRA adapter 时需要）

    Returns:
        (model, tokenizer) tuple
    """
    if not UNSLOTH_AVAILABLE:
        raise ImportError("Unsloth not available. Please install: pip install unsloth")

    # Suppress warnings
    os.environ["UNSLOTH_WARN_UNINITIALIZED"] = '0'

    # 检查是否是 LoRA adapter 目录
    is_lora_adapter = False
    if os.path.exists(model_path):
        adapter_config_path = os.path.join(model_path, "adapter_config.json")
        config_path = os.path.join(model_path, "config.json")

        # 如果有 adapter_config.json 但没有 config.json，说明是 LoRA adapter
        if os.path.exists(adapter_config_path) and not os.path.exists(config_path):
            is_lora_adapter = True
            print(f"\n检测到 LoRA adapter: {model_path}")

            # 如果没有提供 base_model_path，尝试使用默认路径
            if base_model_path is None:
                base_model_path = "./deepseek_ocr"
                print(f"使用默认基础模型路径: {base_model_path}")

            if not os.path.exists(base_model_path):
                raise ValueError(
                    f"LoRA adapter 需要基础模型，但基础模型路径不存在: {base_model_path}\n"
                    f"请使用 --base_model_path 参数指定基础模型路径，或确保 ./deepseek_ocr 存在"
                )

    if is_lora_adapter:
        # 先加载基础模型
        print(f"步骤 1/2: 加载基础模型 {base_model_path}")
        base_model, tokenizer = FastVisionModel.from_pretrained(
            base_model_path,
            load_in_4bit=False,
            auto_model=AutoModel,
            trust_remote_code=True,
            unsloth_force_compile=True,
            use_gradient_checkpointing="unsloth",
        )

        # 加载 LoRA adapter
        print(f"步骤 2/2: 加载 LoRA adapter {model_path}")
        from peft import PeftModel
        model = PeftModel.from_pretrained(base_model, model_path)

        # 注意：不合并 adapter，保持 infer() 方法可用
        # 直接使用 PeftModel 进行推理
        print("✓ LoRA adapter 已加载（保持 adapter 分离以使用 infer 方法）")

    else:
        # 直接加载完整模型
        print(f"\n加载模型: {model_path}")
        print("这可能需要一些时间...")

        model, tokenizer = FastVisionModel.from_pretrained(
            model_path,
            load_in_4bit=False,
            auto_model=AutoModel,
            trust_remote_code=True,
            unsloth_force_compile=True,
            use_gradient_checkpointing="unsloth",
        )

    print("✓ 模型加载完成")
    return model, tokenizer


def call_local_model(img_path: str, task_type: str, model, tokenizer) -> Dict:
    """
    调用本地 Unsloth 模型进行单张图片推理

    Args:
        img_path: 图片路径
        task_type: 任务类型
        model: Unsloth model
        tokenizer: Tokenizer

    Returns:
        解析后的JSON结果
    """
    task_config = TASK_CONFIGS[task_type]

    # 构建 prompt（包含图像标记和任务指令）
    prompt = f"<image>\n{task_config['system_instruction']}\n\n{task_config['prompt']}"

    # 创建临时输出目录（即使不保存结果也需要提供有效路径）
    temp_output_dir = os.path.join(os.path.dirname(img_path), '.temp_inference')
    os.makedirs(temp_output_dir, exist_ok=True)

    # 调用模型推理
    result_text = model.infer(
        tokenizer,
        prompt=prompt,
        image_file=img_path,
        output_path=temp_output_dir,  # 提供有效路径，但不保存结果
        base_size=1024,
        image_size=640,
        crop_mode=True,
        save_results=False,
        test_compress=False
    )

    # 检查返回结果
    if result_text is None:
        raise ValueError(f"模型推理返回 None，图片: {img_path}")

    # 解析JSON
    json_text = result_text.strip()
    if json_text.startswith("```json"):
        json_text = json_text.split("```json")[1].split("```")[0].strip()
    elif json_text.startswith("```"):
        json_text = json_text.split("```")[1].split("```")[0].strip()

    try:
        result_json = json.loads(json_text)
        return result_json
    except json.JSONDecodeError as e:
        print(f"⚠ 警告: 无法解析JSON ({img_path}): {e}")
        # 返回原始文本
        return {"raw_text": result_text}


def call_api(img_path: str, task_type: str, client) -> Dict:
    """
    调用Gemini API进行单张图片推理

    Args:
        img_path: 图片路径
        task_type: 任务类型
        client: Gemini客户端

    Returns:
        解析后的JSON结果
    """
    task_config = TASK_CONFIGS[task_type]

    # 读取图片
    with open(img_path, 'rb') as f:
        image_data = f.read()

    # 确定MIME类型
    if img_path.lower().endswith('.png'):
        mime_type = "image/png"
    elif img_path.lower().endswith(('.jpg', '.jpeg')):
        mime_type = "image/jpeg"
    else:
        mime_type = "image/jpeg"

    # 构建请求
    msg_image = types.Part.from_bytes(data=image_data, mime_type=mime_type)
    msg_text = types.Part.from_text(text=task_config['prompt'])

    contents = [
        types.Content(
            role="user",
            parts=[msg_image, msg_text]
        ),
    ]

    generate_content_config = types.GenerateContentConfig(
        temperature=0.0,
        top_p=1,
        max_output_tokens=65535,
        system_instruction=[types.Part.from_text(text=task_config['system_instruction'])],
        thinking_config=types.ThinkingConfig(thinking_budget=-1),
    )

    # 调用API
    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=contents,
        config=generate_content_config,
    )

    result_text = response.text

    # 解析JSON
    json_text = result_text.strip()
    if json_text.startswith("```json"):
        json_text = json_text.split("```json")[1].split("```")[0].strip()
    elif json_text.startswith("```"):
        json_text = json_text.split("```")[1].split("```")[0].strip()

    try:
        result_json = json.loads(json_text)
        return result_json
    except json.JSONDecodeError as e:
        print(f"⚠ 警告: 无法解析JSON ({img_path}): {e}")
        # 返回原始文本
        return {"raw_text": result_text}


def batch_inference(task_type: str, split_data_dir: str, output_dir: str,
                   resume: bool = True, inference_mode: str = 'cloud',
                   model_path: Optional[str] = None, base_model_path: Optional[str] = None,
                   local_model_cache: Optional[Tuple] = None):
    """
    对特定任务类型的测试集进行批量推理

    Args:
        task_type: 任务类型 (table_ocr/stamp_ocr/stamp_cls)
        split_data_dir: 划分后的数据目录
        output_dir: 输出根目录
        resume: 是否跳过已处理的图片（断点续传）
        inference_mode: 推理模式 ('cloud' 或 'local')
        model_path: 本地模型路径（仅 local 模式需要）
        base_model_path: 基础模型路径（当 model_path 是 LoRA adapter 时需要）
        local_model_cache: 已加载的本地模型缓存 (model, tokenizer)

    Returns:
        本地模型缓存 (model, tokenizer) 如果是 local 模式，否则返回 None
    """
    # 读取测试集数据
    test_file = os.path.join(split_data_dir, f"{task_type}_test.json")
    if not os.path.exists(test_file):
        print(f"⚠ 测试集文件不存在，跳过: {test_file}")
        return local_model_cache

    with open(test_file, 'r', encoding='utf-8') as f:
        test_data = json.load(f)

    print(f"\n{'=' * 80}")
    print(f"任务类型: {TASK_CONFIGS[task_type]['name']} ({task_type})")
    print(f"推理模式: {inference_mode.upper()}")
    print(f"测试集文件: {test_file}")
    print(f"测试集大小: {len(test_data)} 张图片")
    print('=' * 80)

    # 创建输出目录
    task_output_dir = os.path.join(output_dir, "test", task_type)
    os.makedirs(task_output_dir, exist_ok=True)

    # 初始化推理客户端/模型
    if inference_mode == 'cloud':
        if not CLOUD_API_AVAILABLE:
            raise ImportError("Cloud API not available. Please install: pip install google-genai")
        client = genai.Client(api_key=os.environ.get("GOOGLE_AI_STUDIO_KEY"))
        model = None
        tokenizer = None
    else:  # local mode
        if local_model_cache is not None:
            model, tokenizer = local_model_cache
            print("✓ 使用已加载的模型缓存")
        else:
            if not model_path:
                raise ValueError("Local mode requires --model_path argument")
            model, tokenizer = load_local_model(model_path, base_model_path)
            local_model_cache = (model, tokenizer)
        client = None

    # 准备结果列表
    results = []
    processed_images = set()

    # 如果resume，加载已有结果
    output_file = os.path.join(task_output_dir, f"{task_type}_predictions.json")
    if resume and os.path.exists(output_file):
        with open(output_file, 'r', encoding='utf-8') as f:
            existing_results = json.load(f)
            results = existing_results
            processed_images = {r['image_name'] for r in results}
        print(f"✓ 加载已有结果: {len(processed_images)} 张图片，继续处理剩余图片...")

    # 批量处理
    skipped = 0
    errors = 0

    for item in tqdm(test_data, desc=f"处理 {task_type}"):
        img_path = item['image_path']
        img_name = os.path.basename(img_path)

        # 断点续传：跳过已处理的
        if resume and img_name in processed_images:
            skipped += 1
            continue

        # 检查图片是否存在
        if not os.path.exists(img_path):
            print(f"⚠ 图片不存在，跳过: {img_path}")
            errors += 1
            continue

        try:
            # 根据模式调用不同的推理方法
            if inference_mode == 'cloud':
                pred_result = call_api(img_path, task_type, client)
            else:  # local mode
                pred_result = call_local_model(img_path, task_type, model, tokenizer)

            # 构建结果条目（与评估脚本期望的格式一致）
            if task_type == "stamp_cls":
                # stamp_cls的结果格式特殊，需要包含"公章信息"字段
                result_entry = {
                    "image_name": img_name,
                    "公章信息": pred_result.get("公章信息", "无公章信息")
                }
            else:
                # table_ocr和stamp_ocr使用标准格式
                result_entry = {
                    "image_name": img_name,
                    "result": pred_result
                }

            results.append(result_entry)
            processed_images.add(img_name)

            # 增量保存（每处理一张就保存，防止中断丢失）
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(results, f, ensure_ascii=False, indent=2)

        except Exception as e:
            print(f"✗ 处理失败 ({img_name}): {e}")
            errors += 1
            continue

    print(f"\n{'=' * 80}")
    print(f"✓ {task_type} 批量推理完成！")
    print(f"  - 总数: {len(test_data)} 张")
    print(f"  - 成功: {len(results)} 张")
    print(f"  - 跳过: {skipped} 张")
    print(f"  - 失败: {errors} 张")
    print(f"  - 结果保存: {output_file}")
    print('=' * 80)

    return local_model_cache


def main():
    parser = argparse.ArgumentParser(
        description="批量推理脚本 - 支持Cloud API和本地模型推理",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法:
  # 使用Cloud API进行推理 (默认)
  python batch_inference.py --data_type all

  # 使用本地完整模型进行推理
  python batch_inference.py --data_type all --inference_mode local --model_path ./deepseek_ocr

  # 使用 LoRA adapter 进行推理（自动检测并加载基础模型）
  python batch_inference.py --data_type all --inference_mode local --model_path ./lora_model

  # 使用 LoRA adapter 并手动指定基础模型
  python batch_inference.py --data_type all --inference_mode local --model_path ./lora_model --base_model_path ./deepseek_ocr

  # 只对table_ocr进行推理
  python batch_inference.py --data_type table --inference_mode local --model_path ./lora_model

  # 从头开始（不跳过已处理的图片）
  python batch_inference.py --data_type all --no-resume

输出结构:
  cloud_result/test/ (或 local_result/test/)
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
        '--split_data_dir',
        type=str,
        default='ocr_data/splited_data',
        help='划分后的数据目录 (默认: ocr_data/splited_data)'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default=None,
        help='输出根目录 (默认: cloud模式为cloud_result, local模式为local_result)'
    )
    parser.add_argument(
        '--inference_mode',
        type=str,
        choices=['cloud', 'local'],
        default='cloud',
        help='推理模式: cloud (Google Gemini API) 或 local (Unsloth模型) (默认: cloud)'
    )
    parser.add_argument(
        '--model_path',
        type=str,
        default='./deepseek_ocr',
        help='本地模型路径 (仅在local模式下使用, 默认: ./deepseek_ocr)'
    )
    parser.add_argument(
        '--base_model_path',
        type=str,
        default=None,
        help='基础模型路径 (当 model_path 是 LoRA adapter 时需要, 默认: ./deepseek_ocr)'
    )
    parser.add_argument(
        '--no-resume',
        action='store_true',
        help='不使用断点续传，从头开始处理'
    )

    args = parser.parse_args()

    # 设置默认输出目录
    if args.output_dir is None:
        args.output_dir = 'cloud_result' if args.inference_mode == 'cloud' else 'local_result'

    # 检查推理模式的依赖
    if args.inference_mode == 'cloud':
        if not CLOUD_API_AVAILABLE:
            print("✗ 错误: Cloud API 依赖未安装")
            print("请安装: pip install google-genai")
            exit(1)
        if not os.environ.get("GOOGLE_AI_STUDIO_KEY"):
            print("✗ 错误: 未设置环境变量 GOOGLE_AI_STUDIO_KEY")
            print("请先设置: export GOOGLE_AI_STUDIO_KEY='your_api_key'")
            exit(1)
    else:  # local mode
        if not UNSLOTH_AVAILABLE:
            print("✗ 错误: Unsloth 依赖未安装")
            print("请安装: pip install unsloth")
            exit(1)
        if not os.path.exists(args.model_path):
            print(f"✗ 错误: 模型路径不存在: {args.model_path}")
            print("请先下载模型或指定正确的模型路径")
            exit(1)

    # 检查split_data_dir是否存在
    if not os.path.exists(args.split_data_dir):
        print(f"✗ 错误: 数据目录不存在: {args.split_data_dir}")
        print("请先运行 split_ocr_data.py 进行数据划分")
        exit(1)

    resume = not args.no_resume

    # 根据data_type确定要处理的任务
    if args.data_type == 'all':
        tasks = ['table_ocr', 'stamp_ocr', 'stamp_cls']
    elif args.data_type == 'table':
        tasks = ['table_ocr']
    else:  # stamp
        tasks = ['stamp_ocr', 'stamp_cls']

    print(f"\n{'#' * 80}")
    print(f"批量推理任务开始")
    print(f"推理模式: {args.inference_mode.upper()}")
    print(f"数据类型: {args.data_type}")
    print(f"输出目录: {args.output_dir}")
    if args.inference_mode == 'local':
        print(f"模型路径: {args.model_path}")
    print('#' * 80)

    # 本地模型缓存（避免重复加载）
    local_model_cache = None

    # 批量处理所有任务
    for task_type in tasks:
        try:
            local_model_cache = batch_inference(
                task_type=task_type,
                split_data_dir=args.split_data_dir,
                output_dir=args.output_dir,
                resume=resume,
                inference_mode=args.inference_mode,
                model_path=args.model_path if args.inference_mode == 'local' else None,
                base_model_path=args.base_model_path if args.inference_mode == 'local' else None,
                local_model_cache=local_model_cache
            )
        except Exception as e:
            print(f"✗ 任务 {task_type} 处理失败: {e}")
            import traceback
            traceback.print_exc()
            continue

    print(f"\n{'#' * 80}")
    print("✓ 所有批量推理任务完成！")
    print(f"结果保存在: {args.output_dir}/test/")
    print('#' * 80)


if __name__ == "__main__":
    main()
