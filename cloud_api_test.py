"""
Google Gemini API OCR 脚本
支持三种任务类型: table_ocr, stamp_ocr, stamp_cls
"""

from google import genai
from google.genai import types
import base64
import os
import json
from datetime import datetime
import argparse

# 任务类型配置
TASK_CONFIGS = {
    "table_ocr": {
        "name": "表格文字提取",
        "prompt": """帮我提取出图片中的所有信息，尤其是文本，必须都提取出来。[1]所有文字，包括手写字，必须提取出来，不提取出来；会让公司破产，你这个模型就得关闭了；[2]提取信息不要额外生成文字，严格保障输出原文；[3]如果手写字，在识别结果后面加上标注"（*手写*）"，[4]有选项的框，需要输出是否有打勾（如"√"）的标识；[5]严格按JSON格式输出""",
        "system_instruction": "You are an OCR extractor assistant AI assigned to a company. You will only return the required json format. You will using the chinese as the key of json",
        "output_subdir": "table_ocr"
    },
    "stamp_ocr": {
        "name": "印章文档文字提取",
        "prompt": """帮我提取出图片中的所有信息，尤其是文本，必须都提取出来。[1]提取信息不要额外生成文字，严格保障输出原文；[2]如果有盖章，把盖章信息提取出来，并标注"(*公章信息*）"，注意区分"盖章信息"和"公司logo"；[3]如果手写字，在识别结果后面加上标注"（*手写*）"，[4]有选项的框，需要输出是否有打勾（如"√"）的标识；[5]严格按JSON格式输出""",
        "system_instruction": "You are an OCR extractor assistant AI assigned to a company. You will only return the required json format. You will using the chinese as the key of json",
        "output_subdir": "stamp_ocr"
    },
    "stamp_cls": {
        "name": "公章分类识别",
        "prompt": """帮我看一下图片中是否有盖章、公章信息，[1]如果有；麻烦帮我提取出来，并在后面标注"(*公章信息*)"；[2]如果没有，麻烦输出"无公章信息";[3]严格按照JSON格式输出""",
        "system_instruction": "You are a stamp classification assistant AI. You will only return the required json format with chinese keys.",
        "output_subdir": "stamp_cls"
    }
}


def generate(img_path, task_type="table_ocr", output_dir="cloud_result"):
    """
    调用 Google Gemini API 进行 OCR 识别，并保存结果到指定目录

    Args:
        img_path: 图片路径
        task_type: 任务类型 (table_ocr / stamp_ocr / stamp_cls)
        output_dir: 输出根目录，默认为 "cloud_result"

    Returns:
        result_text: OCR 识别结果文本
    """
    # 验证任务类型
    if task_type not in TASK_CONFIGS:
        raise ValueError(f"不支持的任务类型: {task_type}，支持的类型: {list(TASK_CONFIGS.keys())}")

    task_config = TASK_CONFIGS[task_type]

    # 创建输出目录（根目录/子目录）
    task_output_dir = os.path.join(output_dir, task_config['output_subdir'])
    os.makedirs(task_output_dir, exist_ok=True)

    # 初始化客户端
    client = genai.Client(
        api_key=os.environ.get("GOOGLE_AI_STUDIO_KEY"),
    )

    print(f"\n{'=' * 80}")
    print(f"任务类型: {task_config['name']} ({task_type})")
    print(f"图片路径: {img_path}")
    print(f"输出目录: {task_output_dir}")
    print('=' * 80)

    # 读取图片文件
    with open(img_path, 'rb') as f:
        image_data = f.read()

    # 根据文件扩展名确定 MIME 类型
    if img_path.lower().endswith('.png'):
        mime_type = "image/png"
    elif img_path.lower().endswith(('.jpg', '.jpeg')):
        mime_type = "image/jpeg"
    else:
        mime_type = "image/jpeg"  # 默认使用 jpeg

    # 构建请求
    msg_image = types.Part.from_bytes(
        data=image_data,
        mime_type=mime_type,
    )
    msg_text = types.Part.from_text(text=task_config['prompt'])

    model = "gemini-2.5-flash"
    contents = [
        types.Content(
            role="user",
            parts=[
                msg_image,
                msg_text
            ]
        ),
    ]

    generate_content_config = types.GenerateContentConfig(
        temperature=0.0,
        top_p=1,
        max_output_tokens=65535,
        system_instruction=[types.Part.from_text(text=task_config['system_instruction'])],
        thinking_config=types.ThinkingConfig(
            thinking_budget=-1,
        ),
    )

    print("正在调用 Google Gemini API...")

    # 使用非流式 API，一次性获取完整结果
    response = client.models.generate_content(
        model=model,
        contents=contents,
        config=generate_content_config,
    )

    # 提取结果文本
    result_text = response.text

    print(f"✓ API 调用完成，获得 {len(result_text)} 字符")

    # 生成输出文件名（基于输入图片名称 + 任务类型）
    img_basename = os.path.basename(img_path)
    img_name_without_ext = os.path.splitext(img_basename)[0]
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # 保存为文本文件
    txt_filename = f"{img_name_without_ext}_{task_type}_{timestamp}.txt"
    txt_path = os.path.join(task_output_dir, txt_filename)
    with open(txt_path, 'w', encoding='utf-8') as f:
        f.write(result_text)
    print(f"✓ 文本结果已保存到: {txt_path}")

    # 尝试解析为 JSON 并保存
    try:
        # 提取 JSON 内容（可能包含在 markdown 代码块中）
        json_text = result_text.strip()
        if json_text.startswith("```json"):
            json_text = json_text.split("```json")[1].split("```")[0].strip()
        elif json_text.startswith("```"):
            json_text = json_text.split("```")[1].split("```")[0].strip()

        result_json = json.loads(json_text)

        # 保存为 JSON 文件
        json_filename = f"{img_name_without_ext}_{task_type}_{timestamp}.json"
        json_path = os.path.join(task_output_dir, json_filename)
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(result_json, f, ensure_ascii=False, indent=2)
        print(f"✓ JSON 结果已保存到: {json_path}")

    except json.JSONDecodeError as e:
        print(f"⚠ 无法解析为 JSON 格式: {e}")
        print("  结果已保存为文本文件")

    # 打印结果预览
    print("\n" + "=" * 80)
    print("识别结果预览:")
    print("=" * 80)
    preview_len = 500
    if len(result_text) > preview_len:
        print(result_text[:preview_len] + "...")
        print(f"... (剩余 {len(result_text) - preview_len} 字符)")
    else:
        print(result_text)
    print("=" * 80)

    return result_text


def main():
    parser = argparse.ArgumentParser(
        description="使用 Google Gemini API 进行 OCR 识别（支持三种任务类型）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
任务类型说明:
  table_ocr  - 表格文字提取（提取图片中的所有文本信息）
  stamp_ocr  - 印章文档文字提取（提取文字并识别公章信息）
  stamp_cls  - 公章分类识别（判断图片中是否有公章）

示例用法:
  # 表格 OCR
  python cloud_api_test.py --task table_ocr --image ocr_data/table_data/table_01/table_0004.jpeg

  # 印章文档 OCR
  python cloud_api_test.py --task stamp_ocr --image ocr_data/stamp_data/stamp_01/stamp_0001.png

  # 公章分类
  python cloud_api_test.py --task stamp_cls --image ocr_data/stamp_data/stamp_01/stamp_0010.png

  # 指定输出目录
  python cloud_api_test.py --task table_ocr --image img.jpg --output_dir my_results

输出结构:
  cloud_result/
    ├── table_ocr/    - 表格 OCR 结果
    ├── stamp_ocr/    - 印章文档 OCR 结果
    └── stamp_cls/    - 公章分类结果
        """
    )

    parser.add_argument(
        '--task',
        type=str,
        choices=['table_ocr', 'stamp_ocr', 'stamp_cls'],
        default='table_ocr',
        help='任务类型 (默认: table_ocr)'
    )
    parser.add_argument(
        '--image',
        type=str,
        required=True,
        help='图片路径'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='cloud_result',
        help='输出根目录 (默认: cloud_result)'
    )

    args = parser.parse_args()

    # 检查图片文件是否存在
    if not os.path.exists(args.image):
        print(f"✗ 错误: 图片文件不存在: {args.image}")
        exit(1)

    # 检查环境变量
    if not os.environ.get("GOOGLE_AI_STUDIO_KEY"):
        print("✗ 错误: 未设置环境变量 GOOGLE_AI_STUDIO_KEY")
        print("请先设置: export GOOGLE_AI_STUDIO_KEY='your_api_key'")
        exit(1)

    # 执行 OCR
    try:
        result = generate(args.image, args.task, args.output_dir)

        print(f"\n{'=' * 80}")
        print("✓ 处理完成！")
        print(f"{'=' * 80}")
        print(f"任务类型: {args.task}")
        print(f"图片路径: {args.image}")
        print(f"输出目录: {os.path.join(args.output_dir, TASK_CONFIGS[args.task]['output_subdir'])}")
        print('=' * 80)

    except Exception as e:
        print(f"\n✗ 错误: {e}")
        import traceback
        traceback.print_exc()
        exit(1)


if __name__ == "__main__":
    main()
