# DeepSeek OCR 训练与评估框架

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

基于 [Unsloth](https://github.com/unslothai/unsloth) 和 [DeepSeek-OCR](https://huggingface.co/unsloth/DeepSeek-OCR) 的高效 OCR 模型训练和评估框架。支持表格识别、印章识别和文档 OCR 等多种任务。

## ✨ 主要特性

- 🚀 **一键式完整工作流程** - 从模型检查到结果对比的全自动化流程
- 🎯 **多任务支持** - 支持 Table OCR、Stamp OCR、Stamp 分类三种任务
- ⚡ **高效训练** - 基于 Unsloth 的 LoRA 微调，显存占用低、速度快
- 🔧 **灵活配置** - YAML 配置文件 + 命令行参数，完全可定制
- 📊 **自动评估** - 训练前后自动对比，生成详细报告
- 🎨 **友好界面** - 彩色终端输出，清晰的进度显示
- 💾 **断点续传** - 支持中断后继续训练和推理
- 🔄 **智能加载** - 自动检测 LoRA adapter 并加载基础模型

## 📋 目录

- [快速开始](#快速开始)
- [安装](#安装)
- [使用指南](#使用指南)
- [项目结构](#项目结构)
- [文档](#文档)
- [配置说明](#配置说明)
- [常见问题](#常见问题)
- [贡献指南](#贡献指南)

## 🚀 快速开始

### 方式 1: 一键完整流程（推荐）

```bash
# 克隆仓库
git clone <repository-url>
cd deepseek_ocr

# 安装依赖
pip install -r train_requirements.txt

# 运行完整训练和评估流程（自动下载模型）
python train_and_evaluate.py --auto_download_model
```

### 方式 2: 快速测试（10 步训练）

```bash
# 快速验证环境和流程
chmod +x quick_test.sh
./quick_test.sh
```

### 方式 3: 分步执行

```bash
# 1. 检查并下载模型
python check_origin_model.py --auto-download

# 2. 划分数据集
python split_ocr_data.py --data_type all

# 3. 训练模型
python train_model.py

# 4. 推理评估
python batch_inference.py \
    --inference_mode local \
    --model_path ./lora_model
```

## 📦 安装

### 环境要求

- Python 3.8+
- CUDA 11.8+ (GPU 训练推荐)
- 16GB+ GPU 显存 (推荐)
- 8GB+ RAM

### 依赖安装

```bash
# 基础依赖
pip install torch transformers pillow

# Unsloth (加速训练)
pip install unsloth

# PEFT (LoRA 支持)
pip install peft

# 训练额外依赖
pip install -r train_requirements.txt
```

### 可选依赖

```bash
# Cloud API 推理
pip install google-genai

# 模型下载
pip install huggingface_hub
```

## 📖 使用指南

### 完整训练和评估流程

`train_and_evaluate.py` 整合了完整的 6 步流程：

```bash
# 基本用法
python train_and_evaluate.py

# 自定义参数
python train_and_evaluate.py \
    --data_type stamp \
    --max_steps 100 \
    --learning_rate 2e-4

# 跳过某些步骤
python train_and_evaluate.py \
    --skip_model_check \
    --skip_data_split
```

**工作流程：**
1. ✅ 检查基础模型（可自动下载）
2. ✅ 划分数据集（train/test split）
3. ✅ 训练前评估（基线性能）
4. ✅ 训练模型（LoRA 微调）
5. ✅ 训练后评估（LoRA 性能）
6. ✅ 对比结果（生成报告）

### 单独训练模型

使用 `train_model.py` 只进行训练：

```bash
# 使用默认配置
python train_model.py

# 使用自定义配置
python train_model.py --config my_config.yaml

# 覆盖配置参数
python train_model.py \
    --data_type table \
    --max_steps 200 \
    --learning_rate 1e-4
```

### 批量推理

使用 `batch_inference.py` 进行批量推理：

```bash
# 使用基础模型
python batch_inference.py \
    --inference_mode local \
    --model_path ./deepseek_ocr \
    --data_type all

# 使用 LoRA 模型（自动检测并加载基础模型）
python batch_inference.py \
    --inference_mode local \
    --model_path ./lora_model \
    --data_type all

# 使用 Cloud API
export GOOGLE_AI_STUDIO_KEY='your_api_key'
python batch_inference.py \
    --inference_mode cloud \
    --data_type all
```

### 数据准备

#### 数据格式

数据应按以下结构组织：

```
ocr_data/
├── stamp_data/
│   └── stamp_01/
│       ├── stamp_0001.png
│       ├── stamp_0002.png
│       ├── stamp_ocr_01.json          # OCR 标注
│       └── stamp_ocr_01_extracted.json # 分类标注
└── table_data/
    └── table_01/
        ├── table_0001.png
        ├── table_0002.png
        └── table_ocr_01.json          # OCR 标注
```

#### 标注格式

**OCR 标注 (stamp_ocr_01.json / table_ocr_01.json):**

```json
{
  "results": [
    {
      "image_path": "stamp_data/stamp_01/stamp_0001.png",
      "prompt": "帮我提取出图片中的所有信息...",
      "result": {
        "公司名称": "某某科技有限公司",
        "日期": "2024-01-01",
        ...
      }
    }
  ]
}
```

**分类标注 (stamp_ocr_01_extracted.json):**

```json
{
  "results": [
    {
      "image_path": "stamp_data/stamp_01/stamp_0001.png",
      "prompt": "帮我看一下图片中是否有盖章...",
      "公章信息": "某某科技有限公司(*公章信息*)"
    }
  ]
}
```

#### 划分数据

```bash
# 划分所有数据
python split_ocr_data.py --data_type all

# 只划分 table 数据
python split_ocr_data.py --data_type table

# 自定义划分比例
python split_ocr_data.py \
    --data_type stamp \
    --train_ratio 0.8 \
    --seed 42
```

## 📁 项目结构

```
deepseek_ocr/
├── 核心脚本
│   ├── train_and_evaluate.py        # 完整工作流程（推荐）
│   ├── train_model.py               # 训练脚本
│   ├── batch_inference.py           # 批量推理
│   ├── check_origin_model.py        # 模型检查
│   ├── split_ocr_data.py            # 数据划分
│   └── run_pipeline.py              # 原始评估流程
│
├── 配置文件
│   ├── train_config.yaml            # 训练配置
│   └── train_requirements.txt       # 额外依赖
│
├── 数据处理
│   ├── unsloth_data_collator.py     # 数据整理器
│   ├── unsloth_deepseek_ocr.py      # DeepSeek OCR 模块
│   └── eval_utils.py                # 评估工具
│
├── 评估脚本
│   ├── table_ocr_eval/              # Table OCR 评估
│   ├── stamp_ocr_eval/              # Stamp OCR 评估
│   └── stamp_cls_eval/              # Stamp 分类评估
│
├── 文档
│   ├── README.md                    # 本文档
│   ├── COMPLETE_WORKFLOW.md         # 完整工作流程指南
│   ├── TRAINING_README.md           # 训练详细指南
│   ├── INFERENCE_WITH_LORA.md       # LoRA 推理指南
│   └── README_NEW_FEATURES.md       # 新功能总览
│
├── 辅助脚本
│   ├── quick_test.sh                # 快速测试
│   ├── deploy_env.sh                # 环境部署
│   └── test_eval.sh                 # 评估测试
│
└── 数据和模型
    ├── deepseek_ocr/                # 基础模型
    ├── ocr_data/                    # 训练数据
    ├── lora_model/                  # LoRA 模型
    ├── baseline_result/             # 训练前结果
    └── lora_result/                 # 训练后结果
```

## 📚 文档

- **[完整工作流程指南](COMPLETE_WORKFLOW.md)** - train_and_evaluate.py 详细使用说明
- **[训练指南](TRAINING_README.md)** - train_model.py 和 train_config.yaml 详细说明
- **[LoRA 推理指南](INFERENCE_WITH_LORA.md)** - LoRA 模型加载和使用方法
- **[新功能总览](README_NEW_FEATURES.md)** - 所有新增功能和改进

## ⚙️ 配置说明

### 训练配置 (train_config.yaml)

```yaml
# 数据配置
data:
  use_existing_split: false          # 是否使用已有数据划分
  data_type: "all"                   # 数据类型: all, table, stamp
  train_ratio: 0.8                   # 训练集比例

# 模型配置
model:
  model_path: "./deepseek_ocr"       # 基础模型路径
  load_in_4bit: false                # 是否使用 4bit 量化
  lora:
    r: 16                            # LoRA rank
    lora_alpha: 16                   # LoRA alpha
    lora_dropout: 0                  # LoRA dropout

# 训练配置
training:
  per_device_train_batch_size: 2     # Batch size
  gradient_accumulation_steps: 4     # 梯度累积步数
  learning_rate: 2e-4                # 学习率
  max_steps: 60                      # 最大训练步数
  # num_train_epochs: 1              # 或指定训练轮数

# 数据处理配置
data_processing:
  image_size: 640                    # 图像尺寸
  base_size: 1024                    # 基础尺寸
  crop_mode: true                    # 是否裁剪
  train_on_responses_only: true      # 只训练回复

# 保存配置
saving:
  lora_model_path: "lora_model"      # LoRA 模型保存路径
  save_merged_model: false           # 是否保存合并模型
```

### 命令行参数覆盖

```bash
# 覆盖配置文件中的参数
python train_model.py \
    --data_type stamp \
    --max_steps 100 \
    --learning_rate 1e-4 \
    --output_dir my_outputs
```

## 💡 使用场景示例

### 场景 1: 快速验证环境

```bash
# 10 步快速测试
./quick_test.sh
```

### 场景 2: 完整训练实验

```bash
# 训练 3 个 epoch
python train_and_evaluate.py \
    --data_type all \
    --num_train_epochs 3 \
    --summary_file experiments/exp_001.json
```

### 场景 3: 超参数搜索

```bash
# 实验 1: lr=1e-4
python train_and_evaluate.py \
    --learning_rate 1e-4 \
    --lora_model_path lora_lr1e4 \
    --summary_file exp_lr1e4.json

# 实验 2: lr=5e-4
python train_and_evaluate.py \
    --learning_rate 5e-4 \
    --lora_model_path lora_lr5e4 \
    --summary_file exp_lr5e4.json
```

### 场景 4: 特定任务训练

```bash
# 只训练 stamp 任务
python train_model.py \
    --data_type stamp \
    --max_steps 200
```

### 场景 5: 使用训练好的模型

```bash
# 使用 LoRA 模型进行推理
python batch_inference.py \
    --inference_mode local \
    --model_path ./lora_model \
    --data_type all
```

## 🎯 性能优化

### 减少显存占用

```yaml
# train_config.yaml
model:
  load_in_4bit: true                 # 启用 4bit 量化

training:
  per_device_train_batch_size: 1     # 减小 batch size
  gradient_accumulation_steps: 8     # 增加累积步数

data_processing:
  image_size: 512                    # 减小图像尺寸
  base_size: 640
  crop_mode: false                   # 禁用裁剪
```

### 加速训练

```yaml
model:
  use_gradient_checkpointing: "unsloth"  # 使用 unsloth 加速
  unsloth_force_compile: true            # 强制编译优化

training:
  dataloader_num_workers: 4              # 增加数据加载线程
```

## ❓ 常见问题

### Q1: 如何下载模型？

**A:** 脚本会自动提示下载，或使用：

```bash
python check_origin_model.py --auto-download
```

### Q2: 显存不足怎么办？

**A:**
1. 启用 4bit 量化：`load_in_4bit: true`
2. 减小 batch size：`per_device_train_batch_size: 1`
3. 减小图像尺寸：`image_size: 512, base_size: 640`
4. 禁用裁剪：`crop_mode: false`

### Q3: 如何使用训练好的 LoRA 模型？

**A:**

```bash
python batch_inference.py \
    --inference_mode local \
    --model_path ./lora_model
```

脚本会自动检测并加载基础模型。

### Q4: 训练中断了怎么办？

**A:** 使用 `--skip_*` 参数跳过已完成步骤：

```bash
python train_and_evaluate.py \
    --skip_model_check \
    --skip_data_split \
    --skip_baseline_inference
```

### Q5: 如何对比多个实验？

**A:** 使用不同的输出目录和总结文件：

```bash
python train_and_evaluate.py \
    --lora_model_path exp1/lora \
    --lora_output_dir exp1/result \
    --summary_file exp1.json
```

### Q6: 支持哪些数据类型？

**A:** 支持三种数据类型：
- `all`: 所有任务（table_ocr + stamp_ocr + stamp_cls）
- `table`: 表格 OCR
- `stamp`: 印章相关（stamp_ocr + stamp_cls）

### Q7: 如何自定义训练参数？

**A:** 两种方式：
1. 修改 `train_config.yaml`
2. 使用命令行参数覆盖：
   ```bash
   python train_model.py --max_steps 100 --learning_rate 1e-4
   ```

## 📊 评估指标

框架支持多种评估指标：

- **Table OCR**: 字符级准确率、编辑距离
- **Stamp OCR**: 字符级准确率、编辑距离
- **Stamp Classification**: 准确率、精确率、召回率

评估脚本会自动计算并输出详细指标。

## 🔧 故障排除

### 模型加载失败

```
TypeError: Unsloth: Cannot determine model type for config file: None
```

**解决方案**: 这通常是 LoRA adapter 缺少基础模型。脚本现在会自动检测并加载，或手动指定：

```bash
python batch_inference.py \
    --model_path ./lora_model \
    --base_model_path ./deepseek_ocr
```

### 类型错误

```
TypeError: '<=' not supported between instances of 'float' and 'str'
```

**解决方案**: 已修复，所有配置参数现在都有显式类型转换。

### 路径错误

```
expected str, bytes or os.PathLike object, not NoneType
```

**解决方案**: 已修复，`output_path` 参数现在会自动创建有效路径。

## 🤝 贡献指南

欢迎贡献代码、报告问题或提出建议！

### 开发设置

```bash
# 克隆仓库
git clone <repository-url>
cd deepseek_ocr

# 安装开发依赖
pip install -r train_requirements.txt

# 运行测试
./quick_test.sh
```

### 提交代码

1. Fork 仓库
2. 创建特性分支：`git checkout -b feature/AmazingFeature`
3. 提交更改：`git commit -m 'Add some AmazingFeature'`
4. 推送到分支：`git push origin feature/AmazingFeature`
5. 提交 Pull Request

## 📝 更新日志

### v2.0.0 (2025-12-02)

#### 新增
- ✨ 完整的训练和评估工作流程脚本 (`train_and_evaluate.py`)
- ✨ 基于 YAML 的训练配置系统
- ✨ LoRA adapter 自动检测和加载
- ✨ 训练前后自动对比和报告生成
- ✨ 彩色终端输出和进度显示
- ✨ 实验总结 JSON 输出

#### 改进
- 🔧 修复 LoRA 模型加载问题
- 🔧 修复配置参数类型转换问题
- 🔧 修复推理 output_path 错误
- 📚 完善文档系统

#### 优化
- ⚡ 模型加载缓存，避免重复加载
- ⚡ 断点续传支持
- ⚡ 增量保存推理结果

### v1.0.0

- 初始版本
- 基础的训练和推理功能
- Cloud API 支持

## 📄 许可证

本项目采用 MIT 许可证 - 详见 [LICENSE](LICENSE) 文件

## 🙏 致谢

- [Unsloth](https://github.com/unslothai/unsloth) - 高效的 LoRA 训练框架
- [DeepSeek](https://huggingface.co/deepseek-ai) - 强大的基础模型
- [Hugging Face](https://huggingface.co/) - 模型托管和工具

## 📮 联系方式

如有问题或建议，请通过以下方式联系：

- 提交 Issue: [GitHub Issues](https://github.com/your-repo/issues)
- 邮件: your-email@example.com

## ⭐ Star History

如果这个项目对你有帮助，请给我们一个 Star！

---

**快速链接:**
- [快速开始](#快速开始)
- [完整工作流程指南](COMPLETE_WORKFLOW.md)
- [训练详细指南](TRAINING_README.md)
- [LoRA 推理指南](INFERENCE_WITH_LORA.md)
- [常见问题](#常见问题)
