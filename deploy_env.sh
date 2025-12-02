#!/bin/bash
set -e  # Exit on error
set -x  # Print commands

# Define versions for easier maintenance
PYTHON_VERSION="3.12"
TORCH_VERSION="2.6.0"
TORCHVISION_VERSION="0.21.0"
TORCHAUDIO_VERSION="2.6.0"
TRANSFORMERS_VERSION="4.46.3"
TOKENIZERS_VERSION="0.20.3"
FLASH_ATTN_VERSION="2.7.3"

echo "Starting environment deployment..."

# Disable auto-tmux
touch ~/.no_auto_tmux

# Install uv if not found
if ! command -v uv &> /dev/null; then
    echo "Installing uv..."
    curl -LskSf https://astral.sh/uv/install.sh | sh
    # Add to PATH for current session
    export PATH="$HOME/.local/bin:$PATH"
else
    echo "uv is already installed."
    export PATH="$HOME/.local/bin:$PATH"
fi

# Create virtual environment
if [ ! -d "ocr" ]; then
    echo "Creating virtual environment 'ocr' with Python $PYTHON_VERSION..."
    uv venv ocr --python "$PYTHON_VERSION"
else
    echo "Virtual environment 'ocr' already exists."
fi

# Activate virtual environment
source ocr/bin/activate

# Install dependencies
echo "Installing dependencies..."
uv pip install --upgrade pip

uv pip install torch=="${TORCH_VERSION}" torchvision=="${TORCHVISION_VERSION}" torchaudio=="${TORCHAUDIO_VERSION}" --index-url https://download.pytorch.org/whl/cu128

# Optional: vLLM (commented out as in original)
# uv pip install -U vllm --pre --extra-index-url https://wheels.vllm.ai/nightly

#uv pip install transformers=="${TRANSFORMERS_VERSION}" tokenizers=="${TOKENIZERS_VERSION}"
#uv pip install psutil
#uv pip install accelerate

# 基础依赖
uv pip install unsloth transformers==4.56.2 trl==0.22.2 jiwer einops addict easydict

# 如需 GPU 加速（可选但推荐）
uv pip install bitsandbytes accelerate peft triton cut_cross_entropy

# 若需要 tokenizer/data 相关
uv pip install sentencepiece protobuf "datasets==4.3.0" "huggingface_hub>=0.34.0" hf_transfer

uv pip install flash-attn=="${FLASH_ATTN_VERSION}" --no-build-isolation
uv pip install PyMuPDF img2pdf einops easydict addict Pillow numpy matplotlib PyYAML Pillow

# Clone repository
if [ ! -d "DeepSeek-OCR" ]; then
    echo "Cloning DeepSeek-OCR repository..."
    git clone https://github.com/deepseek-ai/DeepSeek-OCR
else
    echo "DeepSeek-OCR repository already exists."
fi

echo "Deployment finished successfully."
