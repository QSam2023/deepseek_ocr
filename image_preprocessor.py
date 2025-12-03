"""
图片预处理和缓存模块
在数据切分阶段预处理图片，训练时直接加载缓存，大幅提升训练速度
"""

import os
import torch
import hashlib
import math
from PIL import Image, ImageOps
from typing import Tuple, List, Dict, Any, Optional
from pathlib import Path


class ImagePreprocessor:
    """
    图片预处理器：将图片预处理为模型可用的张量并缓存
    """

    def __init__(
        self,
        image_size: int = 640,
        base_size: int = 1024,
        crop_mode: bool = True,
        cache_dir: str = "ocr_data/preprocessed_cache",
        image_token_id: int = 128815,
        dtype: torch.dtype = torch.float16,
    ):
        self.image_size = image_size
        self.base_size = base_size
        self.crop_mode = crop_mode
        self.cache_dir = cache_dir
        self.image_token_id = image_token_id
        self.dtype = dtype

        self.patch_size = 16
        self.downsample_ratio = 4

        # 图像标准化参数（与 BasicImageTransform 一致）
        self.mean = torch.tensor([0.5, 0.5, 0.5])
        self.std = torch.tensor([0.5, 0.5, 0.5])

        os.makedirs(cache_dir, exist_ok=True)

    def get_cache_path(self, image_path: str, task_type: str) -> str:
        """
        生成缓存文件路径
        使用图片路径的 hash 避免路径冲突
        """
        # 使用图片路径的 hash 作为缓存文件名
        path_hash = hashlib.md5(image_path.encode()).hexdigest()[:16]

        # 按任务类型分目录
        task_cache_dir = os.path.join(self.cache_dir, task_type)
        os.makedirs(task_cache_dir, exist_ok=True)

        cache_filename = f"{path_hash}.pt"
        return os.path.join(task_cache_dir, cache_filename)

    def image_to_tensor(self, image: Image.Image) -> torch.Tensor:
        """
        将 PIL Image 转换为标准化的张量
        """
        # Convert to tensor [C, H, W], range [0, 1]
        tensor = torch.tensor(
            list(image.getdata()),
            dtype=torch.float32
        ).view(image.size[1], image.size[0], 3).permute(2, 0, 1) / 255.0

        # Normalize
        for i in range(3):
            tensor[i] = (tensor[i] - self.mean[i]) / self.std[i]

        return tensor.to(self.dtype)

    def dynamic_preprocess(
        self,
        image: Image.Image,
        min_num: int = 2,
        max_num: int = 9,
        image_size: int = 640,
    ) -> Tuple[List[Image.Image], Tuple[int, int]]:
        """
        动态裁剪图片（复制自 deepseek_ocr.modeling_deepseekocr）
        """
        orig_width, orig_height = image.size
        aspect_ratio = orig_width / orig_height

        # Calculate target dimensions
        target_ratios = set(
            (i, j) for n in range(min_num, max_num + 1)
            for i in range(1, n + 1) for j in range(1, n + 1)
            if i * j <= max_num and i * j >= min_num
        )
        target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

        # Find best aspect ratio match
        target_aspect_ratio = self._find_closest_aspect_ratio(
            aspect_ratio, target_ratios, orig_width, orig_height, image_size
        )

        # Calculate resize dimensions
        target_width = image_size * target_aspect_ratio[0]
        target_height = image_size * target_aspect_ratio[1]
        blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

        # Resize image
        resized_img = image.resize((target_width, target_height))
        processed_images = []

        # Split into blocks
        for i in range(blocks):
            box = (
                (i % (target_width // image_size)) * image_size,
                (i // (target_width // image_size)) * image_size,
                ((i % (target_width // image_size)) + 1) * image_size,
                ((i // (target_width // image_size)) + 1) * image_size,
            )
            split_img = resized_img.crop(box)
            processed_images.append(split_img)

        return processed_images, target_aspect_ratio

    def _find_closest_aspect_ratio(
        self,
        aspect_ratio: float,
        target_ratios: List[Tuple[int, int]],
        width: int,
        height: int,
        image_size: int
    ) -> Tuple[int, int]:
        """
        找到最接近的宽高比
        """
        best_ratio_diff = float("inf")
        best_ratio = (1, 1)
        area = width * height

        for ratio in target_ratios:
            target_aspect_ratio = ratio[0] / ratio[1]
            ratio_diff = abs(aspect_ratio - target_aspect_ratio)
            if ratio_diff < best_ratio_diff:
                best_ratio_diff = ratio_diff
                best_ratio = ratio
            elif ratio_diff == best_ratio_diff:
                if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                    best_ratio = ratio

        return best_ratio

    def preprocess_image(self, image_path: str) -> Dict[str, Any]:
        """
        预处理单张图片

        Returns:
            包含预处理结果的字典
        """
        # 加载图片
        image = Image.open(image_path).convert("RGB")
        original_size = image.size

        images_list = []
        images_crop_list = []
        images_spatial_crop = []

        if self.crop_mode:
            # 确定裁剪比例
            if image.size[0] <= 640 and image.size[1] <= 640:
                crop_ratio = (1, 1)
                images_crop_raw = []
            else:
                images_crop_raw, crop_ratio = self.dynamic_preprocess(
                    image, min_num=2, max_num=9,
                    image_size=self.image_size
                )

            # 处理全局视图
            global_view = ImageOps.pad(
                image, (self.base_size, self.base_size),
                color=tuple(int(x * 255) for x in self.mean)
            )
            images_list.append(self.image_to_tensor(global_view))

            width_crop_num, height_crop_num = crop_ratio
            images_spatial_crop.append([width_crop_num, height_crop_num])

            # 处理裁剪视图
            if width_crop_num > 1 or height_crop_num > 1:
                for crop_img in images_crop_raw:
                    images_crop_list.append(self.image_to_tensor(crop_img))

            # 计算 image tokens
            num_queries = math.ceil((self.image_size // self.patch_size) / self.downsample_ratio)
            num_queries_base = math.ceil((self.base_size // self.patch_size) / self.downsample_ratio)

            tokenized_image = ([self.image_token_id] * num_queries_base + [self.image_token_id]) * num_queries_base
            tokenized_image += [self.image_token_id]

            if width_crop_num > 1 or height_crop_num > 1:
                tokenized_image += ([self.image_token_id] * (num_queries * width_crop_num) + [self.image_token_id]) * (
                    num_queries * height_crop_num)

        else:  # crop_mode = False
            crop_ratio = (1, 1)
            images_spatial_crop.append([1, 1])

            # 调整图片尺寸
            if self.base_size <= 640:
                resized_image = image.resize((self.base_size, self.base_size), Image.LANCZOS)
                images_list.append(self.image_to_tensor(resized_image))
            else:
                global_view = ImageOps.pad(
                    image, (self.base_size, self.base_size),
                    color=tuple(int(x * 255) for x in self.mean)
                )
                images_list.append(self.image_to_tensor(global_view))

            num_queries = math.ceil((self.base_size // self.patch_size) / self.downsample_ratio)
            tokenized_image = ([self.image_token_id] * num_queries + [self.image_token_id]) * num_queries
            tokenized_image += [self.image_token_id]

        # 准备最终数据
        images_ori = torch.stack(images_list, dim=0)
        images_spatial_crop_tensor = torch.tensor(images_spatial_crop, dtype=torch.long)

        if images_crop_list:
            images_crop = torch.stack(images_crop_list, dim=0)
        else:
            # 空占位符
            images_crop = torch.zeros((1, 3, self.base_size, self.base_size), dtype=self.dtype)

        return {
            'images_ori': images_ori,
            'images_crop': images_crop,
            'images_spatial_crop': images_spatial_crop_tensor,
            'tokenized_image': tokenized_image,
            'crop_ratio': crop_ratio,
            'original_size': original_size,
            'preprocessor_config': {
                'image_size': self.image_size,
                'base_size': self.base_size,
                'crop_mode': self.crop_mode,
            }
        }

    def save_cache(self, preprocessed_data: Dict[str, Any], cache_path: str):
        """
        保存预处理数据到缓存
        """
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        torch.save(preprocessed_data, cache_path)

    def load_cache(self, cache_path: str) -> Optional[Dict[str, Any]]:
        """
        从缓存加载预处理数据
        """
        if not os.path.exists(cache_path):
            return None

        try:
            return torch.load(cache_path, weights_only=False)
        except Exception as e:
            print(f"警告: 加载缓存失败 {cache_path}: {e}")
            return None

    def preprocess_and_cache(self, image_path: str, task_type: str) -> str:
        """
        预处理图片并缓存，返回缓存路径
        """
        cache_path = self.get_cache_path(image_path, task_type)

        # 如果缓存已存在，直接返回
        if os.path.exists(cache_path):
            return cache_path

        # 预处理并保存
        try:
            preprocessed = self.preprocess_image(image_path)
            self.save_cache(preprocessed, cache_path)
            return cache_path
        except Exception as e:
            print(f"错误: 预处理图片失败 {image_path}: {e}")
            return None


def batch_preprocess_images(
    image_paths: List[str],
    task_types: List[str],
    image_size: int = 640,
    base_size: int = 1024,
    crop_mode: bool = True,
    cache_dir: str = "ocr_data/preprocessed_cache",
    verbose: bool = True
) -> List[Optional[str]]:
    """
    批量预处理图片

    Args:
        image_paths: 图片路径列表
        task_types: 任务类型列表
        image_size: 图片尺寸
        base_size: 基础尺寸
        crop_mode: 是否使用裁剪模式
        cache_dir: 缓存目录
        verbose: 是否显示进度

    Returns:
        缓存路径列表
    """
    preprocessor = ImagePreprocessor(
        image_size=image_size,
        base_size=base_size,
        crop_mode=crop_mode,
        cache_dir=cache_dir
    )

    cache_paths = []
    total = len(image_paths)

    for i, (image_path, task_type) in enumerate(zip(image_paths, task_types)):
        if verbose and (i % 10 == 0 or i == total - 1):
            print(f"  预处理进度: {i+1}/{total} ({(i+1)/total*100:.1f}%)", end='\r')

        cache_path = preprocessor.preprocess_and_cache(image_path, task_type)
        cache_paths.append(cache_path)

    if verbose:
        print()  # 换行

    return cache_paths
