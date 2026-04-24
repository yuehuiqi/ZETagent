"""
图像预处理模块（img_processor.py）

注意：本文件命名为 img_processor 以避免与主办方提供的
codePro/utils/image_utils.py 产生命名冲突。

提供分辨率自适应压缩、图像质量优化等功能，
在保证视觉细节的同时最大限度地降低 Token 消耗。
"""

import io
import base64
import logging
from typing import Tuple
from PIL import Image

logger = logging.getLogger(__name__)

TARGET_MAX_WIDTH = 720
JPEG_QUALITY = 85


def preprocess_image(
    image: Image.Image,
    max_width: int = TARGET_MAX_WIDTH,
) -> Image.Image:
    """
    自适应分辨率压缩

    策略：
    1. 若图像宽度 > max_width，则等比例缩放到 max_width
    2. 保持宽高比不变，使用 LANCZOS 高质量插值
    3. 确保输出为 RGB 模式（PNG 有时带 alpha 通道导致 base64 增大）

    Args:
        image: 原始 PIL Image
        max_width: 允许的最大宽度（px）

    Returns:
        处理后的 PIL Image
    """
    original_size = image.size
    result = image

    if image.width > max_width:
        ratio = max_width / image.width
        new_size = (max_width, int(image.height * ratio))
        result = image.resize(new_size, Image.LANCZOS)
        logger.debug(
            f"[ImagePreprocessor] Resized {original_size} -> {result.size}"
        )

    if result.mode != "RGB":
        result = result.convert("RGB")

    return result


def encode_image(
    image: Image.Image,
    image_format: str = "PNG",
    jpeg_quality: int = JPEG_QUALITY,
) -> str:
    """
    将图像编码为 OpenAI image_url 格式的 base64 字符串

    Args:
        image: PIL Image
        image_format: 'PNG' 或 'JPEG'
        jpeg_quality: JPEG 质量 (1-95)

    Returns:
        data:image/xxx;base64,<base64str>
    """
    buffered = io.BytesIO()
    if image_format.upper() == "JPEG":
        image.save(buffered, format="JPEG", quality=jpeg_quality)
    else:
        image.save(buffered, format="PNG")
    b64 = base64.b64encode(buffered.getvalue()).decode("utf-8")
    mime = "jpeg" if image_format.upper() == "JPEG" else "png"
    return f"data:image/{mime};base64,{b64}"


def estimate_token_cost(image: Image.Image) -> int:
    """粗略估算图像编码后的 Token 消耗（约每750像素1 token）"""
    pixels = image.width * image.height
    return max(1, int(pixels / 750))
