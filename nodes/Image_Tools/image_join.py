from __future__ import annotations

import math
import os
import re
from typing import Any, Dict, List, Optional, Tuple

import torch

from ..ComfyUI_Annotator_Preprocess.controlnet_preprocess import (
    _pick_first_image,
    _pil_to_tensor_image,
    _tensor_image_to_pil,
)

try:
    from PIL import Image, ImageColor, ImageDraw, ImageFont
except Exception:
    Image = None
    ImageColor = None
    ImageDraw = None
    ImageFont = None


class ContainsAnyImageTextDict(dict):
    """用于接收前端动态创建图片/文字输入端口的可选输入字典。"""

    def __contains__(self, key: object) -> bool:
        """仅允许以 image_ 或 text_ 开头的动态输入端口通过可选输入校验。"""

        if not isinstance(key, str):
            return False
        k = key.lower()
        return k.startswith("image_") or k.startswith("text_")

    def __getitem__(self, key: str) -> Tuple[str, Dict[str, Any]]:
        """为任意动态输入名提供默认类型，避免校验阶段KeyError。"""

        k = str(key).lower()
        if k.startswith("image_"):
            return ("IMAGE", {})
        if k.startswith("text_"):
            return ("STRING", {"default": ""})
        raise KeyError(key)

    def get(self, key: str, default: Any = None) -> Tuple[str, Dict[str, Any]]:
        """为任意动态输入名提供默认类型。"""

        k = str(key).lower()
        if k.startswith("image_"):
            return ("IMAGE", {})
        if k.startswith("text_"):
            return ("STRING", {"default": ""})
        return default


def _extract_index(name: str) -> int:
    """从 image_1 / text_1 这类名字中提取编号，无法提取则返回较大值用于排序。"""

    m = re.search(r"(\d+)$", str(name))
    if not m:
        return 10**9
    try:
        return int(m.group(1))
    except Exception:
        return 10**9


def _decode_escapes(s: str) -> str:
    """将用户输入的常见转义序列（如 \\n、\\t、\\r、\\\\）转换为真实字符。"""

    if not isinstance(s, str) or not s:
        return ""
    return (
        s.replace("\\\\", "\\")
        .replace("\\n", "\n")
        .replace("\\r", "\r")
        .replace("\\t", "\t")
        .replace("\\0", "\0")
    )


def _load_font(font_size: int) -> "ImageFont.ImageFont":
    """加载尽量兼容中文的字体，失败则回退到 PIL 默认字体。"""

    if ImageFont is None:
        raise RuntimeError("PIL 不可用，无法绘制文字")

    fs = max(6, int(font_size))
    win_dir = os.environ.get("WINDIR") or r"C:\Windows"
    candidates = [
        os.path.join(win_dir, "Fonts", "msyh.ttc"),
        os.path.join(win_dir, "Fonts", "msyh.ttf"),
        os.path.join(win_dir, "Fonts", "simhei.ttf"),
        os.path.join(win_dir, "Fonts", "simsun.ttc"),
        os.path.join(win_dir, "Fonts", "arial.ttf"),
    ]
    for p in candidates:
        try:
            if os.path.exists(p):
                return ImageFont.truetype(p, fs)
        except Exception:
            continue
    try:
        return ImageFont.load_default()
    except Exception:
        return ImageFont.truetype(candidates[-1], fs)


def _parse_color(v: Any, fallback: Tuple[int, int, int] = (0, 0, 0)) -> Tuple[int, int, int]:
    """解析颜色（支持 COLOR/#RRGGBB/rgb()/颜色名/(r,g,b)），解析失败返回 fallback。"""

    if ImageColor is None:
        return fallback
    if isinstance(v, (tuple, list)) and len(v) >= 3:
        try:
            r, g, b = v[0], v[1], v[2]
            if all(isinstance(x, (int, float)) for x in (r, g, b)):
                if all(0.0 <= float(x) <= 1.0 for x in (r, g, b)):
                    return (int(round(float(r) * 255.0)), int(round(float(g) * 255.0)), int(round(float(b) * 255.0)))
                return (int(round(float(r))), int(round(float(g))), int(round(float(b))))
        except Exception:
            return fallback
    try:
        c = ImageColor.getrgb(str(v))
        return (int(c[0]), int(c[1]), int(c[2]))
    except Exception:
        return fallback


def _wrap_text(text: str, draw: "ImageDraw.ImageDraw", font: "ImageFont.ImageFont", max_width: int) -> List[str]:
    """将文本按像素宽度自动换行，优先按单个字符换行以兼容中文。"""

    t = str(text or "")
    if not t:
        return []
    maxw = max(1, int(max_width))

    lines: List[str] = []
    for raw_line in t.splitlines():
        line = raw_line
        if line == "":
            lines.append("")
            continue
        buf = ""
        for ch in line:
            test = buf + ch
            w = draw.textlength(test, font=font) if hasattr(draw, "textlength") else draw.textbbox((0, 0), test, font=font)[2]
            if w <= maxw:
                buf = test
            else:
                if buf:
                    lines.append(buf)
                    buf = ch
                else:
                    lines.append(ch)
                    buf = ""
        if buf != "":
            lines.append(buf)
    return lines


def _resize_cover(pil: "Image.Image", target_size: Tuple[int, int]) -> "Image.Image":
    """将图片等比缩放并居中裁剪到目标尺寸，尽量减少留黑。"""

    if Image is None:
        raise RuntimeError("PIL 不可用，无法缩放图片")

    tw, th = int(target_size[0]), int(target_size[1])
    tw = max(1, tw)
    th = max(1, th)
    w, h = pil.size
    w = max(1, int(w))
    h = max(1, int(h))

    scale = max(float(tw) / float(w), float(th) / float(h))
    nw = max(1, int(round(float(w) * scale)))
    nh = max(1, int(round(float(h) * scale)))

    resample = getattr(getattr(Image, "Resampling", Image), "LANCZOS", getattr(Image, "LANCZOS", 1))
    resized = pil.resize((nw, nh), resample=resample)

    left = max(0, int(round((nw - tw) / 2.0)))
    top = max(0, int(round((nh - th) / 2.0)))
    right = min(nw, left + tw)
    bottom = min(nh, top + th)
    cropped = resized.crop((left, top, right, bottom))
    if cropped.size != (tw, th):
        cropped = cropped.resize((tw, th), resample=resample)
    return cropped


def _render_text_block(
    text: str,
    width: int,
    font: "ImageFont.ImageFont",
    padding: int,
    fg_rgb: Tuple[int, int, int],
) -> Optional["Image.Image"]:
    """渲染与图片等宽的文字块（RGBA，透明底），文本为空返回 None。"""

    t = str(text or "")
    if t.strip() == "":
        return None
    if Image is None or ImageDraw is None:
        raise RuntimeError("PIL 不可用，无法绘制文字")

    w = max(1, int(width))
    pad = max(0, int(padding))
    fg = (int(fg_rgb[0]), int(fg_rgb[1]), int(fg_rgb[2]), 255)

    tmp = Image.new("RGBA", (w, 10), (0, 0, 0, 0))
    draw = ImageDraw.Draw(tmp)
    lines = _wrap_text(t, draw, font, max_width=w - pad * 2)
    if not lines:
        return None

    line_h = draw.textbbox((0, 0), "A", font=font)[3]
    text_h = line_h * len(lines)
    h = max(1, text_h + pad * 2)

    canvas = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    draw2 = ImageDraw.Draw(canvas)
    y = pad
    for line in lines:
        lw = draw2.textlength(line, font=font) if hasattr(draw2, "textlength") else draw2.textbbox((0, 0), line, font=font)[2]
        x = int(max(pad, (w - lw) / 2.0))
        draw2.text((x, y), line, font=font, fill=fg)
        y += line_h
    return canvas


class CXYImageJoinNode:
    """将多张图片按选择的布局拼接，可为每张图片添加上/下方文字。"""

    CATEGORY = "CXY工具/图片处理"
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "run"

    @classmethod
    def INPUT_TYPES(cls) -> Dict[str, Any]:
        """定义节点输入，允许前端动态创建 image_N / text_N 输入端口。"""

        return {
            "required": {
                "image_1": ("IMAGE",),
                "text_1": ("STRING", {"default": ""}),
                "文字位置": (["文字在下方", "文字在上方"], {"default": "文字在下方"}),
                "排列": (["横向长条", "纵向长条", "正网格"], {"default": "正网格"}),
                "字号": ("INT", {"default": 32, "min": 8, "max": 256, "step": 1}),
                "文字颜色": ("COLOR", {"default": "#FFFFFF"}),
                "背景颜色": ("COLOR", {"default": "#000000"}),
                "内边距": ("INT", {"default": 40, "min": 0, "max": 500, "step": 1}),
                "文字内边距": ("INT", {"default": 10, "min": 0, "max": 200, "step": 1}),
            },
            "optional": ContainsAnyImageTextDict(),
        }

    @classmethod
    def VALIDATE_INPUTS(cls, input_types: Dict[str, Any]) -> bool:
        """跳过动态输入类型校验，由节点内部自行处理。"""

        return True

    def run(
        self,
        image_1: torch.Tensor,
        text_1: str = "",
        文字位置: str = "文字在下方",
        排列: str = "正网格",
        字号: int = 32,
        文字颜色: Any = "#FFFFFF",
        背景颜色: Any = "#000000",
        内边距: int = 40,
        文字内边距: int = 10,
        **kwargs: Any,
    ) -> Tuple[torch.Tensor]:
        """按布局将图片拼接为一张新图并输出。"""

        if Image is None or ImageDraw is None or ImageFont is None:
            raise RuntimeError("PIL 不可用，无法处理图片")

        font = _load_font(int(字号))
        fg_rgb = _parse_color(文字颜色, fallback=(255, 255, 255))
        bg_rgb = _parse_color(背景颜色, fallback=(0, 0, 0))
        margin = max(0, int(内边距))
        text_pad = max(0, int(文字内边距))

        pairs: List[Tuple[int, torch.Tensor, str]] = [(1, image_1, str(text_1 or ""))]
        for k, v in kwargs.items():
            if not isinstance(k, str):
                continue
            if not k.lower().startswith("image_"):
                continue
            idx = _extract_index(k)
            if idx <= 1:
                continue
            if not isinstance(v, torch.Tensor):
                continue
            t = kwargs.get(f"text_{idx}", "")
            pairs.append((idx, v, str(t or "")))

        pairs.sort(key=lambda x: x[0])
        src_pils: List[Tuple["Image.Image", str]] = []
        for _, ten, txt in pairs:
            pil = _tensor_image_to_pil(_pick_first_image(ten))
            src_pils.append((pil, txt))

        if not src_pils:
            raise ValueError("至少需要一张图片")

        target_w = max(p[0].size[0] for p in src_pils)
        target_h = max(p[0].size[1] for p in src_pils)
        target_w = max(1, int(target_w))
        target_h = max(1, int(target_h))

        base_images: List["Image.Image"] = []
        text_blocks: List[Optional["Image.Image"]] = []
        for pil, txt in src_pils:
            resized = _resize_cover(pil, (target_w, target_h)).convert("RGBA")
            base_images.append(resized)
            block = _render_text_block(txt, width=target_w, font=font, padding=text_pad, fg_rgb=fg_rgb)
            text_blocks.append(block)

        max_block_h = 0
        for b in text_blocks:
            if b is not None:
                max_block_h = max(max_block_h, int(b.size[1]))

        padded_blocks: List[Optional["Image.Image"]] = []
        if max_block_h > 0:
            for b in text_blocks:
                if b is None:
                    padded_blocks.append(Image.new("RGBA", (target_w, max_block_h), (0, 0, 0, 0)))
                    continue
                if b.size[1] == max_block_h:
                    padded_blocks.append(b)
                    continue
                tmp = Image.new("RGBA", (target_w, max_block_h), (0, 0, 0, 0))
                tmp.paste(b, (0, 0), b)
                padded_blocks.append(tmp)
        else:
            padded_blocks = [None for _ in text_blocks]

        tiles: List["Image.Image"] = []
        for img, blk in zip(base_images, padded_blocks):
            if blk is None:
                tiles.append(img)
                continue
            if str(文字位置) == "文字在上方":
                out = Image.new("RGBA", (target_w, max_block_h + target_h), (0, 0, 0, 0))
                out.paste(blk, (0, 0), blk)
                out.paste(img, (0, max_block_h), img)
                tiles.append(out)
            else:
                out = Image.new("RGBA", (target_w, target_h + max_block_h), (0, 0, 0, 0))
                out.paste(img, (0, 0), img)
                out.paste(blk, (0, target_h), blk)
                tiles.append(out)

        n = len(tiles)
        if 排列 == "横向长条":
            cols, rows = n, 1
        elif 排列 == "纵向长条":
            cols, rows = 1, n
        else:
            side = int(math.ceil(math.sqrt(n)))
            cols, rows = side, side

        cell_w = int(tiles[0].size[0])
        cell_h = int(tiles[0].size[1])
        layer = Image.new("RGBA", (cell_w * cols, cell_h * rows), (0, 0, 0, 0))

        for i, im in enumerate(tiles):
            r = i // cols
            c = i % cols
            if r >= rows:
                break
            x = c * cell_w
            y = r * cell_h
            layer.paste(im, (int(x), int(y)), im)

        bg = Image.new("RGBA", (layer.size[0] + margin * 2, layer.size[1] + margin * 2), (bg_rgb[0], bg_rgb[1], bg_rgb[2], 255))
        bg.alpha_composite(layer, (margin, margin))
        out_pil = bg.convert("RGB")
        out = _pil_to_tensor_image(out_pil).unsqueeze(0)
        return (out,)


NODE_CLASS_MAPPINGS = {
    "CXY_ImageJoin": CXYImageJoinNode,
}


NODE_DISPLAY_NAME_MAPPINGS = {
    "CXY_ImageJoin": "图片拼接（可选文字）",
}
