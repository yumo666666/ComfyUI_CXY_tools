import os
import shutil
import time
import uuid
import urllib.request
from pathlib import Path
from typing import Any, Dict, List, Tuple

import torch

try:
    import numpy as np
except Exception:
    np = None

try:
    from PIL import Image
except Exception:
    Image = None

try:
    import folder_paths
except Exception:
    folder_paths = None


def _pick_first_image(img: torch.Tensor) -> torch.Tensor:
    """从 ComfyUI IMAGE Tensor 中取第一张图，返回 [H,W,C]。"""

    if img.ndim == 4:
        return img[0]
    if img.ndim == 3:
        return img
    raise ValueError("IMAGE Tensor 形状必须为 [B,H,W,C] 或 [H,W,C]")


def _tensor_image_to_pil(image: torch.Tensor) -> "Image.Image":
    """将 ComfyUI IMAGE Tensor 转换为 PIL 图像（RGB）。"""

    if Image is None:
        raise RuntimeError("PIL 不可用，无法处理图片")
    if np is None:
        raise RuntimeError("numpy 不可用，无法处理图片")

    img = image.detach()
    if img.dtype != torch.float32 and img.dtype != torch.float16:
        img = img.float()
    img = torch.clamp(img, 0.0, 1.0)
    if img.ndim != 3 or img.shape[-1] not in (3, 4):
        raise ValueError("IMAGE Tensor 形状必须为 [H,W,C]")

    arr = (img.cpu().numpy() * 255.0).astype("uint8")
    if arr.shape[-1] == 4:
        pil = Image.fromarray(arr, mode="RGBA").convert("RGB")
    else:
        pil = Image.fromarray(arr, mode="RGB")
    return pil


def _pil_to_tensor_image(pil: "Image.Image") -> torch.Tensor:
    """将 PIL 图像转换为 ComfyUI IMAGE Tensor（[H,W,3]，float32，0~1）。"""

    if Image is None:
        raise RuntimeError("PIL 不可用，无法处理图片")
    if np is None:
        raise RuntimeError("numpy 不可用，无法处理图片")

    img = pil.convert("RGB")
    arr = np.asarray(img).astype("float32") / 255.0
    out = torch.from_numpy(arr)
    return out


def _resize_pil_to_max_side(pil: "Image.Image", max_side: int) -> "Image.Image":
    """将 PIL 图像按最大边长等比缩放到 max_side。"""

    if Image is None:
        raise RuntimeError("PIL 不可用，无法缩放图片")

    ms = int(max_side)
    if ms <= 0:
        return pil

    w, h = pil.size
    src_max = max(int(w), int(h))
    if src_max == ms:
        return pil

    scale = float(ms) / float(src_max)
    nw = max(1, int(round(float(w) * scale)))
    nh = max(1, int(round(float(h) * scale)))

    resample = getattr(getattr(Image, "Resampling", Image), "LANCZOS", getattr(Image, "LANCZOS", 1))
    return pil.resize((nw, nh), resample=resample)


def _resize_pil_to_size(pil: "Image.Image", size: Tuple[int, int]) -> "Image.Image":
    """将 PIL 图像缩放到指定尺寸（用于对齐 detector 内部固定输出分辨率）。"""

    if Image is None:
        raise RuntimeError("PIL 不可用，无法缩放图片")

    tw, th = int(size[0]), int(size[1])
    tw = max(1, tw)
    th = max(1, th)
    if pil.size == (tw, th):
        return pil

    resample = getattr(getattr(Image, "Resampling", Image), "LANCZOS", getattr(Image, "LANCZOS", 1))
    return pil.resize((tw, th), resample=resample)


def _call_detector(detector: Any, pil_in: "Image.Image", max_side: int) -> "Image.Image":
    """按 detector 支持的参数调用预处理器并返回输出 PIL 图像。"""

    ms = int(max_side)
    if ms <= 0:
        return detector(pil_in)

    try:
        import inspect

        sig = inspect.signature(detector)
        params = sig.parameters
    except Exception:
        return detector(pil_in)

    can_kwargs = any(p.kind == p.VAR_KEYWORD for p in params.values())
    kwargs: Dict[str, Any] = {}
    for k in (
        "detect_resolution",
        "image_resolution",
        "resolution",
        "input_resolution",
    ):
        if k in params or can_kwargs:
            kwargs[k] = ms
            break

    try:
        return detector(pil_in, **kwargs) if kwargs else detector(pil_in)
    except TypeError:
        return detector(pil_in)


def _get_plugin_root_dir() -> str:
    """获取插件根目录（custom_nodes/ComfyUI_CXY_tools）。"""

    return str(Path(__file__).resolve().parents[2])


def _get_plugin_models_dir() -> str:
    """获取本插件的 models 目录（固定放到 custom_nodes/ComfyUI_CXY_tools/models）。"""

    out_dir = os.path.join(_get_plugin_root_dir(), "models")
    os.makedirs(out_dir, exist_ok=True)
    return out_dir


def _get_annotator_models_dir() -> str:
    """获取 Annotator 权重缓存目录（固定放到本扩展 models/annotators 下）。"""

    base = _get_plugin_models_dir()
    out_dir = os.path.join(base, "annotators")
    os.makedirs(out_dir, exist_ok=True)
    return out_dir


def _get_annotator_models_legacy_dir() -> str:
    """返回旧版 Annotator 权重缓存目录（兼容历史路径）。"""

    base = _get_plugin_models_dir()
    return os.path.join(base, "annotators", "lllyasviel", "Annotators")


def _hf_file_url(filename: str, use_cn_mirror: bool) -> str:
    """构造 HuggingFace 文件下载链接（外网/国内镜像）。"""

    host = "https://hf-mirror.com" if use_cn_mirror else "https://huggingface.co"
    return f"{host}/lllyasviel/Annotators/resolve/main/{filename}"


def _format_bytes(n: int) -> str:
    """将字节数格式化为易读文本。"""

    v = float(max(0, int(n)))
    units = ["B", "KB", "MB", "GB", "TB"]
    idx = 0
    while v >= 1024.0 and idx < len(units) - 1:
        v /= 1024.0
        idx += 1
    if idx == 0:
        return f"{int(v)}{units[idx]}"
    return f"{v:.2f}{units[idx]}"


def _print_download_progress(name: str, downloaded: int, total: int | None, start_ts: float) -> None:
    """在命令行输出下载进度（单行刷新）。"""

    elapsed = max(0.001, float(time.time() - float(start_ts)))
    speed = float(downloaded) / elapsed
    if total and total > 0:
        pct = min(100.0, float(downloaded) * 100.0 / float(total))
        msg = (
            f"下载 {name}：{_format_bytes(downloaded)}/{_format_bytes(total)} "
            f"({pct:5.1f}%)  { _format_bytes(int(speed)) }/s"
        )
    else:
        msg = f"下载 {name}：{_format_bytes(downloaded)}  { _format_bytes(int(speed)) }/s"
    print("\r" + msg.ljust(90), end="", flush=True)


def _download_file(
    url: str,
    dst_path: str,
    timeout_s: int = 60,
    retries: int = 3,
    backoff_s: float = 2.0,
) -> None:
    """下载文件到指定路径（使用 .tmp 临时文件避免半成品，带重试）。"""

    os.makedirs(os.path.dirname(dst_path), exist_ok=True)
    tmp_path = f"{dst_path}.tmp"

    last_err: Exception | None = None
    for attempt in range(1, int(retries) + 1):
        try:
            try:
                if os.path.exists(tmp_path):
                    os.remove(tmp_path)
            except Exception:
                pass

            req = urllib.request.Request(
                url,
                method="GET",
                headers={"User-Agent": "ComfyUI_CXY_tools"},
            )
            with urllib.request.urlopen(req, timeout=timeout_s) as resp:
                total: int | None = None
                try:
                    hdr = resp.getheader("Content-Length")
                    total = int(hdr) if hdr else None
                except Exception:
                    total = None

                downloaded = 0
                start_ts = time.time()
                last_print_ts = 0.0
                name = os.path.basename(dst_path)
                with open(tmp_path, "wb") as f:
                    while True:
                        chunk = resp.read(1024 * 1024)
                        if not chunk:
                            break
                        f.write(chunk)
                        downloaded += len(chunk)
                        now = time.time()
                        if now - last_print_ts >= 0.2:
                            _print_download_progress(name, downloaded, total, start_ts)
                            last_print_ts = now
                _print_download_progress(name, downloaded, total, start_ts)
                print("", flush=True)
            os.replace(tmp_path, dst_path)
            return
        except Exception as e:
            last_err = e
            if attempt < int(retries):
                time.sleep(float(backoff_s) * (2 ** (attempt - 1)))
            continue
        finally:
            try:
                if os.path.exists(tmp_path) and not (os.path.exists(dst_path) and os.path.getsize(dst_path) > 0):
                    os.remove(tmp_path)
            except Exception:
                pass

    if last_err is not None:
        raise last_err


def _ensure_annotator_files(filenames: List[str], use_cn_mirror: bool) -> None:
    """确保指定权重文件已存在，不存在则按源下载。"""

    models_dir = _get_annotator_models_dir()
    legacy_dir = _get_annotator_models_legacy_dir()
    for fn in filenames:
        dst = os.path.join(models_dir, fn)
        if os.path.exists(dst) and os.path.getsize(dst) > 0:
            continue

        legacy = os.path.join(legacy_dir, fn)
        if os.path.exists(legacy) and os.path.getsize(legacy) > 0:
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            try:
                shutil.copy2(legacy, dst)
            except Exception:
                pass
            if os.path.exists(dst) and os.path.getsize(dst) > 0:
                continue

        url = _hf_file_url(fn, use_cn_mirror=use_cn_mirror)
        try:
            _download_file(url, dst_path=dst, timeout_s=180, retries=3, backoff_s=2.0)
        except Exception as e:
            raise RuntimeError(f"下载模型失败：{fn}\nURL：{url}\n保存到：{dst}\n原因：{e}")


def _temp_dir() -> str:
    """获取 ComfyUI 临时目录，获取失败则回退到插件 models/_temp。"""

    if folder_paths is not None and hasattr(folder_paths, "get_temp_directory"):
        try:
            d = folder_paths.get_temp_directory()
            if d and os.path.isdir(d):
                return d
        except Exception:
            pass
    d = os.path.join(_get_plugin_models_dir(), "_temp")
    os.makedirs(d, exist_ok=True)
    return d


def _save_tensor_image_to_temp(image: torch.Tensor, subfolder: str, prefix: str) -> Dict[str, str]:
    """保存一张 IMAGE 到 ComfyUI 临时目录，并返回 ComfyUI 前端可读取的元信息。"""

    if Image is None:
        raise RuntimeError("PIL 不可用，无法保存图片")

    temp_root = _temp_dir()
    safe_sub = str(subfolder or "").strip().replace("\\", "/").strip("/")
    out_dir = os.path.join(temp_root, safe_sub) if safe_sub else temp_root
    os.makedirs(out_dir, exist_ok=True)

    fname = f"{prefix}_{uuid.uuid4().hex}.png"
    out_path = os.path.join(out_dir, fname)
    pil = _tensor_image_to_pil(_pick_first_image(image))
    pil.save(out_path, format="PNG")
    return {
        "filename": fname,
        "subfolder": safe_sub,
        "type": "temp",
    }


class ControlnetPreprocessNode:
    """输入一张图，输出一张预处理图（深度/线稿/姿态）。"""

    CATEGORY = "CXY工具"
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "run"

    @classmethod
    def INPUT_TYPES(cls) -> Dict[str, Any]:
        """定义节点输入。"""

        return {
            "required": {
                "image": ("IMAGE",),
                "类型": (["深度图", "线稿图", "骨架姿势图"], {"default": "线稿图"}),
                "选项": (
                    [
                        "MiDaS",
                        "ZoeDepth",
                        "LeReS",
                        "线稿提取",
                        "软边缘",
                        "硬边缘",
                        "直线",
                        "OpenPose",
                    ],
                    {"default": "线稿提取"},
                ),
                "下载源": (["国内", "外网"], {"default": "国内"}),
                "最大边长": ("INT", {"default": 512, "min": 64, "max": 8192, "step": 1}),
            }
        }

    def run(self, image: torch.Tensor, 类型: str, 选项: str, 下载源: str, 最大边长: int) -> Tuple[torch.Tensor]:
        """执行预处理，自动下载缺失权重后输出处理图。"""

        if Image is None:
            raise RuntimeError("PIL 不可用，无法处理图片")

        use_cn_mirror = str(下载源) != "外网"

        try:
            from controlnet_aux import (
                CannyDetector,
                HEDdetector,
                LeresDetector,
                LineartDetector,
                MLSDdetector,
                MidasDetector,
                OpenposeDetector,
                ZoeDetector,
            )
        except Exception as e:
            raise RuntimeError(f"缺少依赖 controlnet-aux，无法运行该节点：{e}")

        type_s = str(类型)
        opt_s = str(选项)

        required_files: List[str] = []
        detector_kind: str = ""

        if type_s == "深度图":
            if opt_s == "MiDaS":
                detector_kind = "midas"
                required_files = ["dpt_hybrid-midas-501f0c75.pt"]
            elif opt_s == "ZoeDepth":
                detector_kind = "zoe"
                required_files = ["ZoeD_M12_N.pt"]
            else:
                detector_kind = "leres"
                required_files = ["res101.pth"]
        elif type_s == "线稿图":
            if opt_s == "线稿提取":
                detector_kind = "lineart"
                required_files = ["sk_model.pth", "sk_model2.pth"]
            elif opt_s == "软边缘":
                detector_kind = "hed"
                required_files = ["network-bsds500.pth"]
            elif opt_s == "直线":
                detector_kind = "mlsd"
                required_files = ["mlsd_large_512_fp32.pth"]
            else:
                detector_kind = "canny"
                required_files = []
        else:
            detector_kind = "openpose"
            required_files = ["body_pose_model.pth", "hand_pose_model.pth"]

        if required_files:
            _ensure_annotator_files(required_files, use_cn_mirror=use_cn_mirror)

        models_dir = _get_annotator_models_dir()

        def _patch_detector_download(detector_cls: Any) -> None:
            """把 controlnet_aux 内部的 hf_hub_download 替换为本地下载/读取。"""

            import sys

            mod = sys.modules.get(getattr(detector_cls, "__module__", ""))
            if mod is None:
                return
            if not hasattr(mod, "hf_hub_download"):
                return

            def _local_hf_hub_download(repo_id: str, filename: str, *args: Any, **kwargs: Any) -> str:
                base = os.path.basename(str(filename))
                local_path = os.path.join(models_dir, base)
                if not (os.path.exists(local_path) and os.path.getsize(local_path) > 0):
                    legacy_path = os.path.join(_get_annotator_models_legacy_dir(), base)
                    if os.path.exists(legacy_path) and os.path.getsize(legacy_path) > 0:
                        return legacy_path
                    _ensure_annotator_files([base], use_cn_mirror=use_cn_mirror)
                return local_path

            setattr(mod, "hf_hub_download", _local_hf_hub_download)

        if detector_kind == "midas":
            _patch_detector_download(MidasDetector)
            detector = MidasDetector.from_pretrained("lllyasviel/ControlNet")
        elif detector_kind == "zoe":
            _patch_detector_download(ZoeDetector)
            detector = ZoeDetector.from_pretrained("lllyasviel/ControlNet")
        elif detector_kind == "leres":
            _patch_detector_download(LeresDetector)
            detector = LeresDetector.from_pretrained("lllyasviel/ControlNet")
        elif detector_kind == "lineart":
            _patch_detector_download(LineartDetector)
            detector = LineartDetector.from_pretrained("lllyasviel/ControlNet")
        elif detector_kind == "hed":
            _patch_detector_download(HEDdetector)
            detector = HEDdetector.from_pretrained("lllyasviel/ControlNet")
        elif detector_kind == "mlsd":
            _patch_detector_download(MLSDdetector)
            detector = MLSDdetector.from_pretrained("lllyasviel/ControlNet")
        elif detector_kind == "openpose":
            _patch_detector_download(OpenposeDetector)
            detector = OpenposeDetector.from_pretrained("lllyasviel/ControlNet")
        else:
            detector = CannyDetector()

        img3 = _pick_first_image(image)
        pil_in = _tensor_image_to_pil(img3)
        pil_in = _resize_pil_to_max_side(pil_in, int(最大边长))
        out_pil = _call_detector(detector, pil_in, int(最大边长))
        out_pil = _resize_pil_to_size(out_pil, pil_in.size)
        out3 = _pil_to_tensor_image(out_pil)
        return (out3.unsqueeze(0),)


NODE_CLASS_MAPPINGS = {
    "ComfyUI_Annotator_Preprocess": ControlnetPreprocessNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ComfyUI_Annotator_Preprocess": "Controlnet图预处理",
}
