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


def _hf_repo_file_url(repo_id: str, filename: str, use_cn_mirror: bool) -> str:
    host = "https://hf-mirror.com" if use_cn_mirror else "https://huggingface.co"
    rid = str(repo_id).strip().strip("/")
    fn = str(filename).strip().lstrip("/")
    return f"{host}/{rid}/resolve/main/{fn}"


def _ensure_hf_repo_files(repo_id: str, filenames: List[str], use_cn_mirror: bool) -> List[str]:
    models_dir = _get_annotator_models_dir()
    out_paths: List[str] = []
    for fn in filenames:
        base = os.path.basename(str(fn))
        dst = os.path.join(models_dir, base)
        if os.path.exists(dst) and os.path.getsize(dst) > 0:
            out_paths.append(dst)
            continue
        url = _hf_repo_file_url(repo_id=repo_id, filename=base, use_cn_mirror=use_cn_mirror)
        try:
            _download_file(url, dst_path=dst, timeout_s=180, retries=3, backoff_s=2.0)
        except Exception as e:
            raise RuntimeError(f"下载模型失败：{base}\nURL：{url}\n保存到：{dst}\n原因：{e}")
        out_paths.append(dst)
    return out_paths


def _dwpose_guess_onnx_input_shape_dtype(filename: str) -> Tuple[Tuple[int, int], Any]:
    if np is None:
        raise RuntimeError("numpy 不可用，无法运行 DWPose")
    dtype = np.float32
    fn = str(filename).lower()
    if "fp16" in fn:
        dtype = np.float16
    elif "int8" in fn:
        dtype = np.uint8
    input_size = (640, 640) if "yolo" in fn else (192, 256)
    if "384" in fn:
        input_size = (288, 384)
    elif "256" in fn:
        input_size = (256, 256)
    return input_size, dtype


def _dwpose_get_ort_providers() -> List[str]:
    providers = []
    try:
        import onnxruntime as ort

        for p in (
            "CUDAExecutionProvider",
            "DirectMLExecutionProvider",
            "OpenVINOExecutionProvider",
            "ROCMExecutionProvider",
            "CPUExecutionProvider",
        ):
            if p in ort.get_available_providers():
                providers.append(p)
    except Exception:
        return []
    return providers


def _dwpose_is_torchscript(model: Any) -> bool:
    return bool(type(model).__name__ == "RecursiveScriptModule")


def _dwpose_get_model_type(filename: str | None) -> str | None:
    if filename is None:
        return None
    fn = str(filename).lower()
    ort_providers = _dwpose_get_ort_providers()
    if fn.endswith(".onnx") and ort_providers:
        return "ort"
    if fn.endswith(".onnx"):
        return "cv2"
    return "torchscript"


def _dwpose_det_nms(boxes: Any, scores: Any, nms_thr: float) -> List[int]:
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]
    keep: List[int] = []
    while order.size > 0:
        i = int(order[0])
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        ovr = inter / (areas[i] + areas[order[1:]] - inter)
        inds = np.where(ovr <= nms_thr)[0]
        order = order[inds + 1]
    return keep


def _dwpose_det_multiclass_nms(boxes: Any, scores: Any, nms_thr: float, score_thr: float) -> Any:
    final_dets = []
    num_classes = scores.shape[1]
    for cls_ind in range(num_classes):
        cls_scores = scores[:, cls_ind]
        valid_score_mask = cls_scores > score_thr
        if valid_score_mask.sum() == 0:
            continue
        valid_scores = cls_scores[valid_score_mask]
        valid_boxes = boxes[valid_score_mask]
        keep = _dwpose_det_nms(valid_boxes, valid_scores, nms_thr)
        if len(keep) > 0:
            cls_inds = np.ones((len(keep), 1)) * float(cls_ind)
            dets = np.concatenate([valid_boxes[keep], valid_scores[keep, None], cls_inds], 1)
            final_dets.append(dets)
    if len(final_dets) == 0:
        return None
    return np.concatenate(final_dets, 0)


def _dwpose_det_demo_postprocess(outputs: Any, img_size: Tuple[int, int], p6: bool = False) -> Any:
    grids = []
    expanded_strides = []
    strides = [8, 16, 32] if not p6 else [8, 16, 32, 64]
    hsizes = [img_size[0] // stride for stride in strides]
    wsizes = [img_size[1] // stride for stride in strides]
    for hsize, wsize, stride in zip(hsizes, wsizes, strides):
        xv, yv = np.meshgrid(np.arange(wsize), np.arange(hsize))
        grid = np.stack((xv, yv), 2).reshape(1, -1, 2)
        grids.append(grid)
        shape = grid.shape[:2]
        expanded_strides.append(np.full((*shape, 1), stride))
    grids = np.concatenate(grids, 1)
    expanded_strides = np.concatenate(expanded_strides, 1)
    outputs[..., :2] = (outputs[..., :2] + grids) * expanded_strides
    outputs[..., 2:4] = np.exp(outputs[..., 2:4]) * expanded_strides
    return outputs


def _dwpose_det_preprocess(ori_img: Any, input_size: Tuple[int, int], swap: Tuple[int, int, int] = (2, 0, 1)) -> Tuple[Any, float]:
    import cv2

    img = ori_img
    if len(img.shape) == 3:
        padded_img = np.ones((input_size[0], input_size[1], 3), dtype=np.uint8) * 114
    else:
        padded_img = np.ones(input_size, dtype=np.uint8) * 114
    r = min(input_size[0] / img.shape[0], input_size[1] / img.shape[1])
    resized_img = cv2.resize(
        img,
        (int(img.shape[1] * r), int(img.shape[0] * r)),
        interpolation=cv2.INTER_LINEAR,
    ).astype(np.uint8)
    padded_img[: int(img.shape[0] * r), : int(img.shape[1] * r)] = resized_img
    padded_img = padded_img.transpose(swap)
    padded_img = np.ascontiguousarray(padded_img, dtype=np.float32)
    return padded_img, r


def _dwpose_inference_detector(session: Any, ori_img: Any, detect_classes: List[int] | None = None, dtype: Any = None) -> Any:
    import cv2

    if detect_classes is None:
        detect_classes = [0]
    if dtype is None:
        dtype = np.float32
    input_shape = (640, 640)
    img, ratio = _dwpose_det_preprocess(ori_img, input_shape)
    input_blob = img[None, :, :, :].astype(dtype)
    if "InferenceSession" in type(session).__name__:
        input_name = session.get_inputs()[0].name
        output = session.run(None, {input_name: input_blob})
    else:
        out_names = session.getUnconnectedOutLayersNames()
        session.setInput(input_blob)
        output = session.forward(out_names)
    predictions = _dwpose_det_demo_postprocess(output[0], input_shape)[0]
    boxes = predictions[:, :4]
    scores = predictions[:, 4:5] * predictions[:, 5:]
    boxes_xyxy = np.ones_like(boxes)
    boxes_xyxy[:, 0] = boxes[:, 0] - boxes[:, 2] / 2.0
    boxes_xyxy[:, 1] = boxes[:, 1] - boxes[:, 3] / 2.0
    boxes_xyxy[:, 2] = boxes[:, 0] + boxes[:, 2] / 2.0
    boxes_xyxy[:, 3] = boxes[:, 1] + boxes[:, 3] / 2.0
    boxes_xyxy /= ratio
    dets = _dwpose_det_multiclass_nms(boxes_xyxy, scores, nms_thr=0.45, score_thr=0.1)
    if dets is None:
        return None
    final_boxes, final_scores, final_cls_inds = dets[:, :4], dets[:, 4], dets[:, 5]
    isscore = final_scores > 0.3
    iscat = np.isin(final_cls_inds, detect_classes)
    isbbox = [bool(i and j) for (i, j) in zip(isscore, iscat)]
    final_boxes = final_boxes[isbbox]
    return final_boxes


def _dwpose_pose_bbox_xyxy2cs(bbox: Any, padding: float = 1.0) -> Tuple[Any, Any]:
    dim = bbox.ndim
    if dim == 1:
        bbox = bbox[None, :]
    x1, y1, x2, y2 = np.hsplit(bbox, [1, 2, 3])
    center = np.hstack([x1 + x2, y1 + y2]) * 0.5
    scale = np.hstack([x2 - x1, y2 - y1]) * padding
    if dim == 1:
        center = center[0]
        scale = scale[0]
    return center, scale


def _dwpose_pose_fix_aspect_ratio(bbox_scale: Any, aspect_ratio: float) -> Any:
    w, h = np.hsplit(bbox_scale, [1])
    bbox_scale = np.where(
        w > h * aspect_ratio,
        np.hstack([w, w / aspect_ratio]),
        np.hstack([h * aspect_ratio, h]),
    )
    return bbox_scale


def _dwpose_pose_rotate_point(pt: Any, angle_rad: float) -> Any:
    sn, cs = np.sin(angle_rad), np.cos(angle_rad)
    rot_mat = np.array([[cs, -sn], [sn, cs]])
    return rot_mat @ pt


def _dwpose_pose_get_3rd_point(a: Any, b: Any) -> Any:
    direction = a - b
    c = b + np.r_[-direction[1], direction[0]]
    return c


def _dwpose_pose_get_warp_matrix(
    center: Any,
    scale: Any,
    rot: float,
    output_size: Tuple[int, int],
    shift: Tuple[float, float] = (0.0, 0.0),
    inv: bool = False,
) -> Any:
    import cv2

    shift_arr = np.array(shift)
    scale_tmp = scale
    rot_rad = np.pi * rot / 180
    src_w = scale_tmp[0]
    dst_w, dst_h = output_size
    src_dir = _dwpose_pose_rotate_point(np.array([0.0, src_w * -0.5]), rot_rad)
    dst_dir = np.array([0.0, dst_w * -0.5])
    src = np.zeros((3, 2), dtype=np.float32)
    dst = np.zeros((3, 2), dtype=np.float32)
    src[0, :] = center + scale_tmp * shift_arr
    src[1, :] = center + src_dir + scale_tmp * shift_arr
    dst[0, :] = [dst_w * 0.5, dst_h * 0.5]
    dst[1, :] = np.array([dst_w * 0.5, dst_h * 0.5]) + dst_dir
    src[2:, :] = _dwpose_pose_get_3rd_point(src[0, :], src[1, :])
    dst[2:, :] = _dwpose_pose_get_3rd_point(dst[0, :], dst[1, :])
    if inv:
        return cv2.getAffineTransform(np.float32(dst), np.float32(src))
    return cv2.getAffineTransform(np.float32(src), np.float32(dst))


def _dwpose_pose_top_down_affine(input_size: Tuple[int, int], bbox_scale: Any, bbox_center: Any, img: Any) -> Tuple[Any, Any]:
    import cv2

    aspect_ratio = float(input_size[0]) / float(input_size[1])
    bbox_scale = _dwpose_pose_fix_aspect_ratio(bbox_scale, aspect_ratio)
    warp_mat = _dwpose_pose_get_warp_matrix(bbox_center, bbox_scale, 0.0, input_size)
    resized_img = cv2.warpAffine(img, warp_mat, input_size, flags=cv2.INTER_LINEAR)
    return resized_img, bbox_scale


def _dwpose_pose_preprocess(img: Any, out_bbox: Any, input_size: Tuple[int, int] = (192, 256)) -> Tuple[List[Any], List[Any], List[Any]]:
    img_shape = img.shape[:2]
    out_img: List[Any] = []
    out_center: List[Any] = []
    out_scale: List[Any] = []
    if len(out_bbox) == 0:
        out_bbox = [[0, 0, img_shape[1], img_shape[0]]]
    for i in range(len(out_bbox)):
        x0 = out_bbox[i][0]
        y0 = out_bbox[i][1]
        x1 = out_bbox[i][2]
        y1 = out_bbox[i][3]
        bbox = np.array([x0, y0, x1, y1])
        center, scale = _dwpose_pose_bbox_xyxy2cs(bbox, padding=1.25)
        resized_img, scale = _dwpose_pose_top_down_affine(input_size, scale, center, img)
        mean = np.array([123.675, 116.28, 103.53])
        std = np.array([58.395, 57.12, 57.375])
        resized_img = (resized_img - mean) / std
        out_img.append(resized_img)
        out_center.append(center)
        out_scale.append(scale)
    return out_img, out_center, out_scale


def _dwpose_pose_inference(sess: Any, img_list: List[Any], dtype: Any = None) -> List[Any]:
    if dtype is None:
        dtype = np.float32
    all_out: List[Any] = []
    input_blob = np.stack(img_list, axis=0).transpose(0, 3, 1, 2).astype(dtype)
    if "InferenceSession" in type(sess).__name__:
        input_name = sess.get_inputs()[0].name
        all_outputs = sess.run(None, {input_name: input_blob})
        for batch_idx in range(len(all_outputs[0])):
            outputs = [all_outputs[i][batch_idx : batch_idx + 1, ...] for i in range(len(all_outputs))]
            all_out.append(outputs)
        return all_out
    for i in range(len(img_list)):
        input_one = img_list[i].transpose(2, 0, 1)[None, :, :, :]
        out_names = sess.getUnconnectedOutLayersNames()
        sess.setInput(input_one)
        outputs = sess.forward(out_names)
        all_out.append(outputs)
    return all_out


def _dwpose_pose_get_simcc_maximum(simcc_x: Any, simcc_y: Any) -> Tuple[Any, Any]:
    if simcc_x.ndim == 2:
        simcc_x = simcc_x[None, ...]
    if simcc_y.ndim == 2:
        simcc_y = simcc_y[None, ...]

    if simcc_x.ndim != 3 or simcc_y.ndim != 3:
        raise ValueError(f"SimCC 输出形状不符合预期：simcc_x={getattr(simcc_x, 'shape', None)}, simcc_y={getattr(simcc_y, 'shape', None)}")

    if int(simcc_x.shape[1]) > int(simcc_x.shape[2]):
        simcc_x = np.swapaxes(simcc_x, 1, 2)
    if int(simcc_y.shape[1]) > int(simcc_y.shape[2]):
        simcc_y = np.swapaxes(simcc_y, 1, 2)

    x_locs = np.argmax(simcc_x, axis=-1)
    y_locs = np.argmax(simcc_y, axis=-1)
    max_val_x = np.amax(simcc_x, axis=-1)
    max_val_y = np.amax(simcc_y, axis=-1)
    max_val = np.minimum(max_val_x, max_val_y)
    mask = max_val > 0
    x_locs = x_locs * mask
    y_locs = y_locs * mask
    return np.stack((x_locs, y_locs), axis=-1).astype(np.float32), max_val


def _dwpose_pose_decode(simcc_x: Any, simcc_y: Any, simcc_split_ratio: float = 2.0) -> Tuple[Any, Any]:
    keypoints, scores = _dwpose_pose_get_simcc_maximum(simcc_x, simcc_y)
    keypoints = keypoints / float(simcc_split_ratio)
    return keypoints, scores


def _dwpose_pose_postprocess(
    outputs: List[Any],
    model_input_size: Tuple[int, int],
    center: List[Any],
    scale: List[Any],
    simcc_split_ratio: float = 2.0,
) -> Tuple[Any, Any]:
    all_key = []
    all_score = []
    for i in range(len(outputs)):
        simcc_x, simcc_y = outputs[i]
        keypoints, scores = _dwpose_pose_decode(simcc_x, simcc_y, simcc_split_ratio)
        keypoints = keypoints / model_input_size * scale[i] + center[i] - scale[i] / 2
        all_key.append(keypoints[0])
        all_score.append(scores[0])
    return np.array(all_key), np.array(all_score)


def _dwpose_inference_pose(session: Any, out_bbox: Any, ori_img: Any, model_input_size: Tuple[int, int] = (288, 384), dtype: Any = None) -> Tuple[Any, Any]:
    if dtype is None:
        dtype = np.float32
    resized_img, center, scale = _dwpose_pose_preprocess(ori_img, out_bbox, model_input_size)
    outputs = _dwpose_pose_inference(session, resized_img, dtype)
    keypoints, scores = _dwpose_pose_postprocess(outputs, model_input_size, center, scale)
    return keypoints, scores


class _DWPoseKeypoint(tuple):
    __slots__ = ()

    def __new__(cls, x: float, y: float, score: float = 1.0, kid: int = -1):
        return tuple.__new__(cls, (float(x), float(y), float(score), int(kid)))

    @property
    def x(self) -> float:
        return float(self[0])

    @property
    def y(self) -> float:
        return float(self[1])

    @property
    def score(self) -> float:
        return float(self[2])

    @property
    def id(self) -> int:
        return int(self[3])


class _DWPoseBodyResult(tuple):
    __slots__ = ()

    def __new__(cls, keypoints: List[Any], total_score: float = 0.0, total_parts: int = 0):
        return tuple.__new__(cls, (keypoints, float(total_score), int(total_parts)))

    @property
    def keypoints(self) -> List[Any]:
        return self[0]


class _DWPoseResult(tuple):
    __slots__ = ()

    def __new__(cls, body: Any, left_hand: Any, right_hand: Any, face: Any):
        return tuple.__new__(cls, (body, left_hand, right_hand, face))

    @property
    def body(self) -> Any:
        return self[0]

    @property
    def left_hand(self) -> Any:
        return self[1]

    @property
    def right_hand(self) -> Any:
        return self[2]

    @property
    def face(self) -> Any:
        return self[3]


def _dwpose_util_is_normalized(keypoints: List[Any]) -> bool:
    point_normalized = [0 <= abs(k.x) <= 1 and 0 <= abs(k.y) <= 1 for k in keypoints if k is not None]
    if not point_normalized:
        return False
    return all(point_normalized)


def _dwpose_util_draw_bodypose(canvas: Any, keypoints: List[Any], xinsr_stick_scaling: bool = False) -> Any:
    import cv2
    import math

    if not _dwpose_util_is_normalized(keypoints):
        H, W = 1.0, 1.0
    else:
        H, W, _ = canvas.shape
    CH, CW, _ = canvas.shape
    stickwidth = 4
    max_side = max(CW, CH)
    if xinsr_stick_scaling:
        stick_scale = 1 if max_side < 500 else min(2 + (max_side // 1000), 7)
    else:
        stick_scale = 1
    limb_seq = [
        [2, 3],
        [2, 6],
        [3, 4],
        [4, 5],
        [6, 7],
        [7, 8],
        [2, 9],
        [9, 10],
        [10, 11],
        [2, 12],
        [12, 13],
        [13, 14],
        [2, 1],
        [1, 15],
        [15, 17],
        [1, 16],
        [16, 18],
    ]
    colors = [
        [255, 0, 0],
        [255, 85, 0],
        [255, 170, 0],
        [255, 255, 0],
        [170, 255, 0],
        [85, 255, 0],
        [0, 255, 0],
        [0, 255, 85],
        [0, 255, 170],
        [0, 255, 255],
        [0, 170, 255],
        [0, 85, 255],
        [0, 0, 255],
        [85, 0, 255],
        [170, 0, 255],
        [255, 0, 255],
        [255, 0, 170],
        [255, 0, 85],
    ]
    for (k1_index, k2_index), color in zip(limb_seq, colors):
        keypoint1 = keypoints[k1_index - 1]
        keypoint2 = keypoints[k2_index - 1]
        if keypoint1 is None or keypoint2 is None:
            continue
        Y = np.array([keypoint1.x, keypoint2.x]) * float(W)
        X = np.array([keypoint1.y, keypoint2.y]) * float(H)
        mX = np.mean(X)
        mY = np.mean(Y)
        length = ((X[0] - X[1]) ** 2 + (Y[0] - Y[1]) ** 2) ** 0.5
        angle = math.degrees(math.atan2(X[0] - X[1], Y[0] - Y[1]))
        polygon = cv2.ellipse2Poly((int(mY), int(mX)), (int(length / 2), stickwidth * stick_scale), int(angle), 0, 360, 1)
        cv2.fillConvexPoly(canvas, polygon, [int(float(c) * 0.6) for c in color])
    for keypoint, color in zip(keypoints, colors):
        if keypoint is None:
            continue
        x, y = keypoint.x, keypoint.y
        x = int(x * W)
        y = int(y * H)
        cv2.circle(canvas, (int(x), int(y)), 4, color, thickness=-1)
    return canvas


def _dwpose_util_draw_handpose(canvas: Any, keypoints: Any) -> Any:
    import cv2

    def hsv_to_rgb255(h: float, s: float, v: float) -> Tuple[int, int, int]:
        hh = float(h) % 1.0
        ss = max(0.0, min(1.0, float(s)))
        vv = max(0.0, min(1.0, float(v)))
        i = int(hh * 6.0)
        f = (hh * 6.0) - float(i)
        p = vv * (1.0 - ss)
        q = vv * (1.0 - f * ss)
        t = vv * (1.0 - (1.0 - f) * ss)
        i = i % 6
        if i == 0:
            r, g, b = vv, t, p
        elif i == 1:
            r, g, b = q, vv, p
        elif i == 2:
            r, g, b = p, vv, t
        elif i == 3:
            r, g, b = p, q, vv
        elif i == 4:
            r, g, b = t, p, vv
        else:
            r, g, b = vv, p, q
        return int(r * 255.0), int(g * 255.0), int(b * 255.0)

    if not keypoints:
        return canvas
    if not _dwpose_util_is_normalized(keypoints):
        H, W = 1.0, 1.0
    else:
        H, W, _ = canvas.shape
    edges = [
        [0, 1],
        [1, 2],
        [2, 3],
        [3, 4],
        [0, 5],
        [5, 6],
        [6, 7],
        [7, 8],
        [0, 9],
        [9, 10],
        [10, 11],
        [11, 12],
        [0, 13],
        [13, 14],
        [14, 15],
        [15, 16],
        [0, 17],
        [17, 18],
        [18, 19],
        [19, 20],
    ]
    for ie, (e1, e2) in enumerate(edges):
        k1 = keypoints[e1]
        k2 = keypoints[e2]
        if k1 is None or k2 is None:
            continue
        x1 = int(k1.x * W)
        y1 = int(k1.y * H)
        x2 = int(k2.x * W)
        y2 = int(k2.y * H)
        if x1 > 0.01 and y1 > 0.01 and x2 > 0.01 and y2 > 0.01:
            r, g, b = hsv_to_rgb255(ie / float(len(edges)), 1.0, 1.0)
            cv2.line(
                canvas,
                (x1, y1),
                (x2, y2),
                (b, g, r),
                thickness=2,
            )
    for keypoint in keypoints:
        if keypoint is None:
            continue
        x, y = keypoint.x, keypoint.y
        x = int(x * W)
        y = int(y * H)
        if x > 0.01 and y > 0.01:
            cv2.circle(canvas, (x, y), 4, (0, 0, 255), thickness=-1)
    return canvas


def _dwpose_util_draw_facepose(canvas: Any, keypoints: Any) -> Any:
    import cv2

    if not keypoints:
        return canvas
    if not _dwpose_util_is_normalized(keypoints):
        H, W = 1.0, 1.0
    else:
        H, W, _ = canvas.shape
    for keypoint in keypoints:
        if keypoint is None:
            continue
        x, y = keypoint.x, keypoint.y
        x = int(x * W)
        y = int(y * H)
        if x > 0.01 and y > 0.01:
            cv2.circle(canvas, (x, y), 3, (255, 255, 255), thickness=-1)
    return canvas


def _dwpose_draw_poses(poses: List[Any], H: int, W: int, draw_body: bool = True, draw_hand: bool = True, draw_face: bool = True, xinsr_stick_scaling: bool = False) -> Any:
    canvas = np.zeros(shape=(int(H), int(W), 3), dtype=np.uint8)
    for pose in poses:
        if draw_body:
            canvas = _dwpose_util_draw_bodypose(canvas, pose.body.keypoints, xinsr_stick_scaling)
        if draw_hand:
            canvas = _dwpose_util_draw_handpose(canvas, pose.left_hand)
            canvas = _dwpose_util_draw_handpose(canvas, pose.right_hand)
        if draw_face:
            canvas = _dwpose_util_draw_facepose(canvas, pose.face)
    return canvas


def _dwpose_hwc3(x: Any) -> Any:
    if x.ndim == 2:
        x = x[:, :, None]
    H, W, C = x.shape
    if C == 3:
        return x
    if C == 1:
        return np.concatenate([x, x, x], axis=2)
    if C == 4:
        color = x[:, :, 0:3].astype(np.float32)
        alpha = x[:, :, 3:4].astype(np.float32) / 255.0
        y = color * alpha + 255.0 * (1.0 - alpha)
        y = y.clip(0, 255).astype(np.uint8)
        return y
    raise ValueError("不支持的通道数")


def _dwpose_pad64(x: int) -> int:
    return int(np.ceil(float(x) / 64.0) * 64 - x)


def _dwpose_resize_image_with_pad(input_image: Any, resolution: int, upscale_method: str = "", skip_hwc3: bool = False, mode: str = "edge") -> Tuple[Any, Any]:
    import cv2

    img = input_image if skip_hwc3 else _dwpose_hwc3(input_image)
    H_raw, W_raw, _ = img.shape
    if int(resolution) == 0:
        return img, (lambda x: x)
    k = float(resolution) / float(min(H_raw, W_raw))
    H_target = int(np.round(float(H_raw) * k))
    W_target = int(np.round(float(W_raw) * k))
    interp = cv2.INTER_AREA
    if k > 1:
        interp = getattr(cv2, upscale_method, cv2.INTER_CUBIC) if upscale_method else cv2.INTER_CUBIC
    img = cv2.resize(img, (W_target, H_target), interpolation=interp)
    H_pad, W_pad = _dwpose_pad64(H_target), _dwpose_pad64(W_target)
    img_padded = np.pad(img, [[0, H_pad], [0, W_pad], [0, 0]], mode=mode)

    def remove_pad(x: Any) -> Any:
        return np.ascontiguousarray(x[:H_target, :W_target, ...].copy())

    return np.ascontiguousarray(img_padded.copy()), remove_pad


def _dwpose_common_input_validate(input_image: Any, output_type: str | None, **kwargs: Any) -> Tuple[Any, str]:
    if input_image is None:
        raise ValueError("input_image must be defined")
    if not isinstance(input_image, np.ndarray):
        input_image = np.array(input_image, dtype=np.uint8)
        output_type = output_type or "pil"
    else:
        output_type = output_type or "np"
    return input_image, output_type


def _dwpose_format_result(keypoints_info: Any) -> List[Any]:
    def format_keypoint_part(part: Any) -> Any:
        keypoints = [(_DWPoseKeypoint(x, y, score, i) if score >= 0.3 else None) for i, (x, y, score) in enumerate(part)]
        if all(k is None for k in keypoints):
            return None
        return keypoints

    def total_score(keypoints: Any) -> float:
        if keypoints is None:
            return 0.0
        return float(sum(float(k.score) for k in keypoints if k is not None))

    pose_results: List[Any] = []
    if keypoints_info is None:
        return pose_results
    for instance in keypoints_info:
        body_keypoints = format_keypoint_part(instance[:18]) or ([None] * 18)
        left_hand = format_keypoint_part(instance[92:113])
        right_hand = format_keypoint_part(instance[113:134])
        face = format_keypoint_part(instance[24:92])
        if face is not None:
            face.append(body_keypoints[14])
            face.append(body_keypoints[15])
        body = _DWPoseBodyResult(body_keypoints, total_score(body_keypoints), len(body_keypoints))
        pose_results.append(_DWPoseResult(body, left_hand, right_hand, face))
    return pose_results


class _CXYDWPoseWholebody:
    def __init__(self, det_model_path: str | None, pose_model_path: str | None):
        self.det_filename = det_model_path and os.path.basename(det_model_path)
        self.pose_filename = pose_model_path and os.path.basename(pose_model_path)
        self.det = None
        self.pose = None
        if det_model_path:
            det_type = _dwpose_get_model_type(self.det_filename)
            if det_type == "ort":
                import onnxruntime as ort

                self.det = ort.InferenceSession(det_model_path, providers=_dwpose_get_ort_providers() or ["CPUExecutionProvider"])
            elif det_type == "cv2":
                import cv2

                self.det = cv2.dnn.readNetFromONNX(det_model_path)
            else:
                self.det = torch.jit.load(det_model_path)
        if pose_model_path:
            pose_type = _dwpose_get_model_type(self.pose_filename)
            if pose_type == "ort":
                import onnxruntime as ort

                self.pose = ort.InferenceSession(pose_model_path, providers=_dwpose_get_ort_providers() or ["CPUExecutionProvider"])
            elif pose_type == "cv2":
                import cv2

                self.pose = cv2.dnn.readNetFromONNX(pose_model_path)
            else:
                self.pose = torch.jit.load(pose_model_path)
        if self.pose_filename is not None:
            self.pose_input_size, _ = _dwpose_guess_onnx_input_shape_dtype(self.pose_filename)

    def __call__(self, ori_img: Any) -> Any:
        if self.pose is None:
            return None
        det_result = []
        if self.det is not None:
            _, det_dtype = _dwpose_guess_onnx_input_shape_dtype(self.det_filename or "yolox_l.onnx")
            det_result = _dwpose_inference_detector(self.det, ori_img, detect_classes=[0], dtype=det_dtype)
            if det_result is None or (hasattr(det_result, "shape") and det_result.shape[0] == 0):
                return None
        _, pose_dtype = _dwpose_guess_onnx_input_shape_dtype(self.pose_filename or "dw-ll_ucoco_384.onnx")
        if _dwpose_is_torchscript(self.pose):
            raise RuntimeError("当前实现不支持 DWPose TorchScript")
        keypoints, scores = _dwpose_inference_pose(self.pose, det_result, ori_img, self.pose_input_size, dtype=pose_dtype)
        keypoints_info = np.concatenate((keypoints, scores[..., None]), axis=-1)
        neck = np.mean(keypoints_info[:, [5, 6]], axis=1)
        neck[:, 2:4] = np.logical_and(keypoints_info[:, 5, 2:4] > 0.3, keypoints_info[:, 6, 2:4] > 0.3).astype(int)
        new_keypoints_info = np.insert(keypoints_info, 17, neck, axis=1)
        mmpose_idx = [17, 6, 8, 10, 7, 9, 12, 14, 16, 13, 15, 2, 1, 4, 3]
        openpose_idx = [1, 2, 3, 4, 6, 7, 8, 9, 10, 12, 13, 14, 15, 16, 17]
        new_keypoints_info[:, openpose_idx] = new_keypoints_info[:, mmpose_idx]
        return new_keypoints_info


_CXY_DWPOSE_CACHE: dict[str, Any] = {"det": None, "pose": None, "det_fn": None, "pose_fn": None, "wb": None}


class _CXYDWPoseDetector:
    def __init__(self, wholebody: _CXYDWPoseWholebody):
        self.wholebody = wholebody

    @classmethod
    def from_pretrained(cls, det_model_path: str, pose_model_path: str) -> "_CXYDWPoseDetector":
        det_fn = os.path.basename(det_model_path)
        pose_fn = os.path.basename(pose_model_path)
        if _CXY_DWPOSE_CACHE["det"] is None or _CXY_DWPOSE_CACHE["det_fn"] != det_fn or _CXY_DWPOSE_CACHE["pose"] is None or _CXY_DWPOSE_CACHE["pose_fn"] != pose_fn:
            wb = _CXYDWPoseWholebody(det_model_path, pose_model_path)
            _CXY_DWPOSE_CACHE["det"] = wb.det
            _CXY_DWPOSE_CACHE["pose"] = wb.pose
            _CXY_DWPOSE_CACHE["det_fn"] = det_fn
            _CXY_DWPOSE_CACHE["pose_fn"] = pose_fn
            _CXY_DWPOSE_CACHE["wb"] = wb
        return cls(_CXY_DWPOSE_CACHE["wb"])

    def detect_poses(self, ori_img: Any) -> List[Any]:
        with torch.no_grad():
            keypoints_info = self.wholebody(ori_img.copy())
            return _dwpose_format_result(keypoints_info)

    def __call__(
        self,
        input_image: Any,
        detect_resolution: int = 512,
        include_body: bool = True,
        include_hand: bool = True,
        include_face: bool = True,
        output_type: str = "pil",
        image_and_json: bool = False,
        upscale_method: str = "INTER_CUBIC",
        xinsr_stick_scaling: bool = False,
        **kwargs: Any,
    ) -> Any:
        if Image is None:
            raise RuntimeError("PIL 不可用，无法处理图片")
        if np is None:
            raise RuntimeError("numpy 不可用，无法运行 DWPose")
        input_image, output_type = _dwpose_common_input_validate(input_image, output_type, **kwargs)
        input_image, _ = _dwpose_resize_image_with_pad(input_image, 0, upscale_method)
        poses = self.detect_poses(input_image)
        canvas = _dwpose_draw_poses(
            poses,
            int(input_image.shape[0]),
            int(input_image.shape[1]),
            draw_body=bool(include_body),
            draw_hand=bool(include_hand),
            draw_face=bool(include_face),
            xinsr_stick_scaling=bool(xinsr_stick_scaling),
        )
        canvas, remove_pad = _dwpose_resize_image_with_pad(canvas, int(detect_resolution), upscale_method)
        detected_map = remove_pad(canvas)
        detected_map = _dwpose_hwc3(detected_map.astype(np.uint8))
        if output_type == "pil":
            detected_map = Image.fromarray(detected_map)
        if image_and_json:
            return detected_map, {"people": [], "canvas_height": int(input_image.shape[0]), "canvas_width": int(input_image.shape[1])}
        return detected_map


def _ensure_dwpose_models(use_cn_mirror: bool) -> Tuple[str, str]:
    det_repo = "yzd-v/DWPose"
    pose_repo = "yzd-v/DWPose"
    det_fn = "yolox_l.onnx"
    pose_fn = "dw-ll_ucoco_384.onnx"
    det_path = _ensure_hf_repo_files(det_repo, [det_fn], use_cn_mirror=use_cn_mirror)[0]
    pose_path = _ensure_hf_repo_files(pose_repo, [pose_fn], use_cn_mirror=use_cn_mirror)[0]
    return det_path, pose_path


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
                        "DWPose",
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
        dwpose_det_path: str | None = None
        dwpose_pose_path: str | None = None

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
            if opt_s == "OpenPose":
                detector_kind = "openpose"
                required_files = ["body_pose_model.pth", "hand_pose_model.pth"]
            else:
                detector_kind = "dwpose"
                required_files = []

        if detector_kind == "dwpose":
            dwpose_det_path, dwpose_pose_path = _ensure_dwpose_models(use_cn_mirror=use_cn_mirror)
        elif required_files:
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
        elif detector_kind == "dwpose":
            if dwpose_det_path is None or dwpose_pose_path is None:
                dwpose_det_path, dwpose_pose_path = _ensure_dwpose_models(use_cn_mirror=use_cn_mirror)
            detector = _CXYDWPoseDetector.from_pretrained(dwpose_det_path, dwpose_pose_path)
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
