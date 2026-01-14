import base64
import json
import os
import re
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch

try:
    from PIL import Image
except Exception:
    Image = None

try:
    import folder_paths
except Exception:
    folder_paths = None


class ContainsAnyDict(dict):
    """用于接收前端动态创建输入端口的可选输入字典。"""

    def __contains__(self, key: object) -> bool:
        return True

    def __getitem__(self, key: str) -> Tuple[str, Dict[str, Any]]:
        """为任意动态输入名提供默认类型，避免校验阶段KeyError。"""

        return ("IMAGE", {})

    def get(self, key: str, default: Any = None) -> Tuple[str, Dict[str, Any]]:
        """为任意动态输入名提供默认类型。"""

        return ("IMAGE", {})


def _get_plugin_root_dir() -> str:
    """获取插件根目录（custom_nodes/ComfyUI_CXY_tools）。"""

    return str(Path(__file__).resolve().parents[2])


def _get_storage_base_dir() -> str:
    """获取ComfyUI推荐的用户存储目录，无法获取时回退到本扩展目录。"""

    if folder_paths is not None:
        try:
            user_dir = folder_paths.get_user_directory()
            old_dir = os.path.join(user_dir, "ComfyUI_LLM_openai")
            if os.path.isdir(old_dir):
                return old_dir
            return os.path.join(user_dir, "ComfyUI_CXY_tools")
        except Exception:
            pass
    return os.path.join(_get_plugin_root_dir(), "_user")


def _ensure_dirs() -> Tuple[str, str]:
    """确保配置与预设目录存在，返回(config_dir, presets_dir)。"""

    base_dir = _get_storage_base_dir()
    config_dir = base_dir
    presets_dir = os.path.join(base_dir, "presets")
    os.makedirs(config_dir, exist_ok=True)
    os.makedirs(presets_dir, exist_ok=True)
    _ensure_default_presets(presets_dir)
    return config_dir, presets_dir


def _ensure_default_presets(presets_dir: str) -> None:
    """确保内置预设存在（首次运行自动写入presets目录）。"""

    translate_path = os.path.join(presets_dir, "文本翻译.json")
    if not os.path.exists(translate_path):
        _write_json_file(
            translate_path,
            {
                "name": "文本翻译",
                "prompt": "将以下文本从${source_lang}翻译为${target_lang}：\n${text}",
            },
        )


def _config_path() -> str:
    """返回配置文件路径。"""

    config_dir, _ = _ensure_dirs()
    return os.path.join(config_dir, "config.json")


def _preset_file_path(name: str) -> str:
    """返回预设文件路径，自动进行安全文件名清洗。"""

    _, presets_dir = _ensure_dirs()
    safe = re.sub(r"[^0-9A-Za-z\u4e00-\u9fff\-_ ]+", "_", name).strip()
    safe = safe[:128] if safe else "preset"
    return os.path.join(presets_dir, f"{safe}.json")


def _strip_control_chars(s: str) -> str:
    """去除控制字符，保留常见可见字符。"""

    return "".join(ch for ch in s if ch == "\n" or ch == "\t" or (ord(ch) >= 32 and ord(ch) != 127))


def _format_output_text(s: str) -> str:
    """格式化输出内容：去多余空白与特殊控制字符。"""

    s = _strip_control_chars(s)
    s = s.strip()
    s = re.sub(r"[ \f\v\r]+", " ", s)
    return s


def _validate_range(name: str, value: float, min_v: float, max_v: float) -> None:
    """验证数值范围。"""

    if value < min_v or value > max_v:
        raise ValueError(f"{name} 超出范围：{min_v} - {max_v}")


def _read_json_file(path: str) -> Optional[Dict[str, Any]]:
    """读取JSON文件，失败返回None。"""

    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _write_json_file(path: str, data: Dict[str, Any]) -> None:
    """写入JSON文件。"""

    tmp_path = f"{path}.tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    os.replace(tmp_path, path)


def _dpapi_encrypt_to_b64(plain_text: str) -> str:
    """使用Windows DPAPI对敏感信息加密并以base64返回。"""

    import ctypes
    import ctypes.wintypes

    class DATA_BLOB(ctypes.Structure):
        _fields_ = [("cbData", ctypes.wintypes.DWORD), ("pbData", ctypes.POINTER(ctypes.c_byte))]

    crypt32 = ctypes.WinDLL("crypt32", use_last_error=True)
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)

    data = plain_text.encode("utf-8")
    in_blob = DATA_BLOB(len(data), ctypes.cast(ctypes.create_string_buffer(data), ctypes.POINTER(ctypes.c_byte)))
    out_blob = DATA_BLOB()

    if not crypt32.CryptProtectData(ctypes.byref(in_blob), None, None, None, None, 0, ctypes.byref(out_blob)):
        raise OSError("DPAPI 加密失败")

    try:
        raw = ctypes.string_at(out_blob.pbData, out_blob.cbData)
        return base64.b64encode(raw).decode("ascii")
    finally:
        kernel32.LocalFree(out_blob.pbData)


def _dpapi_decrypt_from_b64(cipher_b64: str) -> str:
    """使用Windows DPAPI解密base64密文，返回明文。"""

    import ctypes
    import ctypes.wintypes

    class DATA_BLOB(ctypes.Structure):
        _fields_ = [("cbData", ctypes.wintypes.DWORD), ("pbData", ctypes.POINTER(ctypes.c_byte))]

    crypt32 = ctypes.WinDLL("crypt32", use_last_error=True)
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)

    raw = base64.b64decode(cipher_b64.encode("ascii"))
    in_blob = DATA_BLOB(len(raw), ctypes.cast(ctypes.create_string_buffer(raw), ctypes.POINTER(ctypes.c_byte)))
    out_blob = DATA_BLOB()

    if not crypt32.CryptUnprotectData(ctypes.byref(in_blob), None, None, None, None, 0, ctypes.byref(out_blob)):
        raise OSError("DPAPI 解密失败")

    try:
        plain = ctypes.string_at(out_blob.pbData, out_blob.cbData)
        return plain.decode("utf-8")
    finally:
        kernel32.LocalFree(out_blob.pbData)


@dataclass
class LLMConfig:
    """LLM配置结构。"""

    base_url: str = "https://api.openai.com"
    api_key_enc_b64: str = ""
    models: List[str] = None

    def to_json(self) -> Dict[str, Any]:
        """序列化为JSON对象。"""

        return {
            "base_url": self.base_url,
            "api_key_enc_b64": self.api_key_enc_b64,
            "models": self.models or [],
        }

    @staticmethod
    def from_json(data: Dict[str, Any]) -> "LLMConfig":
        """从JSON对象反序列化。"""

        base_url = str(data.get("base_url") or "https://api.openai.com").strip()
        api_key_enc_b64 = str(data.get("api_key_enc_b64") or "")
        models = data.get("models")
        if not isinstance(models, list):
            models = []
        models = [str(x) for x in models if str(x).strip()]
        if not models:
            models = ["Qwen/Qwen3-8B", "gpt-4o-mini"]
        return LLMConfig(base_url=base_url, api_key_enc_b64=api_key_enc_b64, models=models)


def load_config() -> LLMConfig:
    """读取配置，不存在时返回默认配置。"""

    path = _config_path()
    data = _read_json_file(path)
    if not data:
        return LLMConfig.from_json({})
    try:
        return LLMConfig.from_json(data)
    except Exception:
        return LLMConfig.from_json({})


def save_config(cfg: LLMConfig) -> None:
    """保存配置到JSON文件。"""

    _write_json_file(_config_path(), cfg.to_json())


def list_presets() -> List[Dict[str, Any]]:
    """扫描presets目录并返回预设列表（仅元信息）。"""

    _, presets_dir = _ensure_dirs()
    items: List[Dict[str, Any]] = []
    for fn in os.listdir(presets_dir):
        if not fn.lower().endswith(".json"):
            continue
        path = os.path.join(presets_dir, fn)
        try:
            data = _read_json_file(path) or {}
            name = str(data.get("name") or os.path.splitext(fn)[0])
            items.append({"name": name})
        except Exception:
            continue
    items.sort(key=lambda x: x["name"])
    return items


def load_preset_by_name(name: str) -> Optional[Dict[str, Any]]:
    """按名称加载预设JSON内容。"""

    path = _preset_file_path(name)
    data = _read_json_file(path)
    if not data:
        return None
    if "name" not in data:
        data["name"] = name
    return data


def _tensor_image_to_png_b64(image: torch.Tensor) -> str:
    """将ComfyUI IMAGE Tensor转换为PNG base64（data部分）。"""

    if Image is None:
        raise RuntimeError("PIL 不可用，无法编码图片")

    img = image.detach()
    if img.dtype != torch.float32 and img.dtype != torch.float16:
        img = img.float()
    img = torch.clamp(img, 0.0, 1.0)
    if img.ndim != 3 or img.shape[-1] not in (3, 4):
        raise ValueError("IMAGE Tensor 形状必须为 [H,W,C]")

    arr = (img.cpu().numpy() * 255.0).astype("uint8")
    pil = Image.fromarray(arr, mode="RGBA" if arr.shape[-1] == 4 else "RGB")
    import io

    buf = io.BytesIO()
    pil.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


def _build_openai_payload(
    model: str,
    content: List[Dict[str, Any]],
    temperature: float,
    max_tokens: int,
) -> Dict[str, Any]:
    """构建OpenAI标准Chat Completions请求体（非流式）。"""

    return {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": content,
            }
        ],
        "temperature": float(temperature),
        "max_tokens": int(max_tokens),
        "stream": False,
    }


def _http_post_json(url: str, headers: Dict[str, str], payload: Dict[str, Any], timeout_s: int) -> Tuple[int, str]:
    """发送HTTP JSON POST请求，返回(status_code, response_text)。"""

    data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    req = urllib.request.Request(url, data=data, method="POST")
    for k, v in headers.items():
        req.add_header(k, v)

    try:
        with urllib.request.urlopen(req, timeout=timeout_s) as resp:
            body = resp.read().decode("utf-8", errors="replace")
            return int(resp.status), body
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", errors="replace") if hasattr(e, "read") else str(e)
        return int(getattr(e, "code", 500)), body
    except urllib.error.URLError as e:
        return 599, str(getattr(e, "reason", e))


def _strip_thinking_content(text: str) -> str:
    """过滤思考/推理等非最终回答内容，仅保留最终输出文本。"""

    if not text:
        return ""

    s = str(text)
    s = re.sub(r"<think>[\s\S]*?</think>", "", s, flags=re.IGNORECASE)
    s = re.sub(r"<analysis>[\s\S]*?</analysis>", "", s, flags=re.IGNORECASE)
    s = re.sub(r"<begin_of_box>[\s\S]*?<end_of_box>", "", s, flags=re.IGNORECASE)
    s = re.sub(r"\(begin_of_box\)[\s\S]*?\(end_of_box\)", "", s, flags=re.IGNORECASE)
    s = re.sub(r"^\s*思考[:：][\s\S]*?$", "", s, flags=re.MULTILINE)
    return s.strip()


def _extract_final_answer(resp: Dict[str, Any]) -> str:
    """从OpenAI兼容响应中提取最终assistant回答文本。"""

    if not isinstance(resp, dict):
        return ""

    choices = resp.get("choices")
    if isinstance(choices, list) and choices:
        choice0 = choices[0] if isinstance(choices[0], dict) else {}
        message = choice0.get("message") if isinstance(choice0, dict) else None
        if isinstance(message, dict):
            content = message.get("content")
            if isinstance(content, str):
                return content
            if isinstance(content, list):
                parts: List[str] = []
                for item in content:
                    if isinstance(item, dict) and item.get("type") == "text" and isinstance(item.get("text"), str):
                        parts.append(item["text"])
                return "\n".join(parts)
        content = choice0.get("text") if isinstance(choice0, dict) else None
        if isinstance(content, str):
            return content

    output = resp.get("output")
    if isinstance(output, list) and output:
        parts = []
        for item in output:
            if not isinstance(item, dict):
                continue
            content = item.get("content")
            if isinstance(content, list):
                for c in content:
                    if isinstance(c, dict) and c.get("type") in ("output_text", "text") and isinstance(c.get("text"), str):
                        parts.append(c["text"])
        return "\n".join(parts)

    message = resp.get("message")
    if isinstance(message, dict) and isinstance(message.get("content"), str):
        return message["content"]

    return ""


class ComfyUILLMOpenAINode:
    """支持动态IMAGE输入端口、模型/预设选择与OpenAI非流式调用的节点。"""

    CATEGORY = "CXY工具"
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("response",)
    FUNCTION = "call"

    @classmethod
    def INPUT_TYPES(cls) -> Dict[str, Any]:
        """定义节点输入类型，使用可接收任意动态输入的可选字典。"""

        cfg = load_config()
        models = cfg.models or ["gpt-4o-mini"]
        return {
            "required": {
                "text": ("STRING", {"multiline": True, "default": ""}),
                "model": (models, {"default": models[0] if models else "gpt-4o-mini"}),
                "preset": (["无"] + [p["name"] for p in list_presets()], {"default": "无"}),
                "temperature": ("FLOAT", {"default": 0.7, "min": 0.0, "max": 2.0, "step": 0.05}),
                "max_tokens": ("INT", {"default": 2048, "min": 100, "max": 100000, "step": 1}),
                "timeout": ("INT", {"default": 30, "min": 5, "max": 300, "step": 1}),
                "preset_preview": ("STRING", {"multiline": True, "default": ""}),
            },
            "optional": ContainsAnyDict(),
        }

    @classmethod
    def VALIDATE_INPUTS(cls, input_types: Dict[str, Any]) -> bool:
        """跳过动态输入类型校验，由节点内部自行校验。"""

        return True

    def call(
        self,
        text: str,
        model: str,
        preset: str,
        temperature: float = 0.7,
        max_tokens: int = 2048,
        timeout: int = 30,
        **kwargs: Any,
    ) -> Tuple[str]:
        """执行OpenAI非流式请求并输出最终回答文本。"""

        _validate_range("temperature", float(temperature), 0.0, 2.0)
        if int(max_tokens) < 100:
            raise ValueError("max_tokens 最小为100")
        _validate_range("timeout", float(timeout), 5.0, 300.0)

        user_text = str(text or "").strip()
        preset_text = ""
        preset_has_text_slot = False

        if preset and preset != "无":
            preset_obj = load_preset_by_name(preset)
            if preset_obj and isinstance(preset_obj.get("prompt"), str):
                tmpl_raw = str(preset_obj["prompt"])
                preset_has_text_slot = "${text}" in tmpl_raw

                tmpl = tmpl_raw.replace("${source_lang}", "自动识别").replace("${target_lang}", "中文")
                preset_text = tmpl.replace("${text}", user_text) if preset_has_text_slot else tmpl

        if preset_text.strip():
            if preset_has_text_slot:
                prompt_text = preset_text.strip()
            else:
                prompt_text = preset_text.strip() + (f"\n{user_text}" if user_text else "")
        else:
            prompt_text = user_text

        images: List[torch.Tensor] = []
        for k, v in kwargs.items():
            if not isinstance(k, str) or not k.lower().startswith("image"):
                continue
            if isinstance(v, torch.Tensor):
                if v.ndim == 4:
                    for i in range(v.shape[0]):
                        images.append(v[i])
                elif v.ndim == 3:
                    images.append(v)

        if not prompt_text.strip() and not images:
            raise ValueError("text、preset、images 至少提供一个")

        cfg = load_config()
        if not cfg.api_key_enc_b64:
            raise ValueError("未配置 API Key，请在右键菜单中设置")

        api_key = _dpapi_decrypt_from_b64(cfg.api_key_enc_b64)
        base_url = (cfg.base_url or "https://api.openai.com").rstrip("/")
        if base_url.endswith("/v1"):
            base_url = base_url[:-3]
        url = f"{base_url}/v1/chat/completions"

        content: List[Dict[str, Any]] = []
        if prompt_text.strip():
            content.append({"type": "text", "text": prompt_text.strip()})
        for idx, img in enumerate(images, start=1):
            content.append({"type": "text", "text": f"图{idx}：第{idx}张图"})
            b64 = _tensor_image_to_png_b64(img)
            content.append({"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}})

        payload = _build_openai_payload(
            model=str(model).strip(),
            content=content,
            temperature=float(temperature),
            max_tokens=int(max_tokens),
        )

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        }

        t0 = time.time()
        status, body_text = _http_post_json(url, headers, payload, timeout_s=int(timeout))
        elapsed = time.time() - t0

        out: str
        try:
            parsed = json.loads(body_text)
            out = _extract_final_answer(parsed)
        except Exception:
            out = ""

        if status >= 400:
            raw_err = _format_output_text(body_text)
            hint = ""
            if status == 504 and (
                "Gateway Time-out" in body_text or "Gateway Time-out" in raw_err or "openresty" in body_text
            ):
                hint = "\n\n提示：这是上游网关(openresty)返回的超时(504)，通常与代理/反代超时有关，不一定是本节点timeout参数导致。可尝试：降低max_tokens/减少图片、换更快的模型、或在你的反代/网关里把超时调大。"
            raise RuntimeError(f"HTTP {status}（{elapsed:.2f}s）：{raw_err}{hint}")

        out = _strip_thinking_content(out)
        out = _format_output_text(out)
        if not out:
            raise RuntimeError("模型未返回可用的最终回答内容")

        return (out,)


NODE_CLASS_MAPPINGS = {
    "ComfyUI_LLM_openai": ComfyUILLMOpenAINode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ComfyUI_LLM_openai": "ComfyUI_LLM_openai",
}
