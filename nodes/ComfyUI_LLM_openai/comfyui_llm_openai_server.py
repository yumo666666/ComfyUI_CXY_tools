from typing import Any, Dict

from aiohttp import web

from .llm_openai_nodes import (
    _dpapi_encrypt_to_b64,
    _preset_file_path,
    _write_json_file,
    list_presets,
    load_config,
    load_preset_by_name,
    save_config,
)

try:
    from server import PromptServer
except Exception:
    PromptServer = None


API_PREFIX = "/comfyui_llm_openai"


def _json_response(data: Any, status: int = 200) -> web.Response:
    """返回JSON响应。"""

    return web.json_response(data, status=status)


async def _read_request_json(request: web.Request) -> Dict[str, Any]:
    """读取请求JSON体，失败返回空字典。"""

    try:
        data = await request.json()
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


async def get_config(request: web.Request) -> web.Response:
    """获取当前配置（不返回明文API Key）。"""

    cfg = load_config()
    return _json_response(
        {
            "base_url": cfg.base_url,
            "models": cfg.models or [],
            "has_api_key": bool(cfg.api_key_enc_b64),
        }
    )


async def set_config(request: web.Request) -> web.Response:
    """更新配置（支持更新base_url、models、api_key）。"""

    payload = await _read_request_json(request)
    cfg = load_config()

    base_url = payload.get("base_url")
    if isinstance(base_url, str) and base_url.strip():
        v = base_url.strip().rstrip("/")
        if v.endswith("/v1"):
            v = v[:-3]
        cfg.base_url = v

    models = payload.get("models")
    if isinstance(models, list):
        cfg.models = [str(x).strip() for x in models if str(x).strip()]

    api_key = payload.get("api_key")
    if isinstance(api_key, str):
        api_key = api_key.strip()
        if api_key:
            cfg.api_key_enc_b64 = _dpapi_encrypt_to_b64(api_key)

    if not cfg.models:
        cfg.models = ["gpt-4o-mini"]

    save_config(cfg)
    return _json_response({"ok": True})


async def list_models(request: web.Request) -> web.Response:
    """返回模型列表。"""

    cfg = load_config()
    return _json_response({"models": cfg.models or []})


async def presets_list(request: web.Request) -> web.Response:
    """返回预设列表。"""

    return _json_response({"presets": list_presets()})


async def presets_get(request: web.Request) -> web.Response:
    """获取单个预设。"""

    name = request.match_info.get("name", "")
    preset = load_preset_by_name(name)
    if not preset:
        return _json_response({"error": "not_found"}, status=404)
    return _json_response(preset)


async def presets_upsert(request: web.Request) -> web.Response:
    """创建或更新预设（单文件JSON）。"""

    payload = await _read_request_json(request)
    name = payload.get("name")
    prompt = payload.get("prompt")
    if not isinstance(name, str) or not name.strip():
        return _json_response({"error": "name_required"}, status=400)
    if not isinstance(prompt, str) or not prompt.strip():
        return _json_response({"error": "prompt_required"}, status=400)

    name = name.strip()
    path = _preset_file_path(name)
    _write_json_file(path, {"name": name, "prompt": prompt})
    return _json_response({"ok": True})


async def presets_delete(request: web.Request) -> web.Response:
    """删除预设文件。"""

    name = request.match_info.get("name", "")
    if not name:
        return _json_response({"error": "name_required"}, status=400)
    path = _preset_file_path(name)
    try:
        import os

        if os.path.exists(path):
            os.remove(path)
        return _json_response({"ok": True})
    except Exception:
        return _json_response({"error": "delete_failed"}, status=500)


def _register_routes() -> None:
    """注册aiohttp路由。"""

    if PromptServer is None:
        return

    routes = PromptServer.instance.routes

    routes.get(f"{API_PREFIX}/config")(get_config)
    routes.post(f"{API_PREFIX}/config")(set_config)
    routes.get(f"{API_PREFIX}/models")(list_models)
    routes.get(f"{API_PREFIX}/presets")(presets_list)
    routes.get(f"{API_PREFIX}/presets/{{name}}")(presets_get)
    routes.post(f"{API_PREFIX}/presets")(presets_upsert)
    routes.delete(f"{API_PREFIX}/presets/{{name}}")(presets_delete)


_register_routes()

