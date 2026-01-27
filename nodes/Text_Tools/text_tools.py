from __future__ import annotations

import re
from typing import Any, Dict, List, Tuple


class ContainsAnyTextDict(dict):
    """用于接收前端动态创建文本输入端口的可选输入字典。"""

    def __contains__(self, key: object) -> bool:
        """仅允许以 text_ 开头的动态输入端口通过可选输入校验。"""

        return isinstance(key, str) and key.lower().startswith("text_")

    def __getitem__(self, key: str) -> Tuple[str, Dict[str, Any]]:
        """为任意动态文本输入名提供默认类型，避免校验阶段KeyError。"""

        if isinstance(key, str) and key.lower().startswith("text_"):
            return ("STRING", {"forceInput": True})
        raise KeyError(key)

    def get(self, key: str, default: Any = None) -> Tuple[str, Dict[str, Any]]:
        """为任意动态文本输入名提供默认类型。"""

        if isinstance(key, str) and key.lower().startswith("text_"):
            return ("STRING", {"forceInput": True})
        return default


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


def _extract_index(name: str) -> int:
    """从 text_1 这类名字中提取编号，无法提取则返回较大值用于排序。"""

    m = re.search(r"(\d+)$", str(name))
    if not m:
        return 10**9
    try:
        return int(m.group(1))
    except Exception:
        return 10**9


class CXYTextRemoveNode:
    """按“字面量/正则”删除指定内容并输出处理后的文本。"""

    CATEGORY = "CXY工具/文字处理"
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("text",)
    FUNCTION = "run"

    @classmethod
    def INPUT_TYPES(cls) -> Dict[str, Any]:
        """定义节点输入。"""

        return {
            "required": {
                "text": ("STRING", {"multiline": True, "default": ""}),
                "remove": ("STRING", {"multiline": True, "default": ""}),
                "remove_mode": (["字面量删除", "正则删除"], {"default": "字面量删除"}),
                "解析转义": ("BOOLEAN", {"default": True}),
                "忽略大小写": ("BOOLEAN", {"default": False}),
                "忽略空白项": ("BOOLEAN", {"default": True}),
                "多项模式": (["按行", "按逗号"], {"default": "按行"}),
            }
        }

    def run(
        self,
        text: str,
        remove: str,
        remove_mode: str,
        解析转义: bool = True,
        忽略大小写: bool = False,
        忽略空白项: bool = True,
        多项模式: str = "按行",
    ) -> Tuple[str]:
        """执行文本删除/替换并输出结果。"""

        src = str(text or "")
        raw_remove = str(remove or "")
        if 解析转义:
            raw_remove = _decode_escapes(raw_remove)

        items: List[str]
        if 多项模式 == "按逗号":
            items = [x.strip() for x in raw_remove.split(",")]
        else:
            items = [x.rstrip("\r") for x in raw_remove.split("\n")]

        if 忽略空白项:
            items = [x for x in items if x.strip() != ""]

        if not items:
            return (src,)

        if remove_mode == "正则删除":
            flags = re.IGNORECASE if 忽略大小写 else 0
            for pat in items:
                src = re.sub(pat, "", src, flags=flags)
            return (src,)

        if 忽略大小写:
            for lit in items:
                if not lit:
                    continue
                src = re.sub(re.escape(lit), "", src, flags=re.IGNORECASE)
            return (src,)

        for lit in items:
            if not lit:
                continue
            src = src.replace(lit, "")
        return (src,)


class CXYTextMergeNode:
    """将多个文本输入按端口顺序合并输出。"""

    CATEGORY = "CXY工具/文字处理"
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("text",)
    FUNCTION = "run"

    @classmethod
    def INPUT_TYPES(cls) -> Dict[str, Any]:
        """定义节点输入，允许前端动态创建 text_N 输入端口。"""

        return {
            "required": {
                "separator": ("STRING", {"default": "\\n"}),
                "解析转义": ("BOOLEAN", {"default": True}),
            },
            "optional": ContainsAnyTextDict(),
        }

    @classmethod
    def VALIDATE_INPUTS(cls, input_types: Dict[str, Any]) -> bool:
        """跳过动态输入类型校验，由节点内部自行处理。"""

        return True

    def run(self, separator: str = "\\n", 解析转义: bool = True, **kwargs: Any) -> Tuple[str]:
        """按端口编号从小到大合并所有已连接文本。"""

        sep = str(separator or "")
        if 解析转义:
            sep = _decode_escapes(sep)

        parts: List[str] = []
        keys = [k for k in kwargs.keys() if isinstance(k, str) and k.lower().startswith("text_")]
        keys.sort(key=_extract_index)
        for k in keys:
            v = kwargs.get(k)
            if v is None:
                continue
            parts.append(str(v))
        return (sep.join(parts),)


class CXYTextSplitNode:
    """按分隔符拆分文本并输出多路结果（输出口由前端按连接动态显示）。"""

    CATEGORY = "CXY工具/文字处理"
    RETURN_TYPES = ("STRING",) * 16
    RETURN_NAMES = tuple([f"part_{i}" for i in range(1, 17)])
    FUNCTION = "run"

    @classmethod
    def INPUT_TYPES(cls) -> Dict[str, Any]:
        """定义节点输入。"""

        return {
            "required": {
                "text": ("STRING", {"multiline": True, "default": "", "forceInput": True}),
                "separator": ("STRING", {"default": "\\n"}),
                "解析转义": ("BOOLEAN", {"default": True}),
                "越界策略": (["输出空字符串", "重复最后一段", "循环填充"], {"default": "输出空字符串"}),
            }
        }

    def run(
        self,
        text: str,
        separator: str = "\\n",
        解析转义: bool = True,
        越界策略: str = "输出空字符串",
    ) -> Tuple[str, ...]:
        """拆分后按序输出到固定数量的端口，超过长度时按越界策略填充。"""

        src = str(text or "")
        sep = str(separator or "")
        if 解析转义:
            sep = _decode_escapes(sep)

        if sep == "":
            parts = list(src)
        else:
            parts = src.split(sep)

        out: List[str] = []
        for i in range(16):
            if i < len(parts):
                out.append(str(parts[i]))
                continue

            if 越界策略 == "重复最后一段" and parts:
                out.append(str(parts[-1]))
                continue
            if 越界策略 == "循环填充" and parts:
                out.append(str(parts[i % len(parts)]))
                continue
            out.append("")

        return tuple(out)


NODE_CLASS_MAPPINGS = {
    "CXY_TextRemove": CXYTextRemoveNode,
    "CXY_TextMerge": CXYTextMergeNode,
    "CXY_TextSplit": CXYTextSplitNode,
}


NODE_DISPLAY_NAME_MAPPINGS = {
    "CXY_TextRemove": "文本去除/替换",
    "CXY_TextMerge": "文本合并",
    "CXY_TextSplit": "文本拆分",
}

