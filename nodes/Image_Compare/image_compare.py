from typing import Any, Dict

import torch

from ..ComfyUI_Annotator_Preprocess.controlnet_preprocess import _pick_first_image, _save_tensor_image_to_temp


class ImageCompareNode:
    """输入两张图，节点内可用鼠标悬浮进行对比显示。"""

    CATEGORY = "CXY工具/图片处理"
    OUTPUT_NODE = True
    RETURN_TYPES = ()
    RETURN_NAMES = ()
    FUNCTION = "run"

    @classmethod
    def INPUT_TYPES(cls) -> Dict[str, Any]:
        """定义节点输入。"""

        return {
            "required": {
                "反转": ("BOOLEAN", {"default": False}),
                "image_a": ("IMAGE",),
                "image_b": ("IMAGE",),
            }
        }

    def run(self, 反转: bool, image_a: torch.Tensor, image_b: torch.Tensor) -> Dict[str, Any]:
        """保存两张输入图用于前端展示，不输出端口。"""

        a = _pick_first_image(image_a).unsqueeze(0)
        b = _pick_first_image(image_b).unsqueeze(0)

        ui_images = [
            _save_tensor_image_to_temp(a, subfolder="compare", prefix="a"),
            _save_tensor_image_to_temp(b, subfolder="compare", prefix="b"),
        ]
        return {
            "ui": {
                "images": ui_images,
                "cxy_compare": {"images": ui_images, "invert": bool(反转)},
            },
            "result": (),
        }


NODE_CLASS_MAPPINGS = {
    "Image_Compare": ImageCompareNode,
}


NODE_DISPLAY_NAME_MAPPINGS = {
    "Image_Compare": "图比较",
}
