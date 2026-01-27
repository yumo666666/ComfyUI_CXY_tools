# ComfyUI_CXY_tools

本插件包含以下节点（节点库分类：`CXY工具`）：

## 节点列表

### CXY工具
- ComfyUI_LLM_openai：文本/多图提示词拼接并调用 OpenAI 兼容接口
- Controlnet图预处理：深度/线稿/姿态预处理（自动下载缺失模型到插件内 models）

### CXY工具/文字处理
- 文本去除/替换：按字面量/正则删除指定内容，支持解析 `\\n` 等转义
- 文本合并：动态追加文本输入端口，按顺序合并输出
- 文本拆分：按分隔符切分文本，输出端口按连接动态逐步出现

### CXY工具/图片处理
- 图片拼接（可选文字）：按最大尺寸缩放裁剪减少留黑，支持整体背景颜色与外边距
- 图比较：输入两张图，鼠标左右移动进行对比，支持反转层级

节点详细说明：

- [LLM_openai 节点](./docs/nodes/llm_openai.md)
- [Controlnet 图预处理节点](./docs/nodes/controlnet_preprocess.md)
- [图比较节点](./docs/nodes/image_compare.md)
- [图片拼接（可选文字）节点](./docs/nodes/image_join.md)
- [文本去除/替换节点](./docs/nodes/text_remove.md)
- [文本合并节点](./docs/nodes/text_merge.md)
- [文本拆分节点](./docs/nodes/text_split.md)
