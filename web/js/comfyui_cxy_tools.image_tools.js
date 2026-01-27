import { app } from "../../../scripts/app.js";

const EXT_NAME = "comfyui.cxy.image_tools";

function markDirty() {
  try {
    app?.graph?.setDirtyCanvas(true, true);
  } catch (e) {}
}

function findWidget(node, name) {
  return (node.widgets || []).find((w) => w?.name === name);
}

function removeWidgetByName(node, name) {
  if (!node.widgets || node.widgets.length === 0) return;
  const idx = node.widgets.findIndex((w) => w?.name === name);
  if (idx < 0) return;
  node.widgets.splice(idx, 1);
}

function trySetWidgetPlaceholder(widget, placeholder) {
  const el = widget?.inputEl || widget?.input || widget?.el || widget?.element;
  if (!el || !el.style) return false;
  const tag = String(el.tagName || "").toUpperCase();
  if (tag !== "TEXTAREA" && tag !== "INPUT") return false;
  try {
    el.placeholder = String(placeholder || "");
    return true;
  } catch (e) {
    return false;
  }
}

function scheduleSetWidgetPlaceholder(widget, placeholder) {
  let tries = 0;
  const tick = () => {
    tries += 1;
    const ok = trySetWidgetPlaceholder(widget, placeholder);
    if (!ok && tries < 20) requestAnimationFrame(tick);
  };
  requestAnimationFrame(tick);
}

function ensureTextWidget(node, name, idx) {
  const existing = findWidget(node, name);
  if (existing) {
    if (Number.isFinite(idx)) scheduleSetWidgetPlaceholder(existing, `对应 image_${idx} 的文字（可空）`);
    return existing;
  }
  const w = node.addWidget("text", name, "", undefined, {});
  if (Number.isFinite(idx)) {
    scheduleSetWidgetPlaceholder(w, `对应 image_${idx} 的文字（可空）`);
  }
  return w;
}

function reorderDynamicTextWidgets(node, maxIndex) {
  const widgets = node.widgets || [];
  const anchorIdx = widgets.findIndex((w) => w?.name === "text_1");
  if (anchorIdx < 0) return;

  const dyn = [];
  for (let i = 2; i <= maxIndex; i++) {
    const w = widgets.find((x) => x?.name === `text_${i}`);
    if (w) dyn.push(w);
  }

  const keep = widgets.filter((w) => !/^text_\d+$/.test(String(w?.name || "")) || w?.name === "text_1");
  const before = keep.slice(0, anchorIdx + 1);
  const after = keep.slice(anchorIdx + 1);
  node.widgets = [...before, ...dyn, ...after];
}

function ensureDynamicImageTextPairs(node) {
  const desiredEmptyTail = 1;
  const imageInputs = (node.inputs || []).filter((i) => String(i?.name || "").startsWith("image_"));
  const connected = imageInputs.map((inp) => Boolean(inp.link));

  let lastConnectedIndex = -1;
  for (let i = 0; i < connected.length; i++) {
    if (connected[i]) lastConnectedIndex = i;
  }

  const needCount = Math.max(lastConnectedIndex + 1 + desiredEmptyTail, 1);

  while (imageInputs.length < needCount) {
    const idx = imageInputs.length + 1;
    node.addInput(`image_${idx}`, "IMAGE");
    imageInputs.push(node.inputs[node.inputs.length - 1]);
    ensureTextWidget(node, `text_${idx}`, idx);
  }

  while (imageInputs.length > 1) {
    const last = imageInputs[imageInputs.length - 1];
    const prev = imageInputs[imageInputs.length - 2];
    if (!last.link && !prev.link && imageInputs.length > needCount) {
      const idx = Number(String(last.name).split("_").pop());
      const removeSlot = node.findInputSlot(last.name);
      if (removeSlot >= 0) node.removeInput(removeSlot);
      if (Number.isFinite(idx) && idx >= 2) removeWidgetByName(node, `text_${idx}`);
      imageInputs.pop();
    } else {
      break;
    }
  }

  for (let i = 1; i <= imageInputs.length; i++) {
    ensureTextWidget(node, `text_${i}`, i);
  }
  reorderDynamicTextWidgets(node, imageInputs.length);
}

app.registerExtension({
  name: EXT_NAME,
  async beforeRegisterNodeDef(nodeType, nodeData) {
    const isJoin = nodeData.name === "CXY_ImageJoin";
    if (!isJoin) return;

    const onNodeCreated = nodeType.prototype.onNodeCreated;
    nodeType.prototype.onNodeCreated = function () {
      const r = onNodeCreated ? onNodeCreated.apply(this, arguments) : undefined;
      try {
        ensureDynamicImageTextPairs(this);
        markDirty();
      } catch (e) {}
      return r;
    };

    const onConfigure = nodeType.prototype.onConfigure;
    nodeType.prototype.onConfigure = function () {
      const r = onConfigure ? onConfigure.apply(this, arguments) : undefined;
      try {
        ensureDynamicImageTextPairs(this);
        markDirty();
      } catch (e) {}
      return r;
    };

    const onConnectionsChange = nodeType.prototype.onConnectionsChange;
    nodeType.prototype.onConnectionsChange = function () {
      const r = onConnectionsChange ? onConnectionsChange.apply(this, arguments) : undefined;
      try {
        ensureDynamicImageTextPairs(this);
        markDirty();
      } catch (e) {}
      return r;
    };
  },
});
