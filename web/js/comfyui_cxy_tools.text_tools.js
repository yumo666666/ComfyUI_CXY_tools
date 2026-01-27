import { app } from "../../../scripts/app.js";

const EXT_NAME = "comfyui.cxy.text_tools";

function markDirty() {
  try {
    app?.graph?.setDirtyCanvas(true, true);
  } catch (e) {}
}

function ensureDynamicTextInputs(node) {
  const desiredEmptyTail = 1;
  const textInputs = (node.inputs || []).filter((i) => String(i?.name || "").startsWith("text_"));
  const connected = textInputs.map((inp) => Boolean(inp.link));

  let lastConnectedIndex = -1;
  for (let i = 0; i < connected.length; i++) {
    if (connected[i]) lastConnectedIndex = i;
  }

  const needCount = Math.max(lastConnectedIndex + 1 + desiredEmptyTail, 1);

  while (textInputs.length < needCount) {
    const idx = textInputs.length + 1;
    node.addInput(`text_${idx}`, "STRING");
    textInputs.push(node.inputs[node.inputs.length - 1]);
  }

  while (textInputs.length > 1) {
    const last = textInputs[textInputs.length - 1];
    const prev = textInputs[textInputs.length - 2];
    if (!last.link && !prev.link && textInputs.length > needCount) {
      const removeSlot = node.findInputSlot(last.name);
      if (removeSlot >= 0) node.removeInput(removeSlot);
      textInputs.pop();
    } else {
      break;
    }
  }
}

function ensureDynamicSplitOutputs(node, maxOutputs = 16) {
  const outputs = node.outputs || [];
  if (outputs.length === 0) return;

  const baseType = outputs[0]?.type || "STRING";
  if (outputs[0]?.name !== "part_1") outputs[0].name = "part_1";

  let lastConnectedIndex = -1;
  for (let i = 0; i < outputs.length; i++) {
    const o = outputs[i];
    const linked = Array.isArray(o?.links) && o.links.length > 0;
    if (linked) lastConnectedIndex = i;
  }

  const needCount = Math.max(1, Math.min(maxOutputs, lastConnectedIndex + 2));

  while (outputs.length > needCount) {
    try {
      if (typeof node.removeOutput === "function") {
        node.removeOutput(outputs.length - 1);
      } else {
        outputs.pop();
      }
    } catch (e) {
      break;
    }
  }

  while (outputs.length < needCount) {
    const idx = outputs.length + 1;
    try {
      if (typeof node.addOutput === "function") {
        node.addOutput(`part_${idx}`, baseType);
      } else {
        outputs.push({ name: `part_${idx}`, type: baseType, links: null });
      }
    } catch (e) {
      break;
    }
  }
}

app.registerExtension({
  name: EXT_NAME,
  async beforeRegisterNodeDef(nodeType, nodeData) {
    const isMerge = nodeData.name === "CXY_TextMerge";
    const isSplit = nodeData.name === "CXY_TextSplit";
    if (!isMerge && !isSplit) return;

    const onNodeCreated = nodeType.prototype.onNodeCreated;
    nodeType.prototype.onNodeCreated = function () {
      const r = onNodeCreated ? onNodeCreated.apply(this, arguments) : undefined;
      try {
        if (isMerge) ensureDynamicTextInputs(this);
        if (isSplit) ensureDynamicSplitOutputs(this, 16);
        markDirty();
      } catch (e) {}
      return r;
    };

    const onConfigure = nodeType.prototype.onConfigure;
    nodeType.prototype.onConfigure = function () {
      const r = onConfigure ? onConfigure.apply(this, arguments) : undefined;
      try {
        if (isMerge) ensureDynamicTextInputs(this);
        if (isSplit) ensureDynamicSplitOutputs(this, 16);
        markDirty();
      } catch (e) {}
      return r;
    };

    const onConnectionsChange = nodeType.prototype.onConnectionsChange;
    nodeType.prototype.onConnectionsChange = function () {
      const r = onConnectionsChange ? onConnectionsChange.apply(this, arguments) : undefined;
      try {
        if (isMerge) ensureDynamicTextInputs(this);
        if (isSplit) ensureDynamicSplitOutputs(this, 16);
        markDirty();
      } catch (e) {}
      return r;
    };
  },
});
