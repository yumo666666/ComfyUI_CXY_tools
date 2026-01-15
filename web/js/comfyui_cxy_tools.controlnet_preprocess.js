import { app } from "../../../scripts/app.js";

const EXT_NAME = "comfyui_cxy_tools.controlnet_preprocess";

function markDirty(node) {
  if (node?.setDirtyCanvas) {
    node.setDirtyCanvas(true, true);
    return;
  }
  if (app?.graph?.setDirtyCanvas) {
    app.graph.setDirtyCanvas(true, true);
  }
}

function linkPreprocessCombos(node) {
  const mainWidget = (node.widgets || []).find((w) => w.name === "类型");
  const subWidget = (node.widgets || []).find((w) => w.name === "选项");
  if (!mainWidget || !subWidget) return;
  if (mainWidget._preprocessLinked) return;
  mainWidget._preprocessLinked = true;

  const optionMap = {
    深度图: ["MiDaS", "ZoeDepth", "LeReS"],
    线稿图: ["线稿提取", "软边缘", "硬边缘", "直线"],
    骨架姿势图: ["DWPose", "OpenPose"],
  };

  const applyOptions = () => {
    const mainVal = String(mainWidget.value || "线稿图");
    const opts = optionMap[mainVal] || optionMap["线稿图"];
    subWidget.options.values = opts;
    if (!opts.includes(subWidget.value)) {
      subWidget.value = opts[0];
    }
    markDirty(node);
  };

  const scheduleApply = () => {
    try { applyOptions(); } catch (e) {}
    setTimeout(() => { try { applyOptions(); } catch (e) {} }, 0);
    setTimeout(() => { try { applyOptions(); } catch (e) {} }, 100);
    setTimeout(() => { try { applyOptions(); } catch (e) {} }, 250);
    setTimeout(() => { try { applyOptions(); } catch (e) {} }, 500);
  };

  const arraysEqual = (a, b) => Array.isArray(a) && Array.isArray(b) && a.length === b.length && a.every((x, i) => x === b[i]);
  const syncOptions = () => {
    const mainVal = String(mainWidget.value || "线稿图");
    const desired = optionMap[mainVal] || optionMap["线稿图"];
    const current = (subWidget.options && subWidget.options.values) ? subWidget.options.values : [];
    if (!arraysEqual(current, desired) || !desired.includes(subWidget.value)) {
      subWidget.options.values = desired;
      if (!desired.includes(subWidget.value)) subWidget.value = desired[0];
      markDirty(node);
    }
  };

  const originalCallback = mainWidget.callback;
  mainWidget.callback = function () {
    const rr = originalCallback ? originalCallback.apply(this, arguments) : undefined;
    scheduleApply();
    return rr;
  };

  scheduleApply();
}

function pinCategoryToNodeLibraryTop(categoryName) {
  const key = `cxy_pin_category_${String(categoryName || "")}`;
  if (window[key]) return;
  window[key] = true;

  const tryPinOnce = () => {
    const name = String(categoryName || "").trim();
    if (!name) return false;
    const els = Array.from(document.querySelectorAll("*"));
    for (const el of els) {
      const txt = String(el?.textContent || "").trim();
      if (txt !== name) continue;

      const row = el.closest("li, .tree-item, .node-category, .category, .folder, .entry") || el.parentElement;
      const parent = row && row.parentElement;
      if (!row || !parent) continue;
      if (parent.children.length < 2) continue;
      if (row === parent.firstElementChild) return true;
      parent.insertBefore(row, parent.firstElementChild);
      return true;
    }
    return false;
  };

  const safeTry = () => {
    try {
      return tryPinOnce();
    } catch (e) {
      return false;
    }
  };

  safeTry();
  const obs = new MutationObserver(() => {
    safeTry();
  });
  obs.observe(document.documentElement || document.body, { childList: true, subtree: true });
  setTimeout(safeTry, 500);
  setTimeout(safeTry, 1500);
  setTimeout(safeTry, 5000);
}

if (typeof window !== "undefined" && typeof document !== "undefined") {
  pinCategoryToNodeLibraryTop("CXY工具");
}

app.registerExtension({
  name: EXT_NAME,
  async beforeRegisterNodeDef(nodeType, nodeData) {
    const isPreprocessNode = nodeData.name === "ComfyUI_Annotator_Preprocess";
    if (!isPreprocessNode) return;

    const onNodeCreated = nodeType.prototype.onNodeCreated;
    nodeType.prototype.onNodeCreated = function () {
      const r = onNodeCreated ? onNodeCreated.apply(this, arguments) : undefined;
      try {
        linkPreprocessCombos(this);
      } catch (e) {}
      return r;
    };

    const onConfigure = nodeType.prototype.onConfigure;
    nodeType.prototype.onConfigure = function () {
      const r = onConfigure ? onConfigure.apply(this, arguments) : undefined;
      try {
        linkPreprocessCombos(this);
      } catch (e) {}
      return r;
    };
  },
});

