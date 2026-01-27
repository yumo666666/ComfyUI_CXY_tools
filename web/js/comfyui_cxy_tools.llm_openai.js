import { app } from "../../../scripts/app.js";

const EXT_NAME = "comfyui_cxy_tools.llm_openai";
const API_PREFIX = "/comfyui_llm_openai";

const PRESET_NEW_VALUE = "__comfyui_llm_openai_new__";
const PRESET_PREVIEW_KEY = "preset_preview";
const PRESET_PREVIEW_KEY_OLD = "__preset_preview";
const PRESET_PREVIEW_DISPLAY_OLD = "预设内容";
const TEXT_PLACEHOLDER = "prompt";
const FIVE_LINE_HEIGHT = 100;

const presetDetailCache = new Map();

function buildApiUrl(path) {
  const base = window.location.origin;
  return `${base}${path}`;
}

async function fetchJson(url, options = {}) {
  const resp = await fetch(url, options);
  const text = await resp.text();
  let data;
  try {
    data = JSON.parse(text);
  } catch (e) {
    data = { error: "invalid_json", raw: text };
  }
  if (!resp.ok) {
    const err = new Error(data?.error || `HTTP ${resp.status}`);
    err.data = data;
    err.status = resp.status;
    throw err;
  }
  return data;
}

async function fetchComboData() {
  const [cfg, presets] = await Promise.all([
    fetchJson(buildApiUrl(`${API_PREFIX}/config`)),
    fetchJson(buildApiUrl(`${API_PREFIX}/presets`)),
  ]);

  const models = (cfg.models || []).length ? cfg.models : ["gpt-4o-mini"];
  const presetNames = (presets.presets || []).map((p) => p.name).filter(Boolean);
  return { models, presetNames };
}

function markDirty(node) {
  if (node?.setDirtyCanvas) {
    node.setDirtyCanvas(true, true);
    return;
  }
  if (app?.graph?.setDirtyCanvas) {
    app.graph.setDirtyCanvas(true, true);
  }
}

function applyComboDataToNode(node, data) {
  const modelWidget = (node.widgets || []).find((w) => w.name === "model");
  if (modelWidget && modelWidget.options) {
    modelWidget.options.values = data.models;
    if (!modelWidget.value || !data.models.includes(modelWidget.value)) {
      modelWidget.value = data.models[0];
    }
  }

  const presetWidget = (node.widgets || []).find((w) => w.name === "preset");
  if (presetWidget && presetWidget.options) {
    presetWidget.options.values = ["无", ...data.presetNames];
    if (!presetWidget.value || !presetWidget.options.values.includes(presetWidget.value)) {
      presetWidget.value = "无";
    }
  }

  updatePresetPreview(node).catch(() => {});

  markDirty(node);
}

async function refreshAllManagedNodes() {
  const data = await fetchComboData();
  const nodes = (app?.graph?._nodes || []).filter((n) => n?.properties?._comfyui_llm_openai === true);
  for (const n of nodes) {
    applyComboDataToNode(n, data);
  }
}

function ensureDynamicImagePorts(node) {
  if (!node.properties) node.properties = {};
  if (typeof node.properties._dynImageCount !== "number") {
    node.properties._dynImageCount = 1;
  }

  const desiredEmptyTail = 1;

  const imageInputs = (node.inputs || []).filter((i) => i?.name?.startsWith("image_"));
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
  }

  while (imageInputs.length > 1) {
    const last = imageInputs[imageInputs.length - 1];
    const prev = imageInputs[imageInputs.length - 2];
    if (!last.link && !prev.link && imageInputs.length > needCount) {
      const removeSlot = node.findInputSlot(last.name);
      if (removeSlot >= 0) node.removeInput(removeSlot);
      imageInputs.pop();
    } else {
      break;
    }
  }

  node.properties._dynImageCount = imageInputs.length;
}

async function refreshNodeCombos(node) {
  const data = await fetchComboData();
  applyComboDataToNode(node, data);
}

async function fetchPresetDetailByName(name) {
  const key = String(name || "").trim();
  if (!key || key === "无") return null;

  const cached = presetDetailCache.get(key);
  if (cached && Date.now() - cached.ts < 2000) {
    return cached.data;
  }

  const data = await fetchJson(buildApiUrl(`${API_PREFIX}/presets/${encodeURIComponent(key)}`));
  presetDetailCache.set(key, { ts: Date.now(), data });
  return data;
}

function makeWidgetReadOnly(widget) {
  if (!widget) return;
  widget.serialize = false;
  widget._llmReadOnly = true;
  widget.options = widget.options || {};
  widget.options.readOnly = true;
  if (!widget._llmReadOnlyCallbackPatched) {
    widget._llmReadOnlyCallbackPatched = true;
    const originalCallback = widget.callback;
    widget.callback = function () {
      if (widget._llmReadOnly === true) {
        const keep = widget._llmReadOnlyValue ?? "";
        if (widget.value !== keep) {
          widget.value = keep;
          markDirty(app?.graph);
        }
        return;
      }
      return originalCallback ? originalCallback.apply(this, arguments) : undefined;
    };
  }
}

function trySetWidgetInputReadOnly(widget) {
  const el = widget?.inputEl || widget?.input || widget?.el || widget?.element;
  if (!el || !el.style) return;
  const tag = String(el.tagName || "").toUpperCase();
  if (tag !== "TEXTAREA" && tag !== "INPUT") return;
  try {
    el.readOnly = true;
    el.style.overflowY = "auto";
    el.style.resize = "none";
  } catch (e) {}
}

function scheduleTrySetWidgetInputReadOnly(widget) {
  let tries = 0;
  const tick = () => {
    tries += 1;
    trySetWidgetInputReadOnly(widget);
    if (tries < 10) requestAnimationFrame(tick);
  };
  requestAnimationFrame(tick);
}

function clampWidgetHeight(widget, heightPx) {
  if (!widget) return;
  widget._llmClampHeightValue = heightPx;
  if (widget._llmClampHeightPatched) return;
  widget._llmClampHeightPatched = true;
  const originalComputeSize = widget.computeSize;
  widget.computeSize = function (width) {
    if (widget._llmHidden === true) return [width, 0];
    const h = Number(widget._llmClampHeightValue);
    const height = Number.isFinite(h) && h > 0 ? h : 60;
    const s = originalComputeSize ? originalComputeSize.call(this, width) : [width, height];
    return [s[0], height];
  };
}

function trySetActiveTextareaReadOnly() {
  const el = document.activeElement;
  if (!el || !el.style) return;
  const tag = String(el.tagName || "").toUpperCase();
  if (tag !== "TEXTAREA" && tag !== "INPUT") return;
  try {
    el.readOnly = true;
    el.style.overflowY = "auto";
    el.style.resize = "none";
  } catch (e) {}
}

function scheduleTrySetActiveTextareaReadOnly() {
  let tries = 0;
  const tick = () => {
    tries += 1;
    trySetActiveTextareaReadOnly();
    if (tries < 12) requestAnimationFrame(tick);
  };
  requestAnimationFrame(tick);
}

function patchPresetPreviewWidget(widget) {
  if (!widget || widget._llmPresetPreviewPatched) return;
  widget._llmPresetPreviewPatched = true;

  widget.serialize = false;
  widget.options = widget.options || {};
  widget.options.multiline = true;
  widget.options.readOnly = true;

  const originalCallback = widget.callback;
  widget.callback = function () {
    const r = originalCallback ? originalCallback.apply(this, arguments) : undefined;
    scheduleTrySetWidgetInputReadOnly(widget);
    scheduleTrySetActiveTextareaReadOnly();
    return r;
  };
}

function toThreeLinesText(text) {
  const s = String(text || "");
  const lines = s.split("\n");
  if (lines.length <= 3) return lines;
  const out = [lines[0], lines[1], lines[2]];
  out[2] = out[2].length ? `${out[2]}…` : "…";
  return out;
}

function makeField(labelText, inputEl) {
  const wrap = document.createElement("div");
  wrap.style.display = "flex";
  wrap.style.flexDirection = "column";
  wrap.style.gap = "6px";
  wrap.style.marginBottom = "10px";

  const label = document.createElement("div");
  label.textContent = labelText;
  label.style.fontSize = "12px";
  label.style.opacity = "0.9";
  wrap.appendChild(label);
  wrap.appendChild(inputEl);
  return wrap;
}

function ensureLlmModalStyles() {
  const id = "cxy-llm-openai-modal-style";
  if (document.getElementById(id)) return;
  const style = document.createElement("style");
  style.id = id;
  style.textContent = `
.cxy-llm-openai-modal input,
.cxy-llm-openai-modal textarea,
.cxy-llm-openai-modal select {
  box-sizing: border-box;
  width: 100%;
}
.cxy-llm-openai-modal select {
  color-scheme: dark;
}
.cxy-llm-openai-modal textarea {
  resize: vertical;
  line-height: 1.45;
}
.cxy-llm-openai-modal select option {
  color: #fff;
  background: #111;
}
`;
  document.head.appendChild(style);
}

function showModal(title, body, buttons = [], opts = {}) {
  ensureLlmModalStyles();
  const overlay = document.createElement("div");
  overlay.style.position = "fixed";
  overlay.style.left = "0";
  overlay.style.top = "0";
  overlay.style.right = "0";
  overlay.style.bottom = "0";
  overlay.style.background = "rgba(0,0,0,0.55)";
  overlay.style.zIndex = "9999";
  overlay.style.display = "flex";
  overlay.style.alignItems = "center";
  overlay.style.justifyContent = "center";

  const modal = document.createElement("div");
  modal.classList.add("cxy-llm-openai-modal");
  modal.style.width = opts && opts.width ? String(opts.width) : "min(720px, 92vw)";
  modal.style.maxHeight = "min(680px, 90vh)";
  modal.style.overflow = "hidden";
  modal.style.background = "var(--comfy-menu-bg, #222)";
  modal.style.color = "var(--fg-color, #ddd)";
  modal.style.border = "1px solid rgba(255,255,255,0.12)";
  modal.style.borderRadius = "8px";
  modal.style.display = "flex";
  modal.style.flexDirection = "column";

  const header = document.createElement("div");
  header.style.display = "flex";
  header.style.alignItems = "center";
  header.style.justifyContent = "space-between";
  header.style.padding = "12px 14px";
  header.style.borderBottom = "1px solid rgba(255,255,255,0.10)";

  const hTitle = document.createElement("div");
  hTitle.textContent = title;
  hTitle.style.fontSize = "14px";
  hTitle.style.fontWeight = "600";
  header.appendChild(hTitle);

  const closeBtn = document.createElement("button");
  closeBtn.textContent = "×";
  closeBtn.style.fontSize = "18px";
  closeBtn.style.width = "34px";
  closeBtn.style.height = "34px";
  closeBtn.style.borderRadius = "6px";
  closeBtn.style.border = "1px solid rgba(255,255,255,0.12)";
  closeBtn.style.background = "transparent";
  closeBtn.style.color = "inherit";
  closeBtn.style.cursor = "pointer";
  header.appendChild(closeBtn);

  const content = document.createElement("div");
  content.style.padding = "12px 14px";
  content.style.overflow = "auto";
  content.appendChild(body);

  const footer = document.createElement("div");
  footer.style.display = "flex";
  footer.style.justifyContent = "flex-end";
  footer.style.gap = "10px";
  footer.style.padding = "12px 14px";
  footer.style.borderTop = "1px solid rgba(255,255,255,0.10)";

  const destroy = () => {
    overlay.remove();
  };

  closeBtn.onclick = destroy;
  overlay.addEventListener("click", (e) => {
    if (e.target === overlay) destroy();
  });

  for (const btn of buttons) {
    const b = document.createElement("button");
    b.textContent = btn.text || "确定";
    b.style.border = "1px solid rgba(255,255,255,0.18)";
    b.style.borderRadius = "6px";
    b.style.padding = "8px 12px";
    b.style.background = "rgba(255,255,255,0.06)";
    b.style.color = "inherit";
    b.style.cursor = "pointer";
    b.onclick = async () => {
      try {
        const shouldClose = await (btn.onClick ? btn.onClick() : true);
        if (shouldClose !== false) destroy();
      } catch (e) {
        console.warn(e);
      }
    };
    footer.appendChild(b);
  }

  modal.appendChild(header);
  modal.appendChild(content);
  modal.appendChild(footer);
  overlay.appendChild(modal);
  document.body.appendChild(overlay);
}

async function openApiConfigDialog() {
  const cfg = await fetchJson(buildApiUrl(`${API_PREFIX}/config`));

  const baseInput = document.createElement("input");
  baseInput.type = "text";
  baseInput.value = cfg.base_url || "";
  baseInput.placeholder = "https://api.openai.com";
  baseInput.style.padding = "8px";
  baseInput.style.borderRadius = "6px";
  baseInput.style.border = "1px solid rgba(255,255,255,0.14)";
  baseInput.style.background = "rgba(255,255,255,0.04)";
  baseInput.style.color = "inherit";

  const keyInput = document.createElement("input");
  keyInput.type = "password";
  keyInput.value = "";
  keyInput.placeholder = cfg.has_api_key ? "（已配置）填入以更新" : "sk-...";
  keyInput.style.padding = "8px";
  keyInput.style.borderRadius = "6px";
  keyInput.style.border = "1px solid rgba(255,255,255,0.14)";
  keyInput.style.background = "rgba(255,255,255,0.04)";
  keyInput.style.color = "inherit";

  const body = document.createElement("div");
  body.appendChild(makeField("Base URL", baseInput));
  body.appendChild(makeField("API Key", keyInput));

  showModal("配置 API", body, [
    {
      text: "保存",
      onClick: async () => {
        const payload = {
          base_url: String(baseInput.value || "").trim(),
        };
        const key = String(keyInput.value || "").trim();
        if (key) payload.api_key = key;
        await fetchJson(buildApiUrl(`${API_PREFIX}/config`), {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(payload),
        });
        await refreshAllManagedNodes();
        return true;
      },
    },
    { text: "取消", onClick: async () => true },
  ]);
}

function normalizeModelLines(text) {
  const lines = String(text || "")
    .split("\n")
    .map((s) => s.trim())
    .filter(Boolean);
  const uniq = [];
  const set = new Set();
  for (const l of lines) {
    if (set.has(l)) continue;
    set.add(l);
    uniq.push(l);
  }
  return uniq;
}

async function openModelsDialog() {
  const cfg = await fetchJson(buildApiUrl(`${API_PREFIX}/config`));
  const models = (cfg.models || []).length ? cfg.models : ["gpt-4o-mini"];
  const ta = document.createElement("textarea");
  ta.value = models.join("\n");
  ta.placeholder = "每行一个模型名，例如 gpt-4o-mini";
  ta.style.minHeight = "220px";
  ta.style.padding = "8px";
  ta.style.borderRadius = "6px";
  ta.style.border = "1px solid rgba(255,255,255,0.14)";
  ta.style.background = "rgba(255,255,255,0.04)";
  ta.style.color = "inherit";

  const body = document.createElement("div");
  body.appendChild(makeField("模型列表", ta));

  showModal("管理模型", body, [
    {
      text: "保存",
      onClick: async () => {
        const modelList = normalizeModelLines(ta.value);
        await fetchJson(buildApiUrl(`${API_PREFIX}/config`), {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ models: modelList }),
        });
        await refreshAllManagedNodes();
        return true;
      },
    },
    { text: "取消", onClick: async () => true },
  ]);
}

function patchTextWidgetPlaceholder(node) {
  const w = (node.widgets || []).find((x) => x.name === "text");
  if (!w || w._llmTextPatched) return;
  w._llmTextPatched = true;
  const originalDraw = w.draw;
  w.draw = function (ctx, node, width, y, height) {
    const r = originalDraw ? originalDraw.apply(this, arguments) : undefined;
    try {
      if (!this.value) {
        ctx.save();
        ctx.fillStyle = "rgba(255,255,255,0.30)";
        ctx.font = "12px sans-serif";
        ctx.fillText(TEXT_PLACEHOLDER, 12, y + 18);
        ctx.restore();
      }
    } catch (e) {}
    return r;
  };
}

function ensureNumericDefaults(node) {
  const fix = (name, fallback) => {
    const w = (node.widgets || []).find((x) => x.name === name);
    if (!w) return;
    const v = w.value;
    if (v === null || v === undefined || v === "") {
      w.value = fallback;
    }
  };
  fix("temperature", 0.7);
  fix("max_tokens", 2048);
  fix("timeout", 30);
}

function ensurePresetPreviewWidget(node) {
  const configured = Boolean(node?.properties?._llmConfigured);
  if (configured) {
    for (let i = (node.widgets || []).length - 1; i >= 0; i--) {
      const w = node.widgets[i];
      if (w?.name === PRESET_PREVIEW_DISPLAY_OLD) {
        node.widgets.splice(i, 1);
      }
    }
  }

  const existing = (node.widgets || []).find((w) => w?.name === PRESET_PREVIEW_KEY || w?.name === PRESET_PREVIEW_KEY_OLD);
  if (!existing) return null;

  existing._llmPresetPreview = true;
  patchPresetPreviewWidget(existing);
  existing._llmHidden = false;
  clampWidgetHeight(existing, FIVE_LINE_HEIGHT);

  return existing;
}

async function updatePresetPreview(node) {
  const presetWidget = (node.widgets || []).find((w) => w.name === "preset");
  if (!presetWidget) return;

  const previewWidget = ensurePresetPreviewWidget(node);
  if (!previewWidget) return;
  const name = String(presetWidget.value || "").trim();
  if (!name || name === "无") {
    previewWidget._llmHidden = false;
    previewWidget.value = "";
    markDirty(node);
    return;
  }

  const p = await fetchPresetDetailByName(name);
  const prompt = p && typeof p.prompt === "string" ? p.prompt : "";
  previewWidget._llmHidden = false;
  previewWidget.value = prompt;
  markDirty(node);
}

async function openPresetsDialog() {
  const presets = await fetchJson(buildApiUrl(`${API_PREFIX}/presets`));
  const presetNames = (presets.presets || []).map((p) => p.name).filter(Boolean);

  let currentSelection = "";

  const presetSelect = document.createElement("select");
  presetSelect.style.padding = "8px";
  presetSelect.style.borderRadius = "6px";
  presetSelect.style.border = "1px solid rgba(255,255,255,0.14)";
  presetSelect.style.background = "rgba(255,255,255,0.04)";
  presetSelect.style.color = "inherit";
  presetSelect.style.width = "100%";

  const nameInput = document.createElement("input");
  nameInput.type = "text";
  nameInput.placeholder = "预设名";
  nameInput.style.padding = "8px";
  nameInput.style.borderRadius = "6px";
  nameInput.style.border = "1px solid rgba(255,255,255,0.14)";
  nameInput.style.background = "rgba(255,255,255,0.04)";
  nameInput.style.color = "inherit";
  nameInput.style.width = "100%";

  const promptInput = document.createElement("textarea");
  promptInput.placeholder = "提示词内容";
  promptInput.style.minHeight = "220px";
  promptInput.style.padding = "8px";
  promptInput.style.borderRadius = "6px";
  promptInput.style.border = "1px solid rgba(255,255,255,0.14)";
  promptInput.style.background = "rgba(255,255,255,0.04)";
  promptInput.style.color = "inherit";
  promptInput.style.width = "100%";
  promptInput.style.fontFamily = "ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, 'Liberation Mono', 'Courier New', monospace";

  async function renderSelect(selected) {
    const opts = [
      { value: PRESET_NEW_VALUE, label: "（新建）" },
      ...presetNames.map((x) => ({ value: x, label: x })),
    ];
    presetSelect.innerHTML = "";
    for (const o of opts) {
      const op = document.createElement("option");
      op.value = o.value;
      op.textContent = o.label;
      op.style.color = "#fff";
      op.style.background = "#111";
      presetSelect.appendChild(op);
    }
    presetSelect.value = selected || PRESET_NEW_VALUE;
    currentSelection = presetSelect.value;
  }

  async function applySelectionToInputs() {
    const val = String(presetSelect.value || "");
    currentSelection = val;
    if (val === PRESET_NEW_VALUE) {
      nameInput.value = "";
      promptInput.value = "";
      return;
    }
    nameInput.value = val;
    const p = await fetchPresetDetailByName(val);
    promptInput.value = p && typeof p.prompt === "string" ? p.prompt : "";
  }

  presetSelect.onchange = async () => {
    await applySelectionToInputs();
  };

  const saveBtn = document.createElement("button");
  saveBtn.textContent = "保存";
  saveBtn.style.border = "1px solid rgba(255,255,255,0.18)";
  saveBtn.style.borderRadius = "6px";
  saveBtn.style.padding = "8px 12px";
  saveBtn.style.background = "rgba(255,255,255,0.06)";
  saveBtn.style.color = "inherit";
  saveBtn.style.cursor = "pointer";

  const deleteBtn = document.createElement("button");
  deleteBtn.textContent = "删除";
  deleteBtn.style.border = "1px solid rgba(255,255,255,0.18)";
  deleteBtn.style.borderRadius = "6px";
  deleteBtn.style.padding = "8px 12px";
  deleteBtn.style.background = "rgba(255,255,255,0.06)";
  deleteBtn.style.color = "inherit";
  deleteBtn.style.cursor = "pointer";

  saveBtn.onclick = async () => {
    const name = String(nameInput.value || "").trim();
    const prompt = String(promptInput.value || "").trim();
    if (!name || !prompt) return;
    await fetchJson(buildApiUrl(`${API_PREFIX}/presets`), {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ name, prompt }),
    });
    presetDetailCache.delete(name);
    if (!presetNames.includes(name)) presetNames.push(name);
    presetNames.sort((a, b) => a.localeCompare(b, "zh"));
    await renderSelect(name);
    await applySelectionToInputs();
    await refreshAllManagedNodes();
  };

  deleteBtn.onclick = async () => {
    const oldName = String(currentSelection || "");
    if (!oldName || oldName === PRESET_NEW_VALUE) return;
    await fetchJson(buildApiUrl(`${API_PREFIX}/presets/${encodeURIComponent(oldName)}`), { method: "DELETE" });
    presetDetailCache.delete(oldName);
    const idx = presetNames.indexOf(oldName);
    if (idx >= 0) presetNames.splice(idx, 1);
    await renderSelect(undefined);
    await applySelectionToInputs();
    await refreshAllManagedNodes();
  };

  const btnRow = document.createElement("div");
  btnRow.style.display = "flex";
  btnRow.style.gap = "8px";
  btnRow.style.marginTop = "10px";
  btnRow.appendChild(saveBtn);
  btnRow.appendChild(deleteBtn);

  const body = document.createElement("div");
  body.appendChild(makeField("选择预设", presetSelect));
  body.appendChild(makeField("名字", nameInput));
  body.appendChild(promptInput);
  body.appendChild(btnRow);

  await renderSelect(undefined);
  await applySelectionToInputs();

  showModal("提示词预设管理", body, [{ text: "关闭", onClick: async () => true }]);
}

app.registerExtension({
  name: EXT_NAME,
  async beforeRegisterNodeDef(nodeType, nodeData) {
    const isLLMNode = nodeData.name === "ComfyUI_LLM_openai";
    if (!isLLMNode) return;

    const onNodeCreated = nodeType.prototype.onNodeCreated;
    nodeType.prototype.onNodeCreated = function () {
      const r = onNodeCreated ? onNodeCreated.apply(this, arguments) : undefined;
      if (!this.properties) this.properties = {};
      this.properties._comfyui_llm_openai = true;
      this.properties._llmConfigured = false;
      ensureDynamicImagePorts(this);
      refreshNodeCombos(this).catch(() => {});
      patchTextWidgetPlaceholder(this);
      ensureNumericDefaults(this);
      ensurePresetPreviewWidget(this);
      const presetWidget = (this.widgets || []).find((w) => w.name === "preset");
      if (presetWidget && !presetWidget._llmPreviewPatched) {
        presetWidget._llmPreviewPatched = true;
        const nodeRef = this;
        const originalCallback = presetWidget.callback;
        presetWidget.callback = function () {
          const rr = originalCallback ? originalCallback.apply(this, arguments) : undefined;
          updatePresetPreview(nodeRef).catch(() => {});
          return rr;
        };
      }
      updatePresetPreview(this).catch(() => {});
      return r;
    };

    const onConfigure = nodeType.prototype.onConfigure;
    nodeType.prototype.onConfigure = function () {
      const r = onConfigure ? onConfigure.apply(this, arguments) : undefined;
      try {
        if (!this.properties) this.properties = {};
        this.properties._llmConfigured = true;
        patchTextWidgetPlaceholder(this);
        ensureNumericDefaults(this);
        ensurePresetPreviewWidget(this);
        updatePresetPreview(this).catch(() => {});
        markDirty(this);
      } catch (e) {}
      return r;
    };

    const onConnectionsChange = nodeType.prototype.onConnectionsChange;
    nodeType.prototype.onConnectionsChange = function () {
      const r = onConnectionsChange ? onConnectionsChange.apply(this, arguments) : undefined;
      try {
        ensureDynamicImagePorts(this);
      } catch (e) {}
      return r;
    };

    const getExtraMenuOptions = nodeType.prototype.getExtraMenuOptions;
    nodeType.prototype.getExtraMenuOptions = function (_, options) {
      if (getExtraMenuOptions) {
        getExtraMenuOptions.apply(this, arguments);
      }
      options.push(
        {
          content: "配置 API",
          callback: () => openApiConfigDialog().catch(() => {}),
        },
        {
          content: "管理模型",
          callback: () => openModelsDialog().catch(() => {}),
        },
        {
          content: "管理预设",
          callback: () => openPresetsDialog().catch(() => {}),
        },
        {
          content: "刷新下拉选项",
          callback: () => refreshNodeCombos(this).catch(() => {}),
        }
      );
    };
  },
});

