import { app } from "../../../scripts/app.js";

const EXT_NAME = "comfyui_cxy_tools.image_compare";

function markDirty(node) {
  if (node?.setDirtyCanvas) {
    node.setDirtyCanvas(true, true);
    return;
  }
  if (app?.graph?.setDirtyCanvas) {
    app.graph.setDirtyCanvas(true, true);
  }
}

function buildViewUrlFromMeta(meta) {
  const base = window.location.origin;
  const filename = encodeURIComponent(String(meta?.filename || ""));
  const subfolder = encodeURIComponent(String(meta?.subfolder || ""));
  const type = encodeURIComponent(String(meta?.type || "temp"));
  return `${base}/view?filename=${filename}&subfolder=${subfolder}&type=${type}`;
}

function clamp01(x) {
  const v = Number(x);
  if (!Number.isFinite(v)) return 0;
  return Math.max(0, Math.min(1, v));
}

function getMouseRatioInRect(node, x0, w) {
  const canvas = app?.canvas;
  if (!canvas) return null;
  if (canvas.node_over !== node) return null;
  const gm = canvas.graph_mouse;
  if (!gm || gm.length < 2) return null;
  const x = gm[0] - node.pos[0] - Number(x0 || 0);
  return clamp01(x / Math.max(1, Number(w) || 1));
}

function fitContain(srcW, srcH, dstW, dstH) {
  const sw = Math.max(1, Number(srcW));
  const sh = Math.max(1, Number(srcH));
  const dw = Math.max(1, Number(dstW));
  const dh = Math.max(1, Number(dstH));
  const s = Math.min(dw / sw, dh / sh);
  const w = sw * s;
  const h = sh * s;
  const x = (dw - w) / 2;
  const y = (dh - h) / 2;
  return { x, y, w, h };
}

function ensureCompareState(node) {
  if (!node.properties) node.properties = {};
  if (!node.properties._imageCompare) {
    node.properties._imageCompare = node.properties._cxyCompare || {};
  }
  return node.properties._imageCompare;
}

function installCompareNode(node) {
  const st = ensureCompareState(node);
  if (st._installed) return;
  st._installed = true;
  if (!Array.isArray(node.size) || node.size.length < 2) {
    node.size = [320, 260];
  } else {
    node.size[0] = Math.max(node.size[0], 320);
    node.size[1] = Math.max(node.size[1], 260);
  }

  const invertWidget = (node.widgets || []).find((w) => w?.name === "反转");
  if (invertWidget && !invertWidget._imageComparePatched) {
    invertWidget._imageComparePatched = true;
    const original = invertWidget.callback;
    invertWidget.callback = function () {
      const r = original ? original.apply(this, arguments) : undefined;
      markDirty(node);
      return r;
    };
  }
}

function applyCompareUi(node, ui) {
  const st = ensureCompareState(node);
  const imgs = ui?.cxy_compare?.images || ui?.images;
  if (!Array.isArray(imgs) || imgs.length < 2) return;
  st._imagesMeta = imgs.slice(0, 2);

  const a = new Image();
  const b = new Image();
  a.crossOrigin = "anonymous";
  b.crossOrigin = "anonymous";
  a.onload = () => markDirty(node);
  b.onload = () => markDirty(node);
  a.src = buildViewUrlFromMeta(st._imagesMeta[0]);
  b.src = buildViewUrlFromMeta(st._imagesMeta[1]);
  st._imgA = a;
  st._imgB = b;
}

function drawCompareOnNode(node, ctx) {
  const st = node?.properties?._imageCompare || node?.properties?._cxyCompare;
  const imgA = st?._imgA;
  const imgB = st?._imgB;
  if (!imgA || !imgB || !imgA.complete || !imgB.complete) return;

  const pad = 8;
  const headerH = 32;
  const widgetH = 26;
  const w = node.size[0] - pad * 2;
  const h = node.size[1] - headerH - widgetH - pad * 2;
  if (w <= 8 || h <= 8) return;

  const x0 = pad;
  const y0 = headerH + widgetH + pad;

  const invWidget = (node.widgets || []).find((ww) => ww?.name === "反转");
  const invert = Boolean(invWidget ? invWidget.value : false);

  const ratio = getMouseRatioInRect(node, x0, w);
  const r = ratio === null ? 0.5 : ratio;

  ctx.save();
  ctx.beginPath();
  ctx.rect(x0, y0, w, h);
  ctx.clip();

  const leftImg = invert ? imgB : imgA;
  const rightImg = invert ? imgA : imgB;

  const baseFit = fitContain(rightImg.width, rightImg.height, w, h);
  ctx.drawImage(rightImg, x0 + baseFit.x, y0 + baseFit.y, baseFit.w, baseFit.h);

  const lineX = x0 + w * clamp01(r);
  ctx.save();
  ctx.beginPath();
  ctx.rect(x0, y0, Math.max(0, lineX - x0), h);
  ctx.clip();
  const topFit = fitContain(leftImg.width, leftImg.height, w, h);
  ctx.drawImage(leftImg, x0 + topFit.x, y0 + topFit.y, topFit.w, topFit.h);
  ctx.restore();

  ctx.strokeStyle = "rgba(255,255,255,0.65)";
  ctx.lineWidth = 1;
  ctx.beginPath();
  ctx.moveTo(lineX, y0);
  ctx.lineTo(lineX, y0 + h);
  ctx.stroke();

  ctx.restore();

  if (app?.canvas?.node_over === node) {
    markDirty(node);
  }
}

app.registerExtension({
  name: EXT_NAME,
  async beforeRegisterNodeDef(nodeType, nodeData) {
    const isCompareNode = nodeData.name === "Image_Compare";
    if (!isCompareNode) return;

    const onNodeCreated = nodeType.prototype.onNodeCreated;
    nodeType.prototype.onNodeCreated = function () {
      const r = onNodeCreated ? onNodeCreated.apply(this, arguments) : undefined;
      try {
        installCompareNode(this);
      } catch (e) {}
      return r;
    };

    nodeType.prototype.onExecuted = function () {
      try {
        const msg = arguments && arguments.length ? arguments[0] : null;
        const ui = msg?.ui || msg;
        installCompareNode(this);
        applyCompareUi(this, ui);
        if (Array.isArray(this.imgs)) this.imgs.length = 0;
        this.image = null;
        markDirty(this);
      } catch (e) {}
    };

    const onDrawBackground = nodeType.prototype.onDrawBackground;
    nodeType.prototype.onDrawBackground = function (ctx) {
      try {
        drawCompareOnNode(this, ctx);
      } catch (e) {}
    };

    const onConfigure = nodeType.prototype.onConfigure;
    nodeType.prototype.onConfigure = function () {
      const r = onConfigure ? onConfigure.apply(this, arguments) : undefined;
      try {
        installCompareNode(this);
        markDirty(this);
      } catch (e) {}
      return r;
    };
  },
});

