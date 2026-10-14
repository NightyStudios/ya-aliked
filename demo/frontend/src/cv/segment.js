// Семантическая сегментация через бэкенд (POST /api/segment, Mask2Former).

function loadPngDataUrl(src) {
  return new Promise((resolve, reject) => {
    const el = new Image();
    el.onload = () => resolve(el);
    el.onerror = () => reject(new Error('Не удалось декодировать оверлей масок'));
    el.src = src;
  });
}

// Декод серого PNG карты границ в ImageData для попиксельного опроса.
async function loadBoundary(src) {
  const el = await loadPngDataUrl(src);
  const c = document.createElement('canvas');
  c.width = el.width;
  c.height = el.height;
  const ctx = c.getContext('2d', { willReadFrequently: true });
  if (!ctx) throw new Error('Нет доступа к canvas для карты границ');
  ctx.drawImage(el, 0, 0);
  const img = ctx.getImageData(0, 0, c.width, c.height);
  return { data: img.data, width: c.width, height: c.height };
}

// Попадает ли точка (координаты исходного кадра) в окно радиуса radius px
// вокруг пикселей карты границ. Карта меньше кадра — координаты масштабируются.
export function nearBoundary(boundary, kp, img, radius = 4) {
  if (!boundary || !boundary.width || !kp || !img || !img.width) return false;
  const sx = boundary.width / img.width;
  const sy = boundary.height / img.height;
  const rx = Math.max(1, Math.round(radius * sx));
  const ry = Math.max(1, Math.round(radius * sy));
  const px = Math.round(kp.x * sx);
  const py = Math.round(kp.y * sy);
  const x0 = Math.max(0, px - rx), x1 = Math.min(boundary.width - 1, px + rx);
  const y0 = Math.max(0, py - ry), y1 = Math.min(boundary.height - 1, py + ry);
  if (x0 > x1 || y0 > y1) return false;
  const d = boundary.data;
  for (let y = y0; y <= y1; y++) {
    const row = y * boundary.width;
    for (let x = x0; x <= x1; x++) {
      if (d[(row + x) * 4] > 127) return true;
    }
  }
  return false;
}

// Возвращает { el, boundary, instances, width, height, model, elapsedMs }.
// el — HTMLImageElement с RGBA-PNG, alpha заложен в канал.
// boundary — ImageData-карта границ областей (серый PNG, 255 = граница).
export async function fetchSegmentation(img, { threshold } = {}) {
  if (!img || !img.dataUrl) throw new Error('Нет данных правого изображения');

  const body = {
    image: { image_b64: img.dataUrl.split(',', 2)[1] || img.dataUrl },
  };
  if (threshold != null) body.threshold = threshold;

  const r = await fetch('/api/segment', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  });
  if (!r.ok) {
    const text = await r.text().catch(() => '');
    throw new Error(`segment: ${r.status} ${text}`);
  }

  const d = await r.json();
  const el = await loadPngDataUrl('data:image/png;base64,' + d.overlay_png_b64);
  const boundary = d.boundary_png_b64
    ? await loadBoundary('data:image/png;base64,' + d.boundary_png_b64)
    : null;
  return {
    el,
    boundary,
    classMapPng: d.class_map_png_b64 || null,
    classes: d.classes || [],
    instances: d.instances || [],
    width: d.width,
    height: d.height,
    model: d.model,
    elapsedMs: d.elapsed_ms,
  };
}
