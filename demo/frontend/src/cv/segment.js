// Инстанс-сегментация через бэкенд (POST /api/segment, Mask2Former).

function loadPngDataUrl(src) {
  return new Promise((resolve, reject) => {
    const el = new Image();
    el.onload = () => resolve(el);
    el.onerror = () => reject(new Error('Не удалось декодировать оверлей масок'));
    el.src = src;
  });
}

// Возвращает { el, instances, width, height, model, elapsedMs }.
// el — HTMLImageElement с RGBA-PNG, alpha заложен в канал.
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
  return {
    el,
    instances: d.instances || [],
    width: d.width,
    height: d.height,
    model: d.model,
    elapsedMs: d.elapsed_ms,
  };
}
