// Сравнение с эталоном через бэкенд (POST /api/similarity):
// начальное изображение и наложенное (warp) против выбранного эталона.

const strip = (u) => {
  if (!u) throw new Error('Нет данных изображения');
  return u.includes(',') ? u.split(',', 2)[1] : u;
};

// ref/initial/warped — dataURL изображений.
// Возвращает { ref_size, compared_size, initial, warped, elapsed_ms }.
export async function fetchSimilarity(ref, initial, warped, { tile } = {}) {
  const body = {
    image_ref: { image_b64: strip(ref) },
    image_initial: { image_b64: strip(initial) },
    image_warped: { image_b64: strip(warped) },
  };
  if (tile != null) body.tile = tile;

  const r = await fetch('/api/similarity', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  });
  if (!r.ok) {
    const text = await r.text().catch(() => '');
    throw new Error(`similarity: ${r.status} ${text}`);
  }
  return r.json();
}
