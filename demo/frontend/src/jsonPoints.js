// Загрузка/сохранение точек в формате:
// {
//   "image_a": "name.jpg", "image_b": "name.jpg",
//   "pairs": [{ "a": [x, y], "b": [x, y] }],
//   "unmatched_a": [[x, y]], "unmatched_b": [[x, y]]
// }

const isNum = (v) => typeof v === 'number' && Number.isFinite(v);

const toXY = (v) => {
  if (Array.isArray(v) && v.length >= 2 && isNum(v[0]) && isNum(v[1])) {
    return { x: v[0], y: v[1] };
  }
  if (v && typeof v === 'object' && isNum(v.x) && isNum(v.y)) {
    return { x: v.x, y: v.y };
  }
  return null;
};

const toPair = (v) => {
  if (!v || typeof v !== 'object') return null;
  const a = toXY(v.a), b = toXY(v.b);
  return a && b ? { a, b } : null;
};

export function parsePointsJson(text) {
  let data;
  try { data = JSON.parse(text); }
  catch (e) { throw new Error('Некорректный JSON: ' + e.message); }

  if (!data || typeof data !== 'object' || Array.isArray(data)) {
    throw new Error('Ожидался объект в корне файла');
  }

  const rawPairs = data.pairs ?? [];
  const rawUA = data.unmatched_a ?? [];
  const rawUB = data.unmatched_b ?? [];
  if (!Array.isArray(rawPairs) || !Array.isArray(rawUA) || !Array.isArray(rawUB)) {
    throw new Error('pairs, unmatched_a, unmatched_b должны быть массивами');
  }

  const pairs = [];
  rawPairs.forEach((p, i) => {
    const pr = toPair(p);
    if (!pr) throw new Error(`pairs[${i}]: неверный формат пары`);
    pairs.push(pr);
  });

  const ua = [], ub = [];
  rawUA.forEach((p, i) => {
    const kp = toXY(p);
    if (!kp) throw new Error(`unmatched_a[${i}]: неверный формат точки`);
    ua.push(kp);
  });
  rawUB.forEach((p, i) => {
    const kp = toXY(p);
    if (!kp) throw new Error(`unmatched_b[${i}]: неверный формат точки`);
    ub.push(kp);
  });

  // kpsA = точки пар (левые) + непарные; kpsB — аналогично.
  const kpsA = [...pairs.map(p => p.a), ...ua];
  const kpsB = [...pairs.map(p => p.b), ...ub];
  const matches = pairs.map((_, i) => ({ aIdx: i, bIdx: i, dist: null }));

  return {
    kpsA, kpsB, matches,
    imageA: typeof data.image_a === 'string' ? data.image_a : null,
    imageB: typeof data.image_b === 'string' ? data.image_b : null,
    counts: { pairs: pairs.length, ua: ua.length, ub: ub.length },
  };
}

export function serializePointsJson({ kpsA, kpsB, matches, imageA, imageB }) {
  const valid = matches.filter(m =>
    kpsA[m.aIdx] != null && kpsB[m.bIdx] != null);
  const usedA = new Set(valid.map(m => m.aIdx));
  const usedB = new Set(valid.map(m => m.bIdx));

  const pairs = valid.map(m => ({
    a: [kpsA[m.aIdx].x, kpsA[m.aIdx].y],
    b: [kpsB[m.bIdx].x, kpsB[m.bIdx].y],
  }));
  const unmatched_a = kpsA.filter((_, i) => !usedA.has(i)).map(k => [k.x, k.y]);
  const unmatched_b = kpsB.filter((_, i) => !usedB.has(i)).map(k => [k.x, k.y]);

  return JSON.stringify({
    image_a: imageA ?? null,
    image_b: imageB ?? null,
    pairs,
    unmatched_a,
    unmatched_b,
  }, null, 2);
}

export function downloadJson(text, filename) {
  const blob = new Blob([text], { type: 'application/json' });
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = filename;
  document.body.appendChild(a);
  a.click();
  a.remove();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}
