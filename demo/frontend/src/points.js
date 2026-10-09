// Чистые функции для редактирования точек и пар.
// matches: [{ aIdx, bIdx, dist }] — индексы в kpsA / kpsB.

export const clampKp = (kp, img) => ({
  x: Math.max(0, Math.min(img.width, kp.x)),
  y: Math.max(0, Math.min(img.height, kp.y)),
});

export const addPoint = (kps, kp) => [...kps, kp];

export function removePoint(kps, matches, side, idx) {
  const newKps = kps.filter((_, i) => i !== idx);
  const remap = (i) => (i < idx ? i : i - 1);
  const newMatches = matches
    .filter(m => (side === 'a' ? m.aIdx : m.bIdx) !== idx)
    .map(m => side === 'a' ? { ...m, aIdx: remap(m.aIdx) }
                           : { ...m, bIdx: remap(m.bIdx) });
  return { kps: newKps, matches: newMatches };
}

// Свяжет aIdx с bIdx; обе точки предварительно освобождаются от старых пар.
export function setPair(matches, aIdx, bIdx) {
  const freed = matches.filter(m => m.aIdx !== aIdx && m.bIdx !== bIdx);
  return [...freed, { aIdx, bIdx, dist: null }];
}

export const findPair = (matches, side, idx) =>
  matches.find(m => (side === 'a' ? m.aIdx : m.bIdx) === idx) || null;

export const unpair = (matches, side, idx) =>
  matches.filter(m => (side === 'a' ? m.aIdx : m.bIdx) !== idx);

// Непарные точки: индексы, не входящие ни в одну пару.
export function unmatchedIdx(kps, matches, side) {
  const used = new Set(matches.map(m => (side === 'a' ? m.aIdx : m.bIdx)));
  const out = [];
  for (let i = 0; i < kps.length; i++) if (!used.has(i)) out.push(i);
  return out;
}
