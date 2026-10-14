// Отбор точек после Mask2Former: оставляем точки на границах областей
// или очень важные для деформации (нормированная деформируемость ≥ impThr).
// Фильтр визуальный: возвращаем множества скрытых индексов, данные не трогаем.
//
// Маски считаются только для правого изображения B, поэтому:
//  - пара остаётся целиком, если точка B прошла отбор;
//  - непарная точка B остаётся, если она на границе области;
//  - непарные точки A при включённом фильтре скрываются.

import { nearBoundary } from './cv/segment.js';

// Возвращает { hiddenA, hiddenB, keptPairs, totalPairs } или null,
// если нет карты границ / изображения B.
export function computeHiddenSets({
  kpsA, kpsB, matches, importance, boundary, imgB,
  impThr = 0.5, radius = 4,
  classKeep = null, geometryOn = true,
}) {
  if (!boundary || !imgB || !kpsA || !kpsB) return null;

  const impOk = !!(importance && importance.values.length === matches.length);
  const keptB = new Set();

  matches.forEach((m, i) => {
    const kp = kpsB[m.bIdx];
    if (!kp) return;
    const onEdge = nearBoundary(boundary, kp, imgB, radius);
    const imp = impOk && Number.isFinite(importance.norms[i])
      ? importance.norms[i] >= impThr
      : false;
    const allowed = !classKeep || classKeep[m.bIdx];
    if (allowed && (!geometryOn || onEdge || imp)) keptB.add(m.bIdx);
  });

  const pairedB = new Set(matches.map(m => m.bIdx));
  kpsB.forEach((kp, i) => {
    if (!pairedB.has(i) && (!classKeep || classKeep[i])
        && (!geometryOn || nearBoundary(boundary, kp, imgB, radius))) keptB.add(i);
  });

  const hiddenB = new Set();
  kpsB.forEach((_, i) => { if (!keptB.has(i)) hiddenB.add(i); });

  const keptA = new Set();
  matches.forEach(m => { if (keptB.has(m.bIdx)) keptA.add(m.aIdx); });
  const hiddenA = new Set();
  if (geometryOn) {
    kpsA.forEach((_, i) => { if (!keptA.has(i)) hiddenA.add(i); });
  } else {
    matches.forEach(m => { if (!keptB.has(m.bIdx)) hiddenA.add(m.aIdx); });
  }

  let keptPairs = 0;
  matches.forEach(m => { if (keptB.has(m.bIdx)) keptPairs++; });

  return { hiddenA, hiddenB, keptPairs, totalPairs: matches.length };
}
