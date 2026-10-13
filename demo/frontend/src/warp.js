import Delaunator from 'delaunator';
import { fitAffine } from './importance';

// Кусочно-аффинный warp правого изображения в координаты левого:
// триангуляция Делоне по «зелёным» парам (деформируемость ≤ GREEN_MAX_NORM),
// для каждого треугольника — аффин B→A, рендер через clip + setTransform.

export const GREEN_MAX_NORM = 0.5;

// Индексы matches с наибольшей деформируемостью, которые нужно не учитывать
// при наложении (связи — по индексу пары, для равных значений — меньший индекс).
export function excludedPairSet(matches, importance, excludeN = 0) {
  const n = Math.max(0, Math.floor(excludeN || 0));
  if (!n || !importance || importance.values.length !== matches.length) return null;

  const order = [];
  for (let i = 0; i < matches.length; i++) {
    if (Number.isFinite(importance.values[i])) order.push(i);
  }
  if (order.length > n) {
    order.sort((a, b) => importance.values[b] - importance.values[a] || a - b);
    return new Set(order.slice(0, n));
  }
  return new Set(order);                 // исключаем всё — фильтр «зелёных» отсечёт остальное
}

// Пары с конечной деформируемостью и norm ≤ 0.5, не попавшие в excludeN.
// null — если деформируемость не готова или зелёных пар меньше 3
// (не из чего собрать треугольник).
export function selectGreenPairs(matches, kpsA, kpsB, importance, excludeN = 0) {
  if (!importance || importance.values.length !== matches.length) return null;
  const skip = excludedPairSet(matches, importance, excludeN);
  const left = [], right = [];
  matches.forEach((m, i) => {
    if (skip && skip.has(i)) return;
    const v = importance.values[i];
    const n = importance.norms[i];
    if (!Number.isFinite(v) || !Number.isFinite(n) || n > GREEN_MAX_NORM) return;
    const a = kpsA[m.aIdx], b = kpsB[m.bIdx];
    if (!a || !b) return;
    left.push(a);
    right.push(b);
  });
  return left.length >= 3 ? { left, right } : null;
}

// Сколько пар проходит фильтр «зелёных» (для тултипов/кнопок).
export function greenPairCount(matches, importance, excludeN = 0) {
  if (!importance || importance.values.length !== matches.length) return 0;
  const skip = excludedPairSet(matches, importance, excludeN);
  let k = 0;
  for (let i = 0; i < matches.length; i++) {
    if (skip && skip.has(i)) continue;
    const v = importance.values[i], n = importance.norms[i];
    if (Number.isFinite(v) && Number.isFinite(n) && n <= GREEN_MAX_NORM) k++;
  }
  return k;
}

// Аффин src → dst по трём точкам (x' = a·x + c·y + e, y' = b·x + d·y + f).
// null — вырожденный треугольник.
function affineFromTri(src, dst) {
  const [s0, s1, s2] = src;
  const [d0, d1, d2] = dst;
  const v1x = s1.x - s0.x, v1y = s1.y - s0.y;
  const v2x = s2.x - s0.x, v2y = s2.y - s0.y;
  const det = v1x * v2y - v2x * v1y;
  if (!Number.isFinite(det) || Math.abs(det) < 1e-6) return null;

  const w1x = d1.x - d0.x, w1y = d1.y - d0.y;
  const w2x = d2.x - d0.x, w2y = d2.y - d0.y;

  const a = (w1x * v2y - w2x * v1y) / det;
  const c = (w2x * v1x - w1x * v2x) / det;
  const b = (w1y * v2y - w2y * v1y) / det;
  const d = (w2y * v1x - w1y * v2x) / det;
  const e = d0.x - (a * s0.x + c * s0.y);
  const f = d0.y - (b * s0.x + d * s0.y);
  if (![a, b, c, d, e, f].every(Number.isFinite)) return null;
  return { a, b, c, d, e, f };
}

// Углы кадра B, отображённые общим аффином B→A, — чтобы warp покрывал
// всё изображение, а не только выпуклую оболочку точек.
function withCorners(left, right, imgB) {
  const fit = fitAffine(right, left);      // right(B) → left(A)
  if (!fit || !imgB) return { left, right };
  const w = imgB.width, h = imgB.height;
  const bc = [{ x: 0, y: 0 }, { x: w, y: 0 }, { x: 0, y: h }, { x: w, y: h }];
  const ac = bc.map(fit);
  for (const p of ac) if (!Number.isFinite(p.x) || !Number.isFinite(p.y)) return { left, right };
  return { left: [...left, ...ac], right: [...right, ...bc] };
}

// Расширение треугольника от центроида — иначе на стыках соседних
// треугольников из-за сглаживания остаются волосные швы.
function expandTri(tri, px = 0.75) {
  const cx = (tri[0].x + tri[1].x + tri[2].x) / 3;
  const cy = (tri[0].y + tri[1].y + tri[2].y) / 3;
  return tri.map((p) => {
    const dx = p.x - cx, dy = p.y - cy;
    const len = Math.hypot(dx, dy);
    if (len < 1e-6) return p;
    return { x: p.x + (dx / len) * px, y: p.y + (dy / len) * px };
  });
}

// Деформирует правое изображение в координаты левого.
// Возвращает canvas размера imgA (без фона) или null при ошибке.
export function warpRightToLeft(imgA, imgB, left, right) {
  if (!imgA || !imgB || !imgA.el || !imgB.el) return null;
  if (left.length < 3 || left.length !== right.length) return null;

  const out = document.createElement('canvas');
  out.width = imgA.width;
  out.height = imgA.height;
  const ctx = out.getContext('2d');
  if (!ctx) return null;

  const { left: L, right: R } = withCorners(left, right, imgB);

  let tris;
  try {
    tris = Delaunator.from(R, p => p.x, p => p.y).triangles;
  } catch (e) {
    console.error('Триангуляция не удалась:', e);
    return null;
  }

  for (let i = 0; i < tris.length; i += 3) {
    const src = [R[tris[i]], R[tris[i + 1]], R[tris[i + 2]]];
    const dst = [L[tris[i]], L[tris[i + 1]], L[tris[i + 2]]];
    const m = affineFromTri(src, dst);
    if (!m) continue;

    const clip = expandTri(dst);
    ctx.save();
    ctx.beginPath();
    ctx.moveTo(clip[0].x, clip[0].y);
    ctx.lineTo(clip[1].x, clip[1].y);
    ctx.lineTo(clip[2].x, clip[2].y);
    ctx.closePath();
    ctx.clip();
    ctx.setTransform(m.a, m.b, m.c, m.d, m.e, m.f);
    ctx.drawImage(imgB.el, 0, 0);
    ctx.restore();
  }
  return out;
}

// Композит для сохранения: левое изображение + деформированное правое.
export function compositeOverlay(imgA, warped, alpha) {
  const out = document.createElement('canvas');
  out.width = imgA.width;
  out.height = imgA.height;
  const ctx = out.getContext('2d');
  ctx.drawImage(imgA.el, 0, 0);
  ctx.globalAlpha = Math.max(0, Math.min(1, alpha || 0));
  ctx.drawImage(warped, 0, 0);
  ctx.globalAlpha = 1;
  return out;
}

export function downloadCanvas(canvas, name) {
  canvas.toBlob((blob) => {
    if (!blob) { console.error('Не удалось получить PNG'); return; }
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = name;
    document.body.appendChild(a);
    a.click();
    a.remove();
    setTimeout(() => URL.revokeObjectURL(url), 1000);
  }, 'image/png');
}
