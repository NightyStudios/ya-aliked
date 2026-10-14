import Delaunator from 'delaunator';

// Деформируемость пары = насколько её правая точка смещается, если триангулировать
// левые точки БЕЗ неё и интерполировать её позицию L→R по чужим треугольникам.
//
// В триангуляцию дополнительно включаются 4 угловые точки кадра A — они
// расширяют выпуклую оболочку до прямоугольника изображения, чтобы
// пограничные точки интерполировались, а не экстраполировались через
// произвольный ближайший треугольник. Правые позиции углов — аффинное
// преобразование L→R, оцененное по всем парам (МНК).

const AREA_EPS = 1e-9;
const INSIDE_EPS = -1e-9;

// Барицентрические веса [w0,w1,w2] точки p в треугольнике abc; null — вырожден.
function barycentric(px, py, ax, ay, bx, by, cx, cy) {
  const v0x = bx - ax, v0y = by - ay;
  const v1x = cx - ax, v1y = cy - ay;
  const v2x = px - ax, v2y = py - ay;
  const den = v0x * v1y - v1x * v0y;
  if (Math.abs(den) < AREA_EPS) return null;
  const w1 = (v2x * v1y - v1x * v2y) / den;
  const w2 = (v0x * v2y - v2x * v0y) / den;
  return [1 - w1 - w2, w1, w2];
}

// Ближайшая точка треугольника (Ericson, Real-Time Collision Detection).
function closestOnTriangle(px, py, ax, ay, bx, by, cx, cy) {
  const abx = bx - ax, aby = by - ay;
  const acx = cx - ax, acy = cy - ay;
  const apx = px - ax, apy = py - ay;
  const d1 = abx * apx + aby * apy, d2 = acx * apx + acy * apy;
  if (d1 <= 0 && d2 <= 0) return [ax, ay];

  const bpx = px - bx, bpy = py - by;
  const d3 = abx * bpx + aby * bpy, d4 = acx * bpx + acy * bpy;
  if (d3 >= 0 && d4 <= d3) return [bx, by];

  const vc = d1 * d4 - d3 * d2;
  if (vc <= 0 && d1 >= 0 && d3 <= 0) {
    const v = d1 / (d1 - d3);
    return [ax + abx * v, ay + aby * v];
  }

  const cpx = px - cx, cpy = py - cy;
  const d5 = abx * cpx + aby * cpy, d6 = acx * cpx + acy * cpy;
  if (d6 >= 0 && d5 <= d6) return [cx, cy];

  const vb = d5 * d2 - d1 * d6;
  if (vb <= 0 && d2 >= 0 && d6 <= 0) {
    const w = d2 / (d2 - d6);
    return [ax + acx * w, ay + acy * w];
  }

  const va = d3 * d6 - d5 * d4;
  if (va <= 0 && d4 - d3 >= 0 && d5 - d6 >= 0) {
    const w = (d4 - d3) / ((d4 - d3) + (d5 - d6));
    return [bx + (cx - bx) * w, by + (cy - by) * w];
  }

  const denom = 1 / (va + vb + vc);
  const v = vb * denom, w = vc * denom;
  return [ax + abx * v + acx * w, ay + aby * v + acy * w];
}

// Аффинное преобразование L→R, оцененное методом наименьших квадратов.
// Центрированные координаты — иначе матрица 3×3 плохо обусловлена на
// больших кадрах. Одинаковые коэффициенты решаются для x_B и y_B.
// null — пар меньше 3 или данные вырождены (например, коллинеарны).
export function fitAffine(left, right) {
  const n = left.length;
  if (n < 3 || right.length !== n) return null;

  let cx = 0, cy = 0, cX = 0, cY = 0;
  for (let i = 0; i < n; i++) {
    cx += left[i].x; cy += left[i].y;
    cX += right[i].x; cY += right[i].y;
  }
  cx /= n; cy /= n; cX /= n; cY /= n;

  let sxx = 0, sxy = 0, syy = 0;
  let sxu = 0, syu = 0, sxv = 0, syv = 0;
  for (let i = 0; i < n; i++) {
    const dx = left[i].x - cx, dy = left[i].y - cy;
    const u = right[i].x - cX, v = right[i].y - cY;
    sxx += dx * dx; sxy += dx * dy; syy += dy * dy;
    sxu += dx * u; syu += dy * u;
    sxv += dx * v; syv += dy * v;
  }

  const det = sxx * syy - sxy * sxy;
  if (!Number.isFinite(det) || Math.abs(det) < 1e-9) return null;

  const a = (syy * sxu - sxy * syu) / det;
  const b = (sxx * syu - sxy * sxu) / det;
  const c = (syy * sxv - sxy * syv) / det;
  const d = (sxx * syv - sxy * sxv) / det;
  if (![a, b, c, d].every(Number.isFinite)) return null;

  return (p) => ({
    x: cX + a * (p.x - cx) + b * (p.y - cy),
    y: cY + c * (p.x - cx) + d * (p.y - cy),
  });
}

// Углы кадра A + их правые позиции (через аффинный fit). null — без углов.
export function buildCorners(left, right, img) {
  if (!img || !Number.isFinite(img.width) || !Number.isFinite(img.height)) return null;
  const fit = fitAffine(left, right);
  if (!fit) return null;
  const w = img.width, h = img.height;
  const L = [{ x: 0, y: 0 }, { x: w, y: 0 }, { x: 0, y: h }, { x: w, y: h }];
  return { L, R: L.map(fit) };
}

// Вершины треугольника: данные (0..M-1) и углы (M..M+C-1).
function vertexLeft(id, M, left, corners) {
  return id < M ? left[id] : corners.L[id - M];
}
function vertexRight(id, M, right, corners) {
  return id < M ? right[id] : corners.R[id - M];
}

// Триангуляция left без точки i → интерполяция right[i] в left[i].
// corners (может быть null) — вечные вершины, не участвующие в leave-one-out.
function valueWithout(left, right, i, corners) {
  const M = left.length;
  const C = corners ? corners.L.length : 0;
  const coords = new Float64Array((M - 1 + C) * 2);
  const orig = new Int32Array(M - 1 + C);   // reduced-idx -> vertex-id
  let k = 0;
  for (let j = 0; j < M; j++) {
    if (j === i) continue;
    coords[k * 2] = left[j].x;
    coords[k * 2 + 1] = left[j].y;
    orig[k] = j;
    k++;
  }
  for (let c = 0; c < C; c++) {
    coords[k * 2] = corners.L[c].x;
    coords[k * 2 + 1] = corners.L[c].y;
    orig[k] = M + c;
    k++;
  }

  let tri;
  try { tri = new Delaunator(coords); }
  catch { return NaN; }

  const t = tri.triangles;
  const p = left[i];
  let bestDist2 = Infinity;
  let bestW = null, bestI0 = -1, bestI1 = -1, bestI2 = -1;

  for (let k = 0; k < t.length; k += 3) {
    const r0 = orig[t[k]], r1 = orig[t[k + 1]], r2 = orig[t[k + 2]];
    const a = vertexLeft(r0, M, left, corners);
    const b = vertexLeft(r1, M, left, corners);
    const c = vertexLeft(r2, M, left, corners);
    const w = barycentric(p.x, p.y, a.x, a.y, b.x, b.y, c.x, c.y);
    if (!w) continue;                            // вырожденный треугольник

    if (w[0] >= INSIDE_EPS && w[1] >= INSIDE_EPS && w[2] >= INSIDE_EPS) {
      // точка внутри — интерполяция по ней самой
      const rr0 = vertexRight(r0, M, right, corners);
      const rr1 = vertexRight(r1, M, right, corners);
      const rr2 = vertexRight(r2, M, right, corners);
      const ix = w[0] * rr0.x + w[1] * rr1.x + w[2] * rr2.x;
      const iy = w[0] * rr0.y + w[1] * rr1.y + w[2] * rr2.y;
      return Math.hypot(ix - right[i].x, iy - right[i].y);
    }

    // точка вне выпуклой оболочки (вне кадра): запоминаем ближайший
    // треугольник, интерполяция — по барицентрическим весам самой точки
    const q = closestOnTriangle(p.x, p.y, a.x, a.y, b.x, b.y, c.x, c.y);
    const d2 = (q[0] - p.x) ** 2 + (q[1] - p.y) ** 2;
    if (d2 < bestDist2) {
      bestDist2 = d2;
      bestI0 = r0; bestI1 = r1; bestI2 = r2;
      bestW = w;
    }
  }

  if (!bestW) return NaN;                        // нет треугольников (вырожденный набор)
  const rr0 = vertexRight(bestI0, M, right, corners);
  const rr1 = vertexRight(bestI1, M, right, corners);
  const rr2 = vertexRight(bestI2, M, right, corners);
  const ix = bestW[0] * rr0.x + bestW[1] * rr1.x + bestW[2] * rr2.x;
  const iy = bestW[0] * rr0.y + bestW[1] * rr1.y + bestW[2] * rr2.y;
  return Math.hypot(ix - right[i].x, iy - right[i].y);
}

// Заполняет out[from..to) значениями смещений. Чанкованность — на стороне вызывающего.
export function computeImportanceRange(left, right, corners, from, to, out) {
  for (let i = from; i < to; i++) out[i] = valueWithout(left, right, i, corners);
}

// Нормировка к 95-му перцентилю (устойчиво к одиночным выбросам), срез в [0, 1].
export function normalize(values) {
  const finite = [];
  for (const v of values) if (Number.isFinite(v)) finite.push(v);
  const norms = new Float64Array(values.length);
  if (!finite.length) return norms;

  const sorted = finite.slice().sort((a, b) => a - b);
  let p95 = sorted[Math.floor((sorted.length - 1) * 0.95)];
  if (p95 <= 1e-12) p95 = sorted[sorted.length - 1];
  if (p95 <= 1e-12) return norms;

  for (let i = 0; i < values.length; i++) {
    norms[i] = Number.isFinite(values[i]) ? Math.min(values[i] / p95, 1) : 0;
  }
  return norms;
}

// Пары из состояния приложения: левые (эталон) и правые точки в порядке matches.
// img (левое изображение) — для угловых точек кадра; может быть null.
export function buildPairArrays(matches, kpsA, kpsB, img) {
  const left = [], right = [];
  for (const m of matches) {
    const la = kpsA[m.aIdx], rb = kpsB[m.bIdx];
    if (!la || !rb) return null;                // нарушение инварианта — не считаем
    left.push(la);
    right.push(rb);
  }
  if (!left.length) return null;
  return { left, right, corners: buildCorners(left, right, img) };
}

export const MIN_POINTS_FOR_IMPORTANCE = 4;
