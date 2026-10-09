// ---------------------------------------------------------------------------
// cv.BFMatcher path (для локального OpenCV.js ORB)
// ---------------------------------------------------------------------------

function matchDescriptorsCv(a, b, { crossCheck = true } = {}) {
  const cv = window.cv;
  const bf = new cv.BFMatcher(cv.NORM_HAMMING, crossCheck);
  const m = new cv.DMatchVector();
  bf.match(a.desc, b.desc, m);

  const out = [];
  for (let i = 0; i < m.size(); i++) {
    const d = m.get(i);
    out.push({ aIdx: d.queryIdx, bIdx: d.trainIdx, dist: d.distance });
  }

  m.delete(); bf.delete();
  return out;
}

// ---------------------------------------------------------------------------
// JS brute-force path (для дескрипторов с бэка: Uint8Array | Float32Array)
// ---------------------------------------------------------------------------

function popcountByte(x) {
  x = x - ((x >> 1) & 0x55);
  x = (x & 0x33) + ((x >> 2) & 0x33);
  return (x + (x >> 4)) & 0x0f;
}

function makeDistanceFn(A, B, da, isHamming) {
  if (isHamming) {
    return (oa, ob) => {
      let s = 0;
      const end = oa + da;
      for (let i = oa, j = ob; i < end; i++, j++) {
        s += popcountByte(A[i] ^ B[j]);
      }
      return s;
    };
  }
  return (oa, ob) => {
    let s = 0;
    const end = oa + da;
    for (let i = oa, j = ob; i < end; i++, j++) {
      const d = A[i] - B[j];
      s += d * d;
    }
    return Math.sqrt(s);
  };
}

function matchDescriptorsJs(a, b, { crossCheck = true, ratio = null } = {}) {
  const A = a.desc, B = b.desc;
  const [na, da] = a.descShape;
  const [nb, db] = b.descShape;
  const isHamming = a.norm === 'hamming';

  const dist = makeDistanceFn(A, B, da, isHamming);

  const out = [];

  if (ratio != null) {
    for (let ia = 0; ia < na; ia++) {
      let best = Infinity, second = Infinity, bestIdx = -1;
      const oa = ia * da;
      for (let ib = 0; ib < nb; ib++) {
        const d = dist(oa, ib * db);
        if (d < best) {
          second = best;
          best = d;
          bestIdx = ib;
        } else if (d < second) {
          second = d;
        }
      }
      if (bestIdx >= 0 && best < ratio * second) {
        out.push({ aIdx: ia, bIdx: bestIdx, dist: best });
      }
    }
    return out;
  }

  const nnAB = new Int32Array(na).fill(-1);
  const nnBA = new Int32Array(nb).fill(-1);
  const dAB = new Float32Array(na).fill(Infinity);
  const dBA = new Float32Array(nb).fill(Infinity);

  for (let ia = 0; ia < na; ia++) {
    const oa = ia * da;
    for (let ib = 0; ib < nb; ib++) {
      const d = dist(oa, ib * db);
      if (d < dAB[ia]) { dAB[ia] = d; nnAB[ia] = ib; }
      if (d < dBA[ib]) { dBA[ib] = d; nnBA[ib] = ia; }
    }
  }

  for (let ia = 0; ia < na; ia++) {
    const ib = nnAB[ia];
    if (ib < 0) continue;
    if (crossCheck && nnBA[ib] !== ia) continue;
    out.push({ aIdx: ia, bIdx: ib, dist: dAB[ia] });
  }
  return out;
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

export async function matchDescriptors(a, b, opts = {}) {
  if (a.desc instanceof window.cv.Mat) {
    return matchDescriptorsCv(a, b, opts);
  }
  return matchDescriptorsJs(a, b, opts);
}