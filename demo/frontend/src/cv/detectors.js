import { MAX_FEATURES } from '../constants';

// ---------------------------------------------------------------------------
// Local: OpenCV.js ORB
// ---------------------------------------------------------------------------

async function detectOrbCvLocal(img) {
  const cv = window.cv;
  const src = cv.imread(img.el);
  const gray = new cv.Mat();
  cv.cvtColor(src, gray, cv.COLOR_RGBA2GRAY);

  const orb = new cv.ORB(MAX_FEATURES);
  const mask = new cv.Mat();
  const kps = new cv.KeyPointVector();
  const desc = new cv.Mat();
  orb.detectAndCompute(gray, mask, kps, desc);

  const points = [];
  for (let i = 0; i < kps.size(); i++) {
    const p = kps.get(i).pt;
    points.push({ x: p.x, y: p.y });
  }

  src.delete(); gray.delete(); mask.delete(); orb.delete(); kps.delete();

  return {
    points,
    desc,                      // cv.Mat — матчер распознает и пойдёт в cv.BFMatcher
    descShape: [points.length, desc.cols],
    norm: 'hamming',
  };
}

// ---------------------------------------------------------------------------
// Remote: backend /api/detect
// ---------------------------------------------------------------------------

export async function detectRemote(method, img, params = {}) {
  const body = {
    method,
    image: { image_b64: img.dataUrl.split(',', 2)[1] },
    params,
  };
  const r = await fetch('/api/detect', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  });
  if (!r.ok) {
    const text = await r.text().catch(() => '');
    throw new Error(`${method}: ${r.status} ${text}`);
  }
  const { detection } = await r.json();
  return decodeDetection(detection);
}

function decodeDetection(d) {
  const bin = atob(d.desc_b64);
  const bytes = new Uint8Array(bin.length);
  for (let i = 0; i < bin.length; i++) bytes[i] = bin.charCodeAt(i);

  const [n, dim] = d.desc_shape;
  let desc;
  if (d.desc_dtype === 'uint8') {
    desc = new Uint8Array(bytes.buffer);
  } else {
    desc = new Float32Array(bytes.buffer);
  }

  return {
    points: d.points.map(([x, y]) => ({ x, y })),
    desc,
    descShape: [n, dim],
    norm: d.norm,
  };
}

// ---------------------------------------------------------------------------
// Registry
// ---------------------------------------------------------------------------

export const DETECTORS = {
  orb_cv: {
    id: 'orb_cv', label: 'cv2.ORB',
    detect: detectOrbCvLocal,
  },
  orb_pyramid: {
    id: 'orb_pyramid', label: 'pyramid',
    detect: (img, p) => detectRemote('orb_pyramid', img, p),
  },
  orb_manual: {
    id: 'orb_manual', label: 'manual',
    detect: (img, p) => detectRemote('orb_manual', img, p),
  },
  orb_fast: {
    id: 'orb_fast', label: 'fast',
    detect: (img, p) => detectRemote('orb_fast', img, p),
  },
};