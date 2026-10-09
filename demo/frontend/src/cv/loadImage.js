import { parsePPMBytes, isPpmFile } from './ppm';

export function loadImage(file) {
  return isPpmFile(file) ? loadPpmFile(file) : loadBrowserImage(file);
}

function loadBrowserImage(file) {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => {
      const dataUrl = reader.result;
      const el = new Image();
      el.onload = () => resolve({
        el,
        dataUrl,
        url: dataUrl,
        width: el.naturalWidth,
        height: el.naturalHeight,
        name: file.name,
      });
      el.onerror = () => reject(new Error(`Не удалось открыть ${file.name}`));
      el.src = dataUrl;
    };
    reader.onerror = reject;
    reader.readAsDataURL(file);
  });
}

async function loadPpmFile(file) {
  const bytes = new Uint8Array(await file.arrayBuffer());
  const { width, height, data } = parsePPMBytes(bytes);

  const canvas = document.createElement('canvas');
  canvas.width = width;
  canvas.height = height;
  const ctx = canvas.getContext('2d');
  ctx.putImageData(new ImageData(data, width, height), 0, 0);

  // PNG — lossless, поэтому пиксели дойдут до бэкенда 1:1
  const dataUrl = canvas.toDataURL('image/png');

  const el = new Image();
  await new Promise((res, rej) => {
    el.onload = res;
    el.onerror = () => rej(new Error('PPM: не удалось отрисовать в canvas'));
    el.src = dataUrl;
  });

  return { el, dataUrl, url: dataUrl, width, height, name: file.name };
}

export function screenPoint(panelEl, tf, kp) {
  const r = panelEl.getBoundingClientRect();
  return {
    x: r.left + tf.tx + kp.x * tf.s,
    y: r.top + tf.ty + kp.y * tf.s,
  };
}