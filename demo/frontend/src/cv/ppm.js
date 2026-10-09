export function parsePPMBytes(bytes) {
  let pos = 0;
  const len = bytes.length;

  const isWs = (b) =>
    b === 0x20 || b === 0x09 || b === 0x0a || b === 0x0d ||
    b === 0x0b || b === 0x0c;

  function readToken() {
    while (pos < len && isWs(bytes[pos])) pos++;
    if (pos >= len) throw new Error('PPM: неожиданный конец файла');
    if (bytes[pos] === 0x23) {           // '#'
      while (pos < len && bytes[pos] !== 0x0a) pos++;
      return readToken();
    }
    const start = pos;
    while (pos < len && !isWs(bytes[pos])) pos++;
    return new TextDecoder('ascii').decode(bytes.subarray(start, pos));
  }

  const magic = readToken();
  if (magic !== 'P3' && magic !== 'P5' && magic !== 'P6') {
    throw new Error(`PPM: неподдерживаемый формат ${magic} (ожидаю P3/P5/P6)`);
  }

  const width  = parseInt(readToken(), 10);
  const height = parseInt(readToken(), 10);
  const maxval = parseInt(readToken(), 10);

  if (!Number.isFinite(width) || !Number.isFinite(height) ||
      width <= 0 || height <= 0) {
    throw new Error('PPM: некорректные размеры');
  }
  if (!Number.isFinite(maxval) || maxval <= 0 || maxval > 65535) {
    throw new Error('PPM: некорректный maxval');
  }

  // между maxval и данными ровно один пробельный байт
  if (pos >= len) throw new Error('PPM: отсутствуют пиксельные данные');
  pos++;

  const N = width * height;
  const out = new Uint8ClampedArray(N * 4);
  const scale = maxval === 255 ? 1 : 255 / maxval;

  if (magic === 'P6') {
    if (maxval < 256) {
      if (pos + N * 3 > len) throw new Error('PPM: обрезанные P6-данные');
      for (let i = 0, j = pos; i < N; i++, j += 3) {
        out[i * 4]     = bytes[j];
        out[i * 4 + 1] = bytes[j + 1];
        out[i * 4 + 2] = bytes[j + 2];
        out[i * 4 + 3] = 255;
      }
    } else {
      // 16-бит big-endian, берём старший байт
      if (pos + N * 6 > len) throw new Error('PPM: обрезанные 16-бит P6-данные');
      for (let i = 0, j = pos; i < N; i++, j += 6) {
        out[i * 4]     = bytes[j];
        out[i * 4 + 1] = bytes[j + 2];
        out[i * 4 + 2] = bytes[j + 4];
        out[i * 4 + 3] = 255;
      }
    }
  } else if (magic === 'P5') {
    if (maxval < 256) {
      if (pos + N > len) throw new Error('PPM: обрезанные P5-данные');
      for (let i = 0, j = pos; i < N; i++, j++) {
        const v = bytes[j];
        out[i * 4] = out[i * 4 + 1] = out[i * 4 + 2] = v;
        out[i * 4 + 3] = 255;
      }
    } else {
      if (pos + N * 2 > len) throw new Error('PPM: обрезанные 16-бит P5-данные');
      for (let i = 0, j = pos; i < N; i++, j += 2) {
        const v = bytes[j];
        out[i * 4] = out[i * 4 + 1] = out[i * 4 + 2] = v;
        out[i * 4 + 3] = 255;
      }
    }
  } else {
    // P3 — ASCII
    const tokens = new TextDecoder('ascii')
      .decode(bytes.subarray(pos))
      .split(/\s+/)
      .filter(Boolean);
    if (tokens.length < N * 3) throw new Error('PPM: обрезанные P3-данные');
    for (let i = 0; i < N; i++) {
      out[i * 4]     = Math.round(parseInt(tokens[i * 3],     10) * scale);
      out[i * 4 + 1] = Math.round(parseInt(tokens[i * 3 + 1], 10) * scale);
      out[i * 4 + 2] = Math.round(parseInt(tokens[i * 3 + 2], 10) * scale);
      out[i * 4 + 3] = 255;
    }
  }

  return { width, height, data: out };
}

export function isPpmFile(file) {
  const name = (file.name || '').toLowerCase();
  if (/\.(ppm|pgm|pnm)$/.test(name)) return true;
  const type = (file.type || '').toLowerCase();
  return type === 'image/x-portable-pixmap' ||
         type === 'image/x-portable-graymap' ||
         type === 'image/x-portable-anymap';
}