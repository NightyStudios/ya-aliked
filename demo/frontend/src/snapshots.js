// Слоты снимков точек в localStorage:
//  - initial — база: состояние при загрузке JSON-файла или завершении детекта;
//  - actual  — актуальный рабочий снимок (Ctrl+S, кнопка, тихий сейв, автосейв).
// Храним готовый текст serializePointsJson — dirty-сравнение байт-в-байт.

const PREFIX = 'aliked.points.';
const KEY_INITIAL = PREFIX + 'initial';
const KEY_ACTUAL = PREFIX + 'actual';

const safeStorage = (storage) => {
  if (storage) return storage;
  try { return globalThis.localStorage; } catch { return null; }
};

const readSlot = (storage, slot) => {
  const key = slot === 'initial' ? KEY_INITIAL : KEY_ACTUAL;
  const s = safeStorage(storage);
  if (!s) return null;
  try {
    const raw = s.getItem(key);
    if (!raw) return null;
    const v = JSON.parse(raw);
    if (!v || typeof v !== 'object' || typeof v.text !== 'string') return null;
    return {
      text: v.text,
      savedAt: Number.isFinite(v.savedAt) ? v.savedAt : null,
      imageA: typeof v.imageA === 'string' ? v.imageA : null,
      imageB: typeof v.imageB === 'string' ? v.imageB : null,
    };
  } catch {
    return null;
  }
};

const writeSlot = (storage, slot, { text, imageA, imageB, savedAt = Date.now() }) => {
  const key = slot === 'initial' ? KEY_INITIAL : KEY_ACTUAL;
  const s = safeStorage(storage);
  if (!s) return false;
  try {
    s.setItem(key, JSON.stringify({
      v: 1, savedAt, text, imageA: imageA ?? null, imageB: imageB ?? null,
    }));
    return true;
  } catch {
    return false;   // quota / private mode — молча, приложение работает без снапшотов
  }
};

const clearSlot = (storage, slot) => {
  const key = slot === 'initial' ? KEY_INITIAL : KEY_ACTUAL;
  const s = safeStorage(storage);
  if (!s) return;
  try { s.removeItem(key); } catch { /* ignore */ }
};

export { readSlot, writeSlot, clearSlot };
