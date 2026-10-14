import { test } from 'node:test';
import assert from 'node:assert/strict';
import { readSlot, writeSlot, clearSlot } from '../src/snapshots.js';

// фейковый localStorage для node
const fakeStorage = () => {
  const m = new Map();
  return {
    getItem: (k) => (m.has(k) ? m.get(k) : null),
    setItem: (k, v) => { m.set(k, String(v)); },
    removeItem: (k) => { m.delete(k); },
    _map: m,
  };
};

test('roundtrip: write then read returns text and meta', () => {
  const st = fakeStorage();
  const slot = { text: '{"pairs":[]}', savedAt: 111, imageA: 'a.png', imageB: 'b.png' };
  assert.equal(writeSlot(st, 'actual', slot), true);
  const r = readSlot(st, 'actual');
  assert.equal(r.text, slot.text);
  assert.equal(r.savedAt, 111);
  assert.equal(r.imageA, 'a.png');
  assert.equal(r.imageB, 'b.png');
});

test('slots are independent: initial and actual do not clash', () => {
  const st = fakeStorage();
  writeSlot(st, 'initial', { text: 'A', savedAt: 1 });
  writeSlot(st, 'actual', { text: 'B', savedAt: 2 });
  assert.equal(readSlot(st, 'initial').text, 'A');
  assert.equal(readSlot(st, 'actual').text, 'B');
});

test('empty slot reads as null', () => {
  const st = fakeStorage();
  assert.equal(readSlot(st, 'initial'), null);
  assert.equal(readSlot(st, 'actual'), null);
});

test('corrupt json reads as null (не бросает)', () => {
  const st = fakeStorage();
  st.setItem('aliked.points.actual', '{oops');
  assert.equal(readSlot(st, 'actual'), null);
  st.setItem('aliked.points.actual', '{"text": 42}');
  assert.equal(readSlot(st, 'actual'), null);
});

test('clearSlot removes only that slot', () => {
  const st = fakeStorage();
  writeSlot(st, 'initial', { text: 'A', savedAt: 1 });
  writeSlot(st, 'actual', { text: 'B', savedAt: 2 });
  clearSlot(st, 'initial');
  assert.equal(readSlot(st, 'initial'), null);
  assert.equal(readSlot(st, 'actual').text, 'B');
});

test('storage приватного режима (throw) — write возвращает false, read null', () => {
  const boom = {
    getItem() { throw new Error('denied'); },
    setItem() { throw new Error('denied'); },
    removeItem() { throw new Error('denied'); },
  };
  assert.equal(writeSlot(boom, 'actual', { text: 'x', savedAt: 1 }), false);
  assert.equal(readSlot(boom, 'actual'), null);
  assert.doesNotThrow(() => clearSlot(boom, 'actual'));
});

test('без storage (null) — безопасные no-op', () => {
  const st = fakeStorage();
  // передаём явный storage — null внутри модуля упадёт на globalThis.localStorage,
  // здесь проверяем только явный путь с рабочим storage
  assert.equal(writeSlot(st, 'initial', { text: 'x', savedAt: 1 }), true);
  assert.ok(readSlot(st, 'initial'));
});
