import { test } from 'node:test';
import assert from 'node:assert/strict';
import { argminNorms, removePair, restoreEntries } from '../src/prune.js';

const mkState = () => ({
  kpsA: [{ x: 0, y: 0 }, { x: 1, y: 0 }, { x: 2, y: 0 }, { x: 3, y: 0 }],
  kpsB: [{ x: 0, y: 5 }, { x: 1, y: 5 }, { x: 2, y: 5 }, { x: 3, y: 5 }],
  matches: [
    { aIdx: 0, bIdx: 0, dist: 1.5 },
    { aIdx: 1, bIdx: 1, dist: 2.5 },
    { aIdx: 2, bIdx: 2, dist: null },
    { aIdx: 3, bIdx: 3, dist: 0.5 },
  ],
});

test('argminNorms: минимальное конечное значение', () => {
  assert.equal(argminNorms([0.5, 0.1, 0.9]), 1);
  assert.equal(argminNorms([0.1, 0.1, 0.2]), 0);      // равенство — первый
  assert.equal(argminNorms([NaN, 0.3, Infinity]), 1); // нефинитные пропускаются
  assert.equal(argminNorms([NaN, Infinity]), null);
  assert.equal(argminNorms([]), null);
});

test('removePair: удаляет пару целиком, индексы пересчитаны', () => {
  const s = mkState();
  const r = removePair(s.kpsA, s.kpsB, s.matches, 1);
  assert.equal(r.kpsA.length, 3);
  assert.equal(r.kpsB.length, 3);
  assert.equal(r.matches.length, 3);
  // kpsA[1] (x=1) удалён — осталось x=0,2,3
  assert.deepEqual(r.kpsA.map(k => k.x), [0, 2, 3]);
  assert.deepEqual(r.kpsB.map(k => k.x), [0, 2, 3]);
  // matches пересчитаны: бывшая пара 2 теперь aIdx=1,bIdx=1
  assert.deepEqual(r.matches.map(m => m.aIdx), [0, 1, 2]);
  assert.equal(r.matches[1].dist, null);
  // entry хранит удалённое и длины до удаления
  assert.equal(r.entry.ka.x, 1);
  assert.equal(r.entry.kb.y, 5);
  assert.equal(r.entry.dist, 2.5);
  assert.equal(r.entry.la, 4);
  assert.equal(r.entry.lb, 4);
  assert.equal(r.entry.lm, 4);
});

test('removePair: неверный индекс → null', () => {
  const s = mkState();
  assert.equal(removePair(s.kpsA, s.kpsB, s.matches, 99), null);
  assert.equal(removePair(s.kpsA, s.kpsB, s.matches, -1), null);
});

test('restoreEntries: точный возврат в исходные позиции (LIFO)', () => {
  const s = mkState();
  const r1 = removePair(s.kpsA, s.kpsB, s.matches, 2);
  const r2 = removePair(r1.kpsA, r1.kpsB, r1.matches, 0);
  const r3 = removePair(r2.kpsA, r2.kpsB, r2.matches, 0);

  const back = restoreEntries(r3.kpsA, r3.kpsB, r3.matches,
    [r1.entry, r2.entry, r3.entry]);
  assert.equal(back.failed, 0);
  assert.equal(back.restored, 3);
  assert.deepEqual(back.kpsA, s.kpsA);
  assert.deepEqual(back.kpsB, s.kpsB);
  assert.deepEqual(back.matches, s.matches);
});

test('restoreEntries: частичный возврат — хвост стека снимается', () => {
  const s = mkState();
  const r1 = removePair(s.kpsA, s.kpsB, s.matches, 0);
  const r2 = removePair(r1.kpsA, r1.kpsB, r1.matches, 2);
  // снимаем только последнюю запись
  const back = restoreEntries(r2.kpsA, r2.kpsB, r2.matches, [r2.entry]);
  assert.equal(back.failed, 0);
  assert.deepEqual(back.matches, r1.matches);
  assert.deepEqual(back.kpsA, r1.kpsA);
});

test('restoreEntries: внешняя правка (длины не сходятся) → fallback в конец', () => {
  const s = mkState();
  const r1 = removePair(s.kpsA, s.kpsB, s.matches, 0);
  // внешняя правка: добавили точку
  const tampered = { ...r1, kpsA: [...r1.kpsA, { x: 99, y: 99 }] };
  const back = restoreEntries(tampered.kpsA, tampered.kpsB, tampered.matches, [r1.entry]);
  assert.equal(back.failed, 1);
  assert.equal(back.restored, 0);
  // fallback: точка добавлена в конец, пара в конце matches
  assert.equal(back.kpsA.length, 5);
  assert.equal(back.kpsA[4].x, 0);                      // восстановленная точка (aIdx=0)
  const last = back.matches[back.matches.length - 1];
  assert.equal(back.kpsA[last.aIdx].x, 0);
  assert.equal(back.kpsB[last.bIdx].y, 5);
});

test('restoreEntries: повреждённая запись (тот же объект на месте) → fallback', () => {
  const s = mkState();
  const r1 = removePair(s.kpsA, s.kpsB, s.matches, 1);
  // симуляция: кто-то подсунул тот же объект обратно без вставки
  const same = { ...r1, kpsA: r1.kpsA };  // длины как после удаления, но позиция может совпасть
  // запись валидна по длинам, но объект может совпасть — если a[e.aIdx] === e.ka, это подмена
  const back = restoreEntries(same.kpsA, same.kpsB, same.matches, [r1.entry]);
  // длины сходятся (la-1), позиция содержит другой объект (x=2) — валидный возврат
  assert.equal(back.failed, 0);
  assert.deepEqual(back.kpsA, s.kpsA);
  assert.deepEqual(back.matches, s.matches);
});

test('removePair → restore → removePair снова: стек работает повторно', () => {
  const s = mkState();
  const r1 = removePair(s.kpsA, s.kpsB, s.matches, 1);
  const back = restoreEntries(r1.kpsA, r1.kpsB, r1.matches, [r1.entry]);
  const r2 = removePair(back.kpsA, back.kpsB, back.matches, 1);
  assert.equal(r2.entry.ka.x, 1);
  assert.deepEqual(r2.kpsA.map(k => k.x), [0, 2, 3]);
});
