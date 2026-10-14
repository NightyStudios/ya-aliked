import { test } from 'node:test';
import assert from 'node:assert/strict';
import { computeHiddenSets } from '../src/filterPoints.js';

const inputs = {
  kpsA: [{ x: 0, y: 0 }, { x: 1, y: 0 }, { x: 2, y: 0 }],
  kpsB: [{ x: 0, y: 0 }, { x: 1, y: 0 }, { x: 2, y: 0 }],
  matches: [{ aIdx: 0, bIdx: 0 }, { aIdx: 1, bIdx: 1 }],
  boundary: { width: 3, height: 1, data: new Uint8ClampedArray(12) },
  imgB: { width: 3, height: 1 },
  importance: { values: [1, 1], norms: [1, 1] },
};

test('class prohibition overrides importance and removes both ends of a pair', () => {
  const r = computeHiddenSets({ ...inputs, classKeep: [false, true, true], geometryOn: true });
  assert.deepEqual([...r.hiddenB], [0, 2]);
  assert.ok(r.hiddenA.has(0));
  assert.equal(r.keptPairs, 1);
});

test('class-only mode preserves allowed unmatched points and unmatched left points', () => {
  const r = computeHiddenSets({ ...inputs, classKeep: [false, true, true], geometryOn: false });
  assert.deepEqual([...r.hiddenB], [0]);
  assert.deepEqual([...r.hiddenA], [0]);
});

test('empty prohibitions preserve all points in class-only mode', () => {
  const r = computeHiddenSets({ ...inputs, geometryOn: false });
  assert.equal(r.hiddenA.size, 0);
  assert.equal(r.hiddenB.size, 0);
  assert.equal(r.keptPairs, 2);
});
