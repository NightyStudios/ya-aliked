// Итеративное удаление пар с наименьшей деформируемостью (ползунок в фильтрах).
// Удаление реальное: пара уходит из matches, обе точки — из kpsA/kpsB.
// Стек записей хранит снимок удалённого — ползунок влево восстанавливает
// в исходные позиции (LIFO), если массивы с тех пор не менялись стороной.

// Индекс минимального конечного значения (первый при равенстве); null — все NaN.
export function argminNorms(norms) {
  let best = -1, bestV = Infinity;
  for (let i = 0; i < norms.length; i++) {
    const v = norms[i];
    if (!Number.isFinite(v)) continue;
    if (v < bestV) { bestV = v; best = i; }
  }
  return best < 0 ? null : best;
}

// Удаляет пару matches[matchIdx] целиком. Возвращает { kpsA, kpsB, matches, entry }
// или null при неверном индексе. Индексы оставшихся пар пересчитываются.
// entry хранит длины ДО удаления — для валидации при восстановлении
// (внешние правки сдвигают индексы).
export function removePair(kpsA, kpsB, matches, matchIdx) {
  const m = matches[matchIdx];
  if (!m || kpsA[m.aIdx] == null || kpsB[m.bIdx] == null) return null;

  const entry = {
    aIdx: m.aIdx, bIdx: m.bIdx, mIdx: matchIdx,
    ka: kpsA[m.aIdx], kb: kpsB[m.bIdx],
    dist: m.dist ?? null,
    la: kpsA.length, lb: kpsB.length, lm: matches.length,
  };

  return {
    kpsA: kpsA.filter((_, i) => i !== m.aIdx),
    kpsB: kpsB.filter((_, i) => i !== m.bIdx),
    matches: matches.filter((_, i) => i !== matchIdx).map(x => ({
      ...x,
      aIdx: x.aIdx > m.aIdx ? x.aIdx - 1 : x.aIdx,
      bIdx: x.bIdx > m.bIdx ? x.bIdx - 1 : x.bIdx,
    })),
    entry,
  };
}

// Восстанавливает записи в обратном порядке. Записи, длины которых не сходятся
// с текущим состоянием (внешняя правка), выкидываются; если хоть одна выкинута —
// остальные идут fallback'ом «в конец» (порядок пар меняется, данные корректны).
// Возвращает { kpsA, kpsB, matches, restored, failed }.
export function restoreEntries(kpsA, kpsB, matches, entries) {
  let a = kpsA, b = kpsB, ms = matches;
  let failed = 0;

  for (let i = entries.length - 1; i >= 0; i--) {
    const e = entries[i];
    if (a.length === e.la - 1 && b.length === e.lb - 1 && ms.length === e.lm - 1
        && a[e.aIdx] !== e.ka && b[e.bIdx] !== e.kb) {
      a = [...a.slice(0, e.aIdx), e.ka, ...a.slice(e.aIdx)];
      b = [...b.slice(0, e.bIdx), e.kb, ...b.slice(e.bIdx)];
      // индексы пар, сдвинутые вставкой точек
      ms = ms.map(x => ({
        ...x,
        aIdx: x.aIdx >= e.aIdx ? x.aIdx + 1 : x.aIdx,
        bIdx: x.bIdx >= e.bIdx ? x.bIdx + 1 : x.bIdx,
      }));
      ms = [...ms.slice(0, e.mIdx), { aIdx: e.aIdx, bIdx: e.bIdx, dist: e.dist },
        ...ms.slice(e.mIdx)];
    } else {
      failed++;
      a = [...a, e.ka];
      b = [...b, e.kb];
      ms = [...ms, { aIdx: a.length - 1, bIdx: b.length - 1, dist: e.dist }];
    }
  }

  return { kpsA: a, kpsB: b, matches: ms, restored: entries.length - failed, failed };
}
