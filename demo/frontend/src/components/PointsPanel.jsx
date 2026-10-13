import { useState } from 'react';
import { pairColor, COLOR_UNMATCHED, importanceColor } from '../constants';
import { unmatchedIdx } from '../points';
import { MIN_POINTS_FOR_IMPORTANCE } from '../importance';

const fmt = (v) => (Math.round(v * 10) / 10).toString();

// Две независимые закладки у правого края, каждая раскрывает свой сайдбар.
// Сайдбары — flex-элементы .panels: при открытии они забирают место у
// картинок (изображения сдвигаются), а не накладываются на них.
export function PointsPanel({
  pairsOpen, setPairsOpen, impOpen, setImpOpen,
  matches, kpsA, kpsB, selected, setSelected, setHovered,
  importance, importanceBusy, hiddenA, hiddenB,
}) {
  const [tab, setTab] = useState('pairs');

  // точка отсеяна фильтром «границы ∪ важные» — строка приглушается
  const isHidden = (side, idx) =>
    (side === 'a' ? hiddenA : hiddenB)?.has(idx) || false;
  const dimCls = (side, idx) => (isHidden(side, idx) ? ' dim' : '');

  const ua = unmatchedIdx(kpsA, matches, 'a');
  const ub = unmatchedIdx(kpsB, matches, 'b');
  const unpCount = ua.length + ub.length;

  const isActive = (side, idx) =>
    selected && selected.side === side && selected.idx === idx;

  const pick = (side, idx) => {
    setSelected(s => (s && s.side === side && s.idx === idx) ? null : { side, idx });
  };
  const hoverProps = (side, idx) => ({
    onMouseEnter: () => setHovered({ side, idx }),
    onMouseLeave: () => setHovered(h =>
      (h && h.side === side && h.idx === idx) ? null : h),
  });

  const unpSection = (side, idxs, kps, label) => (
    <div className="unp-section">
      <div className="matchlist-sub">
        <span>{label}</span>
        <span className="matchlist-count">{idxs.length}</span>
      </div>
      {idxs.map(idx => {
        const kp = kps[idx];
        if (!kp) return null;
        return (
          <button
            key={idx}
            className={'matchrow unp' + (isActive(side, idx) ? ' active' : '') + dimCls(side, idx)}
            style={{ '--pair': COLOR_UNMATCHED }}
            onClick={() => pick(side, idx)}
            {...hoverProps(side, idx)}
          >
            <span className="matchrow-coords">
              <span className={'coord coord-' + side}>
                <i className="dot" />{side.toUpperCase()} ({fmt(kp.x)}, {fmt(kp.y)})
              </span>
            </span>
          </button>
        );
      })}
    </div>
  );

  // строки сайдбара «Деформируемость» — по убыванию деформируемости
  const impRows = importance
    ? matches
        .map((m, i) => ({ m, i, v: importance.values[i], n: importance.norms[i] ?? 0 }))
        .filter(r => Number.isFinite(r.v))
        .sort((a, b) => b.v - a.v)
    : [];

  const impBody = () => {
    if (!importance) {
      return (
        <div className="matchlist-empty">
          {matches.length < MIN_POINTS_FOR_IMPORTANCE
            ? `Нужно ≥ ${MIN_POINTS_FOR_IMPORTANCE} пар для триангуляции`
            : importanceBusy ? 'Считается…' : 'Нет данных'}
        </div>
      );
    }
    if (!impRows.length) return <div className="matchlist-empty">Нет данных</div>;
    return impRows.map(r => {
      const kb = kpsB[r.m.bIdx];
      if (!kb) return null;
      return (
          <button
            key={r.i}
            className={'matchrow improw' + (isActive('b', r.m.bIdx) ? ' active' : '') + dimCls('b', r.m.bIdx)}
          style={{ '--pair': pairColor(r.i), '--imp': importanceColor(r.n) }}
          onClick={() => pick('b', r.m.bIdx)}
          {...hoverProps('b', r.m.bIdx)}
        >
          <span className="matchrow-no">{r.i + 1}</span>
          <span className="matchrow-coords">
            <span className="coord coord-b">
              <i className="dot" />B ({fmt(kb.x)}, {fmt(kb.y)})
            </span>
            <span className="impbar"><i style={{ width: (r.n * 100).toFixed(1) + '%' }} /></span>
          </span>
          <span className="matchrow-dist impval">{fmt(r.v)} px</span>
        </button>
      );
    });
  };

  return (
    <>
      {pairsOpen && (
        <aside className="sidebar pointspopup">
          <div className="pointspopup-head">
            <button
              className={'tab' + (tab === 'pairs' ? ' active' : '')}
              onClick={() => setTab('pairs')}
            >
              Пары <span className="matchlist-count">{matches.length}</span>
            </button>
            <button
              className={'tab' + (tab === 'unp' ? ' active' : '')}
              onClick={() => setTab('unp')}
            >
              Непарные <span className="matchlist-count">{unpCount}</span>
            </button>
            <button className="popup-close" onClick={() => setPairsOpen(false)}
                    title="Закрыть (Escape)">✕</button>
          </div>

          <div className="pointspopup-body">
            {tab === 'pairs' && (
              <>
                {!matches.length &&
                  <div className="matchlist-empty">Нет сопоставлений</div>}
                {matches.map((m, i) => {
                  const ka = kpsA[m.aIdx], kb = kpsB[m.bIdx];
                  if (!ka || !kb) return null;
                  return (
                  <button
                    key={i}
                    className={'matchrow' + (isActive('a', m.aIdx) ? ' active' : '')
                      + dimCls('a', m.aIdx) + dimCls('b', m.bIdx)}
                      style={{ '--pair': pairColor(i) }}
                      onClick={() => pick('a', m.aIdx)}
                      {...hoverProps('a', m.aIdx)}
                    >
                      <span className="matchrow-no">{i + 1}</span>
                      <span className="matchrow-coords">
                        <span className="coord coord-a">
                          <i className="dot" />A ({fmt(ka.x)}, {fmt(ka.y)})
                        </span>
                        <span className="coord coord-b">
                          <i className="dot" />B ({fmt(kb.x)}, {fmt(kb.y)})
                        </span>
                      </span>
                      <span className="matchrow-dist">{m.dist ?? '—'}</span>
                    </button>
                  );
                })}
              </>
            )}

            {tab === 'unp' && (
              <>
                {unpCount === 0 &&
                  <div className="matchlist-empty">Все точки в парах</div>}
                {unpSection('a', ua, kpsA, 'Непарные A')}
                {unpSection('b', ub, kpsB, 'Непарные B')}
              </>
            )}
          </div>
        </aside>
      )}

      {impOpen && (
        <aside className="sidebar impsidebar">
          <div className="pointspopup-head">
            <span className="sidebar-title">
              Деформируемость <span className="matchlist-count">{impRows.length}</span>
            </span>
            <button className="popup-close" onClick={() => setImpOpen(false)}
                    title="Закрыть (Escape)">✕</button>
          </div>

          <div className="imp-note">
            Насколько правую точку нельзя предсказать по остальным парам
            (триангуляция левых точек без неё, px на правой картинке).
            Левое изображение — эталон.
            <span className="imp-legend">
              <span>нет</span>
              <i />
              <span>деформируема</span>
            </span>
          </div>

          <div className="pointspopup-body">{impBody()}</div>
        </aside>
      )}

      <div className="bookmarks">
        <button
          className={'bookmark' + (pairsOpen ? ' open' : '')}
          onClick={() => setPairsOpen(o => !o)}
          title={pairsOpen ? 'Скрыть список пар' : 'Показать список пар'}
        >
          Пары <span className="bookmark-count">{matches.length}</span>
        </button>
        <button
          className={'bookmark' + (impOpen ? ' open' : '')}
          onClick={() => setImpOpen(o => !o)}
          title="Деформируемость точек правой картинки (левое — эталон)"
        >
          Деформируемость <span className="bookmark-count">{impRows.length}</span>
        </button>
      </div>
    </>
  );
}
