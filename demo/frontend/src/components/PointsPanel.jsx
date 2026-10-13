import { useState } from 'react';
import { pairColor, COLOR_UNMATCHED, importanceColor } from '../constants';
import { unmatchedIdx } from '../points';
import { MIN_POINTS_FOR_IMPORTANCE } from '../importance';

const fmt = (v) => (Math.round(v * 10) / 10).toString();

// Метрики проверки с эталоном: cmp — как выбирать лучшую ячейку
// (lower/higher — меньше/больше лучше, abs — по модулю, neutral — без подсветки)
const CHECK_METRICS = [
  { key: 'mae', label: 'MAE (0–255)', cmp: 'lower', fmt: v => v.toFixed(2) },
  { key: 'psnr', label: 'PSNR, дБ', cmp: 'higher', fmt: v => v.toFixed(2) },
  { key: 'ssim', label: 'SSIM', cmp: 'higher', fmt: v => v.toFixed(3) },
  { key: 'ncc', label: 'корреляция', cmp: 'higher', fmt: v => v.toFixed(3) },
  { key: 'dx', label: 'сдвиг dx, px', cmp: 'abs', fmt: v => (v >= 0 ? '+' : '') + v.toFixed(2) },
  { key: 'dy', label: 'сдвиг dy, px', cmp: 'abs', fmt: v => (v >= 0 ? '+' : '') + v.toFixed(2) },
  { key: 'tiles', label: 'кусков оценено', cmp: 'neutral', fmt: v => String(Math.round(v)) },
  { key: 'tile_median', label: 'медиана сдвига, px', cmp: 'lower', fmt: v => v.toFixed(2) },
  { key: 'tiles_under_1px', label: 'кусков < 1 px', cmp: 'higher', fmt: v => (v * 100).toFixed(0) + '%' },
];

// 0 — лучше initial, 1 — лучше warped, -1 — не сравнить
const betterIdx = (a, b, cmp) => {
  if (cmp === 'neutral' || a == null || b == null || a === b) return -1;
  if (cmp === 'lower') return a < b ? 0 : 1;
  if (cmp === 'higher') return a > b ? 0 : 1;
  if (cmp === 'abs') return Math.abs(a) < Math.abs(b) ? 0 : 1;
  return -1;
};

// Две независимые закладки у правого края, каждая раскрывает свой сайдбар.
// Сайдбары — flex-элементы .panels: при открытии они забирают место у
// картинок (изображения сдвигаются), а не накладываются на них.
export function PointsPanel({
  pairsOpen, setPairsOpen, impOpen, setImpOpen,
  matches, kpsA, kpsB, selected, setSelected, setHovered,
  importance, importanceBusy, hiddenA, hiddenB,
  checkOpen, setCheckOpen, checkData, checkBusy, checkError,
}) {
  const [tab, setTab] = useState('pairs');

  // точка отсеяна фильтром «границы ∪ важные» — строка приглушается
  const isHidden = (side, idx) =>
    (side === 'a' ? hiddenA : hiddenB)?.has(idx) || false;
  const dimCls = (side, idx) => (isHidden(side, idx) ? ' dim' : '');
  // пара исключена из триангуляции кликом в режиме деформации
  const exclCls = (...kps) => (kps.some(k => k && k.ex) ? ' excl' : '');

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
            className={'matchrow unp' + (isActive(side, idx) ? ' active' : '')
              + dimCls(side, idx) + exclCls(kp)}
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
            className={'matchrow improw' + (isActive('b', r.m.bIdx) ? ' active' : '')
              + dimCls('b', r.m.bIdx) + exclCls(kb)}
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

  // тело сайдбара «Проверка»: таблица метрик начальное/наложенное
  const checkBody = () => {
    if (checkBusy) return <div className="matchlist-empty">Считаем метрики…</div>;
    if (checkError) return <div className="matchlist-empty">Ошибка: {checkError}</div>;
    if (!checkData) {
      return (
        <div className="matchlist-empty">
          Нажмите «Проверить» в тулбаре и выберите файл-эталон.
          Нужны правое изображение и рассчитанное наложение.
        </div>
      );
    }

    const { refName, compared_size, initial, warped, elapsed_ms } = checkData;
    let compared = 0, warpedWins = 0;
    CHECK_METRICS.forEach(m => {
      const bi = betterIdx(initial[m.key], warped[m.key], m.cmp);
      if (bi >= 0) { compared++; if (bi === 1) warpedWins++; }
    });

    const cell = (m, v, bi, idx) => (
      <td className={v == null ? 'na' : (bi === idx ? 'better' : '')}>
        {v == null ? '—' : m.fmt(v)}
      </td>
    );

    return (
      <>
        <div className="check-note">
          Эталон: <b>{refName}</b>
          {' '}· область {compared_size[0]}×{compared_size[1]} px
          {elapsed_ms != null && <> · {Math.round(elapsed_ms)} мс</>}
        </div>
        <div className="pointspopup-body">
          <table className="checktable">
            <thead>
              <tr><th>метрика</th><th>начальное</th><th>наложенное</th></tr>
            </thead>
            <tbody>
              {CHECK_METRICS.map(m => {
                const a = initial[m.key], b = warped[m.key];
                const bi = betterIdx(a, b, m.cmp);
                return (
                  <tr key={m.key}>
                    <td>{m.label}</td>
                    {cell(m, a, bi, 0)}
                    {cell(m, b, bi, 1)}
                  </tr>
                );
              })}
            </tbody>
          </table>
          {compared > 0 && (
            <div className="check-verdict">
              Наложенное лучше по {warpedWins} из {compared} сравнимых метрик
            </div>
          )}
        </div>
      </>
    );
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
                      + dimCls('a', m.aIdx) + dimCls('b', m.bIdx) + exclCls(ka, kb)}
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

      {checkOpen && (
        <aside className="sidebar checksidebar">
          <div className="pointspopup-head">
            <span className="sidebar-title">Проверка с эталоном</span>
            <button className="popup-close" onClick={() => setCheckOpen(false)}
                    title="Закрыть (Escape)">✕</button>
          </div>
          {checkBody()}
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
        <button
          className={'bookmark' + (checkOpen ? ' open' : '')}
          onClick={() => setCheckOpen(o => !o)}
          title="Сравнение начального и наложенного изображений с выбранным эталоном"
        >
          Проверка
        </button>
      </div>
    </>
  );
}
