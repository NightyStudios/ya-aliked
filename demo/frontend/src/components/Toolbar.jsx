import { useEffect, useRef, useState } from 'react';
import { visibleMethods } from '../cv/methods';

const FALLBACK = [{ id: 'orb_cv', label: 'orb_cv', group: 'ORB' }];

// --- inline-SVG иконки (без внешних зависимостей) ---
const SV = {
  width: 17, height: 17, viewBox: '0 0 24 24', fill: 'none',
  stroke: 'currentColor', strokeWidth: 1.7,
  strokeLinecap: 'round', strokeLinejoin: 'round',
};

// детект точек: перекрестие с точками
const IcPoints = () => (
  <svg {...SV}>
    <circle cx="12" cy="12" r="6.5" />
    <path d="M12 2v3.2M12 18.8V22M2 12h3.2M18.8 12H22" />
    <circle cx="12" cy="12" r="1.4" fill="currentColor" stroke="none" />
  </svg>
);

// соединить пару: звенья цепи
const IcLink = () => (
  <svg {...SV}>
    <path d="M10 13a5 5 0 0 0 7.54.54l3-3a5 5 0 0 0-7.07-7.07l-1.72 1.71" />
    <path d="M14 11a5 5 0 0 0-7.54-.54l-3 3a5 5 0 0 0 7.07 7.07l1.71-1.71" />
  </svg>
);

// наложение: слои
const IcLayers = () => (
  <svg {...SV}>
    <path d="M12.83 2.18a2 2 0 0 0-1.66 0L2.6 6.08a1 1 0 0 0 0 1.83l8.58 3.91a2 2 0 0 0 1.66 0l8.58-3.9a1 1 0 0 0 0-1.83Z" />
    <path d="m22 17.65-9.17 4.16a2 2 0 0 1-1.66 0L2 17.65" />
    <path d="m22 12.65-9.17 4.16a2 2 0 0 1-1.66 0L2 12.65" />
  </svg>
);

// триангуляция: треугольник с внутренними рёбрами
const IcMesh = () => (
  <svg {...SV}>
    <path d="M12 2.6 22 20H2Z" />
    <path d="M12 13.5V2.6M12 13.5 2 20M12 13.5 22 20" />
  </svg>
);

// режим деформации: треугольник с «выбитой» вершиной (×)
const IcDeform = () => (
  <svg {...SV}>
    <path d="M12 4.5 21.5 20.5h-19Z" />
    <path d="m10 2.5 4 4M14 2.5l-4 4" />
  </svg>
);

// режим редактирования: карандаш
const IcEdit = () => (
  <svg {...SV}>
    <path d="M17 3a2.85 2.83 0 1 1 4 4L7.5 20.5 2 22l1.5-5.5Z" />
    <path d="m15 5 4 4" />
  </svg>
);

// solo: глаз с перечёркиванием — скрыть всё кроме выбранной пары
const IcSolo = () => (
  <svg {...SV}>
    <path d="M10.73 5.08A10.43 10.43 0 0 1 12 5c7 0 10 7 10 7a13.16 13.16 0 0 1-1.67 2.68" />
    <path d="M6.61 6.61A13.5 13.5 0 0 0 2 12s3 7 10 7a9.74 9.74 0 0 0 5.39-1.61" />
    <path d="M14.12 14.12a3 3 0 1 1-4.24-4.24" />
    <path d="m2 2 20 20" />
  </svg>
);

// маски: область-«капля»
const IcMask = () => (
  <svg {...SV}>
    <path d="M12 3c4.6 0 8 3.1 8 7.6 0 5-4.4 10.4-8 10.4s-8-5.4-8-10.4C4 6.1 7.4 3 12 3Z" />
    <path d="M9.2 10.5h.01M14.8 10.5h.01M9.5 14.6c1.6 1.1 3.4 1.1 5 0" />
  </svg>
);

// фильтр: воронка
const IcFilter = () => (
  <svg {...SV}>
    <path d="M22 3H2l8 9.46V19l4 2v-8.54L22 3Z" />
  </svg>
);

// проверка с эталоном: лупа с галочкой
const IcCheck = () => (
  <svg {...SV}>
    <circle cx="10.5" cy="10.5" r="6.5" />
    <path d="m15.5 15.5 4.5 4.5" />
    <path d="m7.8 10.6 1.9 1.9 3.5-3.5" />
  </svg>
);

// ещё: три точки
const IcMore = () => (
  <svg {...SV}>
    <circle cx="5" cy="12" r="1.5" fill="currentColor" stroke="none" />
    <circle cx="12" cy="12" r="1.5" fill="currentColor" stroke="none" />
    <circle cx="19" cy="12" r="1.5" fill="currentColor" stroke="none" />
  </svg>
);

export function Toolbar({
  method, setMethod, methodsMeta, onReset,
  busy, showAllLines, onToggleAllLines,
  onJsonFile, onSaveJson, connectMode, onToggleConnect,
  deformMode, onToggleDeform,
  editMode, onToggleEdit, soloMode, onToggleSolo,
  onGenerate, canGenerate,
  canOverlay, overlayVisible, onToggleOverlay,
  overlayAlpha, setOverlayAlpha,
  canRWarp, rwarpVisible, onToggleRWarp,
  rwarpAlpha, setRWarpAlpha,
  warpCount, pairCount,
  onSaveImage, canSave,
  maskVisible, onToggleMask, maskBusy, maskError, canMask,
  maskAlpha, setMaskAlpha, maskCount, maskMs,
  filterOn, onToggleFilter, canFilter, filterInfo,
  filterDirty, onApplyFilter, onExpandFilter,
  filterImpThr, setFilterImpThr, filterRadius, setFilterRadius,
  filterReady,
  pruneTarget, setPruneTarget, pruneMax, pruneBusy, pruneProgress, pruneRemoved,
  semanticClasses, excludedClasses, setExcludedClasses,
  classMargin, setClassMargin, geometryFilterOn, setGeometryFilterOn,
  classFilterBusy, classFilterError, marginSavedCount,
  onCheckFile, checkBusy, checkError, refName,
  snapshotDirty, snapshotInitialAt, snapshotActualAt,
  onSaveSnapshot, onRestoreSnapshot, autoSave, setAutoSave,
}) {
  const list = visibleMethods(methodsMeta);
  const entries = list.length ? list : FALLBACK;

  const groups = {};
  entries.forEach(m => {
    const g = m.group || 'ORB';
    (groups[g] = groups[g] || []).push(m);
  });

  const current = entries.find(m => m.id === method) || entries[0];

  // --- выпадающее меню методов генерации и общее меню «ещё» ---
  const [menuOpen, setMenuOpen] = useState(false);
  const [moreOpen, setMoreOpen] = useState(false);
  const genRef = useRef(null);
  const moreRef = useRef(null);
  const checkInputRef = useRef(null);

  useEffect(() => {
    if (!menuOpen && !moreOpen) return;
    const onDown = (e) => {
      if (genRef.current && !genRef.current.contains(e.target)) setMenuOpen(false);
      if (moreRef.current && !moreRef.current.contains(e.target)) setMoreOpen(false);
    };
    const onKey = (e) => {
      if (e.key !== 'Escape') return;
      setMenuOpen(false);
      setMoreOpen(false);
    };
    window.addEventListener('mousedown', onDown);
    window.addEventListener('keydown', onKey);
    return () => {
      window.removeEventListener('mousedown', onDown);
      window.removeEventListener('keydown', onKey);
    };
  }, [menuOpen, moreOpen]);

  useEffect(() => { if (busy) setMenuOpen(false); }, [busy]);

  // клик по методу: выбрать его и сразу запустить детект
  const pickMethod = (id) => {
    setMenuOpen(false);
    if (id !== method) setMethod(id);
    onGenerate();
  };

  const excluded = Math.max(0, pairCount - warpCount);
  const overlayTitle = canOverlay
    ? `Наложить правое изображение на левое по ${warpCount} парам триангуляции ` +
      `(всего пар: ${pairCount}${excluded ? `, исключено кликом: ${excluded}` : ''})`
    : 'Нужны обе картинки и ≥ 3 пар (исключённые кликом не считаются)';

  const rwarpTitle = canRWarp
    ? `Наложить деформированное правое (warp B→A) на неизменённое правое · ` +
      `сетка по ${warpCount} парам` +
      (excluded ? ` (исключено кликом: ${excluded})` : '')
    : 'Нужны обе картинки и ≥ 3 пар (исключённые кликом не считаются)';

  const maskTitle = !canMask
    ? 'Сначала загрузите правое изображение'
    : maskBusy
      ? 'Mask2Former считает маски… (первый запрос может качать веса модели)'
      : maskError
        ? `Ошибка: ${maskError}`
        : maskVisible
          ? `Скрыть маски Mask2Former · областей: ${maskCount}` +
            (maskMs != null ? ` · ${Math.round(maskMs)} мс` : '')
          : 'Показать семантические области Mask2Former на всём правом изображении';

  const filterTitle = !canFilter
    ? 'Сначала посчитайте маски Mask2Former'
    : filterOn
      ? filterDirty
        ? 'Выключить фильтры и показать все точки'
        : 'Показать панель настроек фильтра (фильтр активен)'
      : 'Настроить запрещённые классы и марджин для точек правого снимка';

  const checkTitle = checkBusy
    ? 'Считаем метрики сходства с эталоном…'
    : checkError
      ? `Ошибка: ${checkError}`
      : refName
        ? `Эталон «${refName}» — метрики обновляются автоматически. Выбрать другой файл`
        : 'Выбрать эталон: MAE, PSNR, SSIM, корреляция и сдвиги — метрики ' +
          'пересчитываются автоматически при любой правке наложения';

  const closeMenus = () => { setMenuOpen(false); setMoreOpen(false); };

  return (
    <div className="toolbar">
      <div className="title">ALIKE · Keypoint Matcher</div>

      {/* генерация точек: иконка + чип метода */}
      <div className="genwrap" ref={genRef}>
        <button
          className={'ibtn gen' + (menuOpen ? ' active' : '') + (busy ? ' busy' : '')}
          onClick={() => setMenuOpen(o => !o)}
          disabled={!canGenerate || busy}
          title="Сгенерировать точки — выберите метод, детект запустится сразу для обеих картинок"
        >
          <IcPoints />
          <span className="gen-method">{current ? (current.label || current.id) : ''}</span>
          <span className="caret">▾</span>
        </button>

        {menuOpen && (
          <div className="method-menu">
            {Object.entries(groups).map(([group, items]) => (
              <div key={group} className="method-menu-group">
                <span className="group-label">{group}</span>
                {items.map(m => (
                  <button
                    key={m.id}
                    className={'method-item' + (method === m.id ? ' active' : '')}
                    title={m.description || ''}
                    onClick={() => pickMethod(m.id)}
                  >
                    {m.label || m.id}
                    {method === m.id && <span className="method-current">текущий</span>}
                  </button>
                ))}
              </div>
            ))}
          </div>
        )}
      </div>

      {/* режимы — компактные иконки-тумблеры */}
      <button
        className={'ibtn' + (connectMode ? ' active' : '')}
        onClick={onToggleConnect}
        disabled={busy}
        title="Соединить: клик по точке на A, затем по точке на B"
        aria-label="Соединить"
      >
        <IcLink />
      </button>

      <button
        className={'ibtn' + (deformMode ? ' active' : '')}
        onClick={onToggleDeform}
        disabled={busy}
        title={deformMode
          ? 'Режим деформации: клик по точке — исключить её пару из триангуляции ' +
            '(пары и JSON не меняются); повторный клик — вернуть. Esc или кнопка — выйти'
          : 'Режим деформации: клик по точке исключает её пару из триангуляции'}
        aria-label="Режим деформации"
      >
        <IcDeform />
      </button>

      <button
        className={'ibtn' + (editMode ? ' active' : '')}
        onClick={onToggleEdit}
        disabled={busy}
        title={editMode
          ? 'Режим правки: выбранная точка мельчает (видно под ней), координаты ' +
            'в HUD сверху; стрелки ±1px, Shift+стрелки ±10px. Esc или кнопка — выйти'
          : 'Режим правки: выбранная точка мельчает, координаты в HUD, движение стрелками'}
        aria-label="Режим правки"
      >
        <IcEdit />
      </button>

      <button
        className={'ibtn' + (overlayVisible ? ' active' : '')}
        onClick={onToggleOverlay}
        disabled={!canOverlay && !overlayVisible}
        title={overlayTitle}
        aria-label="Наложить правое на левое"
      >
        <IcLayers />
      </button>

      {overlayVisible && (
        <div className="ctxbar">
          <label className="alpha-ctl" title="Прозрачность наложения правого изображения">
            <span>α</span>
            <input
              type="range" min="0" max="1" step="0.05"
              value={overlayAlpha}
              onChange={(e) => setOverlayAlpha(Number(e.target.value))}
            />
          </label>
        </div>
      )}

      <button
        className={'ibtn' + (rwarpVisible ? ' active' : '')}
        onClick={onToggleRWarp}
        disabled={!canRWarp && !rwarpVisible}
        title={rwarpTitle}
        aria-label="Наложить деформированное правое на правое"
      >
        <IcMesh />
      </button>

      {rwarpVisible && (
        <div className="ctxbar">
          <label className="alpha-ctl" title="Прозрачность наложения warp поверх правого изображения">
            <span>α</span>
            <input
              type="range" min="0" max="1" step="0.05"
              value={rwarpAlpha}
              onChange={(e) => setRWarpAlpha(Number(e.target.value))}
            />
          </label>
        </div>
      )}

      <button
        className={'ibtn' + (maskVisible ? ' active' : '')}
        onClick={onToggleMask}
        disabled={!canMask}
        title={maskTitle}
        aria-label="Маски Mask2Former"
      >
        <IcMask />
      </button>

      {maskVisible && (
        <div className="ctxbar">
          <label className="alpha-ctl" title="Прозрачность масок Mask2Former на правой картинке">
            <span>маски</span>
            <input
              type="range" min="0" max="1" step="0.05"
              value={maskAlpha}
              onChange={(e) => setMaskAlpha(Number(e.target.value))}
            />
          </label>
        </div>
      )}

      <button
        className={'ibtn' + (soloMode ? ' active' : '')}
        onClick={onToggleSolo}
        title={soloMode
          ? 'Solo: показать только выбранную пару; повторное нажатие — вернуть все точки. Esc — выйти'
          : 'Отключить все точки, кроме выбранной пары (пары без выделения — всё скрыто)'}
        aria-label="Только выбранная пара"
      >
        <IcSolo />
      </button>

      {/* фильтр точек после масок */}
      <button
        className={'ibtn' + (filterOn ? ' active' : '')}
        onClick={() => {
          // свёрнутый фильтр: клик по воронке разворачивает панель, не выключает
          if (filterOn && !filterDirty) onExpandFilter();
          else onToggleFilter();
        }}
        disabled={!canFilter}
        title={filterTitle}
        aria-label="Фильтр точек"
      >
        <IcFilter />
      </button>

      {/* свёрнутое состояние: фильтр применён, показываем только счётчик */}
      {filterOn && filterInfo && !filterDirty && (
        <button
          className="filter-count filter-count-btn"
          onClick={onExpandFilter}
          title="Фильтр применён — показать панель настроек"
        >
          {filterInfo.keptPairs}/{filterInfo.totalPairs} пар
        </button>
      )}

      {filterOn && filterInfo && filterDirty && (
        <div className="ctxbar class-filter-controls">
          <details className="class-picker" open>
            <summary>Запрещённые классы · {excludedClasses.length}</summary>
            <div className="class-picker-body">
              <p>Точки правого снимка внутри отмеченных классов скрываются вместе с парой слева.</p>
              <div className="class-options">
                {semanticClasses.map(c => (
                  <label key={c.id}>
                    <input type="checkbox" checked={excludedClasses.includes(c.id)}
                      onChange={e => setExcludedClasses(prev => e.target.checked
                        ? [...prev, c.id] : prev.filter(id => id !== c.id))} />
                    <i style={{ background: c.color }} />
                    <span>{c.label}</span>
                    {!c.area && <small>нет на снимке</small>}
                  </label>
                ))}
              </div>
              <label className="margin-input">
                Марджин, px
                <input type="number" min="0" step="0.5" aria-label="Марджин, px"
                  value={classMargin}
                  onChange={e => {
                    const n = Number(e.target.value);
                    if (Number.isFinite(n) && n >= 0) setClassMargin(n);
                  }} />
              </label>
              <p>Сохранять точку в запрещённом классе, если до разрешённого пикселя не больше марджина. 0 — строгий запрет.</p>
              <label className="geometry-option">
                <input type="checkbox" checked={geometryFilterOn}
                  onChange={e => setGeometryFilterOn(e.target.checked)} />
                Дополнительно: только границы или важные точки
              </label>
              <div className="class-filter-status" role="status">
                {classFilterError || (classFilterBusy ? 'Пересчитываем фильтр…'
                  : `Осталось ${filterInfo.keptPairs}/${filterInfo.totalPairs} пар · марджин сохранил ${marginSavedCount} точек`)}
              </div>
            </div>
          </details>
          {geometryFilterOn && <>
          <label
            className="alpha-ctl"
            title={filterReady
              ? 'Оставлять пары с нормированной деформируемостью не ниже порога'
              : 'Деформируемость ещё не посчитана — работает только фильтр по границам'}
          >
            <span className="ctl-note">важность ≥</span>
            <input
              type="range" min="0" max="1" step="0.05"
              value={filterImpThr}
              disabled={!filterReady}
              onChange={(e) => setFilterImpThr(Number(e.target.value))}
            />
            <span className="ctl-val">{filterImpThr.toFixed(2)}</span>
          </label>
          <label className="alpha-ctl" title="Радиус вокруг границы области, px исходного кадра">
            <span className="ctl-note">граница ±</span>
            <input
              type="range" min="0" max="16" step="1"
              value={filterRadius}
              onChange={(e) => setFilterRadius(Number(e.target.value))}
            />
            <span className="ctl-val">{filterRadius}px</span>
          </label>
          </>}
          <label
            className="alpha-ctl"
            title={filterReady
              ? 'Удалить пары с наименьшей деформируемостью — после каждой пары ' +
                'важность пересчитывается заново; ползунок влево восстанавливает'
              : 'Деформируемость ещё не посчитана — ползунок недоступен'}
          >
            <span className="ctl-note">удалить слабые</span>
            <input
              type="range" min="0" max={pruneMax} step="1"
              value={Math.min(pruneTarget, pruneMax)}
              disabled={!filterReady || pruneBusy}
              onChange={(e) => setPruneTarget(Number(e.target.value))}
            />
            <span className="ctl-val" title="Удалённых пар">
              {pruneBusy && pruneProgress
                ? `…${pruneProgress.i}/${pruneProgress.n}`
                : `${pruneTarget}`}
            </span>
          </label>
          <span className="filter-count" title="Пар, оставшихся после фильтра">
            {filterInfo.keptPairs}/{filterInfo.totalPairs} пар
          </span>
          <button
            className="apply-btn"
            onClick={onApplyFilter}
            title="Применить настройки и свернуть панель — фильтр останется активным"
          >
            Применить
          </button>
        </div>
      )}

      {/* проверка с эталоном: кнопка активна всегда — эталон запоминается
          на сессию, метрики появятся, как только будет наложение */}
      <button
        className={'ibtn' + (checkBusy ? ' busy' : '')}
        onClick={() => checkInputRef.current?.click()}
        title={checkTitle}
        aria-label="Проверить с эталоном"
      >
        <IcCheck />
      </button>
      <input
        ref={checkInputRef}
        type="file"
        accept="image/*,.ppm,.pgm,.pnm"
        hidden
        onChange={(e) => {
          const f = e.target.files?.[0];
          if (f) onCheckFile(f);
          e.target.value = '';
        }}
      />

      {/* всё второстепенное — в одном меню */}
      <div className="menuwrap" ref={moreRef}>
        <button
          className={'ibtn' + (moreOpen ? ' active' : '')}
          onClick={() => setMoreOpen(o => !o)}
          title="Ещё: JSON, сохранение, линии, сброс вида"
          aria-label="Ещё"
        >
          <IcMore />
        </button>

        {moreOpen && (
          <div className="more-menu">
            <label
              className="menu-item"
              title="Загрузить точки и пары из JSON"
            >
              Загрузить JSON
              <input
                type="file"
                accept=".json,application/json"
                hidden
                onChange={(e) => {
                  const f = e.target.files?.[0];
                  if (f) { closeMenus(); onJsonFile(f); }
                  e.target.value = '';
                }}
              />
            </label>

            <button
              className="menu-item"
              disabled={busy}
              onClick={() => { closeMenus(); onSaveJson(); }}
              title="Сохранить текущие точки и пары в JSON-файл"
            >
              Сохранить JSON
            </button>

            <div className="menu-sep" />

            <button
              className="menu-item"
              onClick={() => { closeMenus(); onSaveSnapshot(); }}
              title="Записать актуальный снимок в браузере (localStorage)"
            >
              Сохранить снимок · Ctrl+S
              {snapshotDirty && <span className="menu-dot" title="есть несохранённые правки">●</span>}
            </button>
            <button
              className="menu-item"
              disabled={!snapshotInitialAt}
              onClick={() => { closeMenus(); onRestoreSnapshot('initial'); }}
              title={snapshotInitialAt
                ? `Вернуть состояние при загрузке (${new Date(snapshotInitialAt).toLocaleTimeString()})`
                : 'Базового снимка ещё нет'}
            >
              К начальному{snapshotInitialAt
                ? ` · ${new Date(snapshotInitialAt).toLocaleTimeString()}` : ''}
            </button>
            <button
              className="menu-item"
              disabled={!snapshotActualAt}
              onClick={() => { closeMenus(); onRestoreSnapshot('actual'); }}
              title={snapshotActualAt
                ? `Вернуть последний сохранённый снимок (${new Date(snapshotActualAt).toLocaleTimeString()})`
                : 'Актуального снимка ещё нет'}
            >
              К актуальному{snapshotActualAt
                ? ` · ${new Date(snapshotActualAt).toLocaleTimeString()}` : ''}
            </button>
            <button
              className={'menu-item' + (autoSave ? ' on' : '')}
              onClick={() => { setAutoSave(v => !v); }}
              title="Автоматически записывать актуальный снимок после каждой правки (с задержкой)"
            >
              Автосохранение {autoSave && <span className="menu-check">✓</span>}
            </button>

            <div className="menu-sep" />

            <button
              className="menu-item"
              disabled={!canSave}
              onClick={() => { closeMenus(); onSaveImage('composite'); }}
              title={canSave ? 'Левое изображение + деформированное правое' : 'Сначала рассчитайте наложение'}
            >
              Сохранить композит PNG
            </button>
            <button
              className="menu-item"
              disabled={!canSave}
              onClick={() => { closeMenus(); onSaveImage('warped'); }}
              title={canSave ? 'Только деформированное правое изображение' : 'Сначала рассчитайте наложение'}
            >
              Сохранить деформированное PNG
            </button>

            <div className="menu-sep" />

            <button
              className={'menu-item' + (showAllLines ? ' on' : '')}
              onClick={() => { closeMenus(); onToggleAllLines(); }}
              title="Показать линии между всеми парами точек"
            >
              Все линии {showAllLines && <span className="menu-check">✓</span>}
            </button>
            <button
              className="menu-item"
              onClick={() => { closeMenus(); onReset(); }}
              title="Сбросить панорамирование и зум"
            >
              Сброс вида
            </button>
          </div>
        )}
      </div>
    </div>
  );
}
