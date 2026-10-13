import { useEffect, useRef, useState } from 'react';
import { visibleMethods } from '../cv/methods';
import { GREEN_MAX_NORM } from '../warp';

const FALLBACK = [{ id: 'orb_cv', label: 'orb_cv', group: 'ORB' }];

export function Toolbar({
  method, setMethod, methodsMeta, onReset,
  busy, showAllLines, onToggleAllLines,
  onJsonFile, onSaveJson, connectMode, onToggleConnect,
  onGenerate, canGenerate,
  canOverlay, overlayVisible, onToggleOverlay,
  overlayAlpha, setOverlayAlpha,
  excludeTop, setExcludeTop, maxExclude, excludeReady,
  greenCount, pairCount,
  onSaveImage, canSave,
  maskVisible, onToggleMask, maskBusy, maskError, canMask,
  maskAlpha, setMaskAlpha, maskCount, maskMs,
}) {
  const list = visibleMethods(methodsMeta);
  const entries = list.length ? list : FALLBACK;

  const groups = {};
  entries.forEach(m => {
    const g = m.group || 'ORB';
    (groups[g] = groups[g] || []).push(m);
  });

  const current = entries.find(m => m.id === method) || entries[0];

  // --- выпадающие меню (методы генерации, сохранение)
  const [menuOpen, setMenuOpen] = useState(false);
  const [saveOpen, setSaveOpen] = useState(false);
  const genRef = useRef(null);
  const saveRef = useRef(null);

  useEffect(() => {
    if (!menuOpen && !saveOpen) return;
    const onDown = (e) => {
      if (genRef.current && !genRef.current.contains(e.target)) setMenuOpen(false);
      if (saveRef.current && !saveRef.current.contains(e.target)) setSaveOpen(false);
    };
    const onKey = (e) => {
      if (e.key !== 'Escape') return;
      setMenuOpen(false);
      setSaveOpen(false);
    };
    window.addEventListener('mousedown', onDown);
    window.addEventListener('keydown', onKey);
    return () => {
      window.removeEventListener('mousedown', onDown);
      window.removeEventListener('keydown', onKey);
    };
  }, [menuOpen, saveOpen]);

  useEffect(() => { if (busy) setMenuOpen(false); }, [busy]);
  useEffect(() => { if (!canSave) setSaveOpen(false); }, [canSave]);

  // клик по методу: выбрать его и сразу запустить детект
  const pickMethod = (id) => {
    setMenuOpen(false);
    if (id !== method) setMethod(id);
    onGenerate();
  };

  const overlayTitle = canOverlay
    ? `Наложить правое изображение на левое по ${greenCount} зелёным парам ` +
      `(деформируемость ≤ ${GREEN_MAX_NORM}` +
      (excludeTop ? `, не учтено ${excludeTop} самых деформируемых` : '') +
      `); пар всего: ${pairCount}`
    : 'Нужны обе картинки и ≥ 3 зелёных пар (деформируемость должна быть посчитана)';

  return (
    <div className="toolbar">
      <div className="title">ALIKE · Keypoint Matcher</div>

      <label className="filebtn" title="Загрузить точки и пары из JSON">
        Загрузить JSON
        <input
          type="file"
          accept=".json,application/json"
          hidden
          onChange={(e) => {
            const f = e.target.files?.[0];
            if (f) onJsonFile(f);
            e.target.value = '';
          }}
        />
      </label>

      <button onClick={onSaveJson} disabled={busy}
              title="Сохранить текущие точки и пары в JSON">
        Сохранить JSON
      </button>

      <button
        className={connectMode ? 'active' : ''}
        onClick={onToggleConnect}
        disabled={busy}
        title="Режим соединения: клик по точке на A, затем по точке на B"
      >
        Соединить
      </button>

      <div className="genwrap" ref={genRef}>
        <button
          className={'generate' + (menuOpen ? ' active' : '')}
          onClick={() => setMenuOpen(o => !o)}
          disabled={!canGenerate || busy}
          title="Выберите метод — детект запустится сразу для обеих картинок"
        >
          {busy ? 'Генерация…' : 'Сгенерировать точки'}
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

      <button
        className={overlayVisible ? 'active' : ''}
        onClick={onToggleOverlay}
        disabled={!canOverlay && !overlayVisible}
        title={overlayTitle}
      >
        {overlayVisible ? 'Скрыть наложение' : 'Наложить правое на левое'}
      </button>

      {overlayVisible && (
        <label className="alpha-ctl" title="Прозрачность наложения правого изображения">
          <span>α</span>
          <input
            type="range" min="0" max="1" step="0.05"
            value={overlayAlpha}
            onChange={(e) => setOverlayAlpha(Number(e.target.value))}
          />
        </label>
      )}

      {overlayVisible && (
        <label
          className="alpha-ctl excl-ctl"
          title={'Не учитывать при наложении N самых деформируемых пар: ' +
                 `сейчас отброшено ${excludeTop}, зелёных останется ${greenCount}`}
        >
          <span className="ctl-note">не учитывать</span>
          <input
            type="range" min="0" max={maxExclude} step="1"
            value={excludeTop}
            disabled={!excludeReady}
            onChange={(e) => setExcludeTop(Number(e.target.value))}
          />
          <span className="ctl-val">{excludeTop}</span>
        </label>
      )}

      <button
        className={maskVisible ? 'active' : ''}
        onClick={onToggleMask}
        disabled={!canMask}
        title={
          !canMask
            ? 'Сначала загрузите правое изображение'
            : maskBusy
              ? 'Mask2Former считает маски… (первый запрос может качать веса модели)'
              : maskError
                ? `Ошибка: ${maskError}`
                : maskVisible
                  ? `Скрыть маски Mask2Former · инстансов: ${maskCount}` +
                    (maskMs != null ? ` · ${Math.round(maskMs)} мс` : '')
                  : 'Показать маски Mask2Former поверх правого изображения'
        }
      >
        {maskBusy ? 'Mask2Former…' : 'Mask2Former'}
      </button>

      {maskVisible && (
        <label className="alpha-ctl" title="Прозрачность масок Mask2Former на правой картинке">
          <span className="ctl-note">маски</span>
          <input
            type="range" min="0" max="1" step="0.05"
            value={maskAlpha}
            onChange={(e) => setMaskAlpha(Number(e.target.value))}
          />
        </label>
      )}

      <div className="savewrap" ref={saveRef}>
        <button
          className={saveOpen ? 'active' : ''}
          onClick={() => setSaveOpen(o => !o)}
          disabled={!canSave}
          title={canSave
            ? 'Сохранить PNG: композит с наложением или только деформированное правое'
            : 'Сначала рассчитайте наложение'}
        >
          Сохранить изображение <span className="caret">▾</span>
        </button>

        {saveOpen && (
          <div className="method-menu">
            <button className="method-item"
                    onClick={() => { setSaveOpen(false); onSaveImage('composite'); }}>
              Композит (с наложением)
            </button>
            <button className="method-item"
                    onClick={() => { setSaveOpen(false); onSaveImage('warped'); }}>
              Только деформированное правое
            </button>
          </div>
        )}
      </div>

      <button
        className={showAllLines ? 'active' : ''}
        onClick={onToggleAllLines}
        disabled={busy}
        title="Показать линии между всеми парами точек"
      >
        Все линии
      </button>

      <button onClick={onReset} disabled={busy}>Сброс вида</button>
    </div>
  );
}
