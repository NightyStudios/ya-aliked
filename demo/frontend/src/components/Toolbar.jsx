import { visibleMethods } from '../cv/methods';

const FALLBACK = [{ id: 'orb_cv', label: 'orb_cv', group: 'ORB' }];

export function Toolbar({
  method, setMethod, methodsMeta, onReset, onFiles,
  status, cvReady, busy, showAllLines, onToggleAllLines,
  onJsonFile, onSaveJson, connectMode, onToggleConnect,
}) {
  const list = visibleMethods(methodsMeta);
  const entries = list.length ? list : FALLBACK;

  const groups = {};
  entries.forEach(m => {
    const g = m.group || 'ORB';
    (groups[g] = groups[g] || []).push(m);
  });

  return (
    <div className="toolbar">
      <div className="title">ALIKE · Keypoint Matcher</div>

      <label className="filebtn">
        Загрузить 2 изображения
        <input
          type="file"
          accept="image/*,.ppm,.pgm,.pnm"
          multiple hidden
          onChange={(e) => { onFiles(e.target.files); e.target.value = ''; }}
        />
      </label>

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

      <div className="methods">
        {Object.entries(groups).map(([group, items]) => (
          <div key={group} className="method-group">
            <span className="group-label">{group}</span>
            {items.map(m => (
              <button
                key={m.id}
                className={method === m.id ? 'active' : ''}
                disabled={busy}
                title={m.description || ''}
                onClick={() => setMethod(m.id)}
              >
                {m.label || m.id}
              </button>
            ))}
          </div>
        ))}
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

      <div className="status">
        {!cvReady && '⏳ загрузка OpenCV… '}
        {busy && '⏳ '}
        {status}
      </div>
    </div>
  );
}
