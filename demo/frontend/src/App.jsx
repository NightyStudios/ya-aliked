import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { Toolbar } from './components/Toolbar';
import { Panel } from './components/Panel';
import { MatchList } from './components/MatchList';
import { ConnectionLine } from './components/ConnectionLine';
import { useOpenCV } from './hooks/useOpenCV';
import { usePanZoom } from './hooks/usePanZoom';
import { useMatcher } from './hooks/useMatcher';
import { loadImage, screenPoint } from './cv/loadImage';
import { fetchMethods } from './cv/methods';
import { pairColor } from './constants';
import { clampKp, removePoint, setPair, unmatchedIdx } from './points';
import { parsePointsJson, serializePointsJson, downloadJson } from './jsonPoints';

export default function App() {
  const cvReady = useOpenCV();
  const { tf, setTf, onMouseDown, makeWheelHandler, wasDrag, reset } = usePanZoom();

  const [imgA, setImgA] = useState(null);
  const [imgB, setImgB] = useState(null);

  const [methodsMeta, setMethodsMeta] = useState({});
  const [method, setMethod] = useState('orb_cv');
  const [params, setParams] = useState({});   // новая ссылка = форс-перезапуск детекта

  const [hovered, setHovered] = useState(null);
  const [selected, setSelected] = useState(null);
  const [line, setLine] = useState(null);
  const [showAllLines, setShowAllLines] = useState(false);
  const [allLines, setAllLines] = useState([]);

  // --- редактируемое состояние точек/пар (переопределяет результат детекта)
  const [kpsA, setKpsA] = useState([]);
  const [kpsB, setKpsB] = useState([]);
  const [matches, setMatches] = useState([]);

  // --- режим соединения
  const [connectMode, setConnectMode] = useState(false);
  const [connectFrom, setConnectFrom] = useState(null);

  // --- собственный статус (JSON-операции), перекрывает статус детекта
  const [notice, setNotice] = useState('');

  const panelARef = useRef(null);
  const panelBRef = useRef(null);

  const lastJsonLoadAtRef = useRef(0);
  const pendingJsonRef = useRef(null);
  // JSON активен: авто-детект (поздний cvReady, смена картинок) его не затирает;
  // только явный клик по кнопке метода возвращает управление детекту.
  const jsonSuppressRef = useRef(false);
  const explicitDetectRef = useRef(false);
  const kpDragRef = useRef(null);
  const kpDragMoved = useRef(false);
  const tfRef = useRef(tf);
  useEffect(() => { tfRef.current = tf; });

  // --- метаданные методов с бэка
  useEffect(() => {
    fetchMethods()
      .then(m => {
        setMethodsMeta(m.detector_meta || {});
        const meta = m.detector_meta || {};
        // если дефолт недоступен/скрыт — переключиться на первый видимый
        const cur = meta[method];
        if (!cur || !cur.available || cur.hidden) {
          const first = Object.entries(meta)
            .find(([_, v]) => v.available && !v.hidden);
          if (first) setMethod(first[0]);
        }
      })
      .catch(e => console.error('fetchMethods failed:', e));
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  // --- пайплайн
  const det = useMatcher({ cvReady, imgA, imgB, method, params });
  const { status, busy } = det;

  // --- применение результата детекта (не затирает свежий JSON)
  useEffect(() => {
    const explicit = explicitDetectRef.current;
    explicitDetectRef.current = false;
    if (!det.runId) return;

    if (pendingJsonRef.current) {
      const p = pendingJsonRef.current;
      pendingJsonRef.current = null;
      applyParsed(p, 'JSON загружен: ');
      return;
    }

    if (det.startedAt <= lastJsonLoadAtRef.current) return;   // JSON новее детекта
    if (!explicit && jsonSuppressRef.current) return;         // JSON активен, авто-запуск

    jsonSuppressRef.current = false;
    setKpsA(det.kpsA);
    setKpsB(det.kpsB);
    setMatches(det.matches);
    setSelected(null);
    setConnectFrom(null);
    setNotice('');
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [det.runId]);

  // --- idx -> match idx, для раскраски пар
  const matchColorIdx = useMemo(() => {
    const a = new Map(), b = new Map();
    matches.forEach((m, i) => { a.set(m.aIdx, i); b.set(m.bIdx, i); });
    return { a, b };
  }, [matches]);

  const findMatch = useCallback((side, idx) =>
    matches.find(m => (side === 'a' ? m.aIdx : m.bIdx) === idx) || null,
  [matches]);

  // --- номера пар для нумерации точек (1..N в порядке matches)
  const pairNo = useMemo(() => {
    const a = new Map(), b = new Map();
    matches.forEach((m, i) => { a.set(m.aIdx, i + 1); b.set(m.bIdx, i + 1); });
    return { a, b };
  }, [matches]);

  const isActive = useCallback((side, idx) => {
    if (!hovered) return true;
    if (hovered.side === side && hovered.idx === idx) return true;
    const m = findMatch(hovered.side, hovered.idx);
    if (!m) return false;
    const otherIdx = hovered.side === 'a' ? m.bIdx : m.aIdx;
    return side !== hovered.side && idx === otherIdx;
  }, [hovered, findMatch]);

  // --- авто-фит при загрузке
  useEffect(() => {
    if (!imgA || !imgB) return;
    const pa = panelARef.current, pb = panelBRef.current;
    if (!pa || !pb) return;
    const s = Math.min(pa.clientWidth / imgA.width, pb.clientWidth / imgB.width) * 0.92;
    setTf({ s, tx: 10, ty: 10 });
    setSelected(null);
  }, [imgA, imgB, setTf]);

  // --- линия между выбранной точкой и её парой
  useEffect(() => {
    if (!selected) { setLine(null); return; }
    const m = findMatch(selected.side, selected.idx);
    if (!m) { setLine(null); return; }

    const update = () => {
      const pa = panelARef.current, pb = panelBRef.current;
      if (!pa || !pb) return;
      const ka = kpsA[m.aIdx], kb = kpsB[m.bIdx];
      if (!ka || !kb) return;
      const p1 = screenPoint(pa, tf, ka);
      const p2 = screenPoint(pb, tf, kb);
      setLine({ x1: p1.x, y1: p1.y, x2: p2.x, y2: p2.y });
    };
    update();
    window.addEventListener('resize', update);
    return () => window.removeEventListener('resize', update);
  }, [selected, tf, matches, kpsA, kpsB, findMatch]);

  const consumePointDrag = () => {
    if (kpDragMoved.current) { kpDragMoved.current = false; return true; }
    return false;
  };

  const handlePanelClick = useCallback(() => {
    if (wasDrag() || kpDragMoved.current) { kpDragMoved.current = false; return; }
    setSelected(null);
    setConnectFrom(null);
  }, [wasDrag]);

  // --- перетаскивание точки
  const startPointDrag = useCallback((side, idx, e) => {
    if (e.button !== 0) return;
    const img = side === 'a' ? imgA : imgB;
    const kp = (side === 'a' ? kpsA : kpsB)[idx];
    if (!img || !kp) return;
    kpDragMoved.current = false;
    kpDragRef.current = { side, idx, img, kp, x: e.clientX, y: e.clientY, moved: false };
  }, [imgA, imgB, kpsA, kpsB]);

  useEffect(() => {
    const onMove = (e) => {
      const d = kpDragRef.current;
      if (!d) return;
      const dx = e.clientX - d.x, dy = e.clientY - d.y;
      if (!d.moved && Math.hypot(dx, dy) < 3) return;
      d.moved = true;
      const s = tfRef.current.s;
      const np = clampKp({ x: d.kp.x + dx / s, y: d.kp.y + dy / s }, d.img);
      if (d.side === 'a') setKpsA(prev => prev.map((k, i) => (i === d.idx ? np : k)));
      else setKpsB(prev => prev.map((k, i) => (i === d.idx ? np : k)));
    };
    const onUp = () => {
      if (!kpDragRef.current) return;
      kpDragMoved.current = kpDragRef.current.moved;
      kpDragRef.current = null;
    };
    window.addEventListener('mousemove', onMove);
    window.addEventListener('mouseup', onUp);
    return () => {
      window.removeEventListener('mousemove', onMove);
      window.removeEventListener('mouseup', onUp);
    };
  }, []);

  // --- клик по точке: выбор или создание пары (режим «Соединить»)
  const handlePointClick = useCallback((side, idx) => {
    if (consumePointDrag()) return;

    if (connectMode) {
      if (!connectFrom) { setConnectFrom({ side, idx }); return; }
      if (connectFrom.side === side && connectFrom.idx === idx) {
        setConnectFrom(null);               // повторный клик — отмена
        return;
      }
      if (connectFrom.side === side) {
        setConnectFrom({ side, idx });      // та же картинка — перенос выбора
        return;
      }
      const aIdx = side === 'a' ? idx : connectFrom.idx;
      const bIdx = side === 'b' ? idx : connectFrom.idx;
      setMatches(m => setPair(m, aIdx, bIdx));
      setConnectFrom(null);
      setSelected({ side: 'a', idx: aIdx });
      return;
    }

    setSelected(s =>
      (s && s.side === side && s.idx === idx) ? null : { side, idx });
  }, [connectMode, connectFrom]);

  // --- двойной клик по пустому месту — новая точка
  const handlePanelDblClick = useCallback((side, e) => {
    if (kpDragMoved.current || wasDrag()) { kpDragMoved.current = false; return; }
    const img = side === 'a' ? imgA : imgB;
    const panel = side === 'a' ? panelARef.current : panelBRef.current;
    if (!img || !panel) return;

    const rect = panel.getBoundingClientRect();
    const { s, tx, ty } = tfRef.current;
    const kp = clampKp({
      x: (e.clientX - rect.left - tx) / s,
      y: (e.clientY - rect.top - ty) / s,
    }, img);

    const arr = side === 'a' ? kpsA : kpsB;
    const idx = arr.length;
    if (side === 'a') setKpsA([...arr, kp]);
    else setKpsB([...arr, kp]);
    setSelected({ side, idx });
    setConnectFrom(null);
  }, [imgA, imgB, kpsA, kpsB, wasDrag]);

  // --- Delete/Backspace — удалить выбранную точку; Escape — отмена пары
  useEffect(() => {
    const onKey = (e) => {
      if (e.key === 'Escape') { setConnectFrom(null); return; }
      if (e.key !== 'Delete' && e.key !== 'Backspace') return;
      if (!selected) return;
      e.preventDefault();

      const { side, idx } = selected;
      const kps = side === 'a' ? kpsA : kpsB;
      const { kps: newKps, matches: newMatches } = removePoint(kps, matches, side, idx);
      if (side === 'a') setKpsA(newKps);
      else setKpsB(newKps);
      setMatches(newMatches);
      setSelected(null);
      setHovered(h => (h && h.side === side && h.idx === idx) ? null : h);
      setConnectFrom(c => (c && c.side === side && c.idx === idx) ? null : c);
    };
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, [selected, kpsA, kpsB, matches]);

  // --- все линии матчей (тумблер в тулбаре)
  useEffect(() => {
    if (!showAllLines) { setAllLines([]); return; }

    const update = () => {
      const pa = panelARef.current, pb = panelBRef.current;
      if (!pa || !pb) return;
      const out = [];
      matches.forEach((m, i) => {
        const ka = kpsA[m.aIdx], kb = kpsB[m.bIdx];
        if (!ka || !kb) return;
        const p1 = screenPoint(pa, tf, ka);
        const p2 = screenPoint(pb, tf, kb);
        out.push({ x1: p1.x, y1: p1.y, x2: p2.x, y2: p2.y, color: pairColor(i) });
      });
      setAllLines(out);
    };
    update();
    window.addEventListener('resize', update);
    return () => window.removeEventListener('resize', update);
  }, [showAllLines, tf, matches, kpsA, kpsB]);

  const handleFiles = useCallback(async (files) => {
    const arr = Array.from(files).filter(f => f.type.startsWith('image/')).slice(0, 2);
    if (!arr.length) return;
    const imgs = await Promise.all(arr.map(loadImage));
    // новый контекст изображений — JSON-точки (от старых картинок) больше не авторитетны
    jsonSuppressRef.current = false;
    if (imgs[0]) setImgA(imgs[0]);
    if (imgs[1]) setImgB(imgs[1]);
  }, []);

  // клик по кнопке метода = явное «пере-детектировать» (форс-перезапуск через params)
  const handleMethodChange = useCallback((m) => {
    explicitDetectRef.current = true;
    setMethod(m);
    setParams({});
  }, []);

  // --- JSON: применить распарсенные точки
  const applyParsed = useCallback((p, prefix = '') => {
    let warn = '';
    if (imgA && p.imageA && imgA.name !== p.imageA) warn += ` · A: «${p.imageA}» ≠ «${imgA.name}»`;
    if (imgB && p.imageB && imgB.name !== p.imageB) warn += ` · B: «${p.imageB}» ≠ «${imgB.name}»`;

    setKpsA(p.kpsA);
    setKpsB(p.kpsB);
    setMatches(p.matches);
    setSelected(null);
    setConnectFrom(null);
    lastJsonLoadAtRef.current = Date.now();
    jsonSuppressRef.current = true;
    setNotice(`${prefix}${p.counts.pairs} пар · ${p.counts.ua}/${p.counts.ub} непарных${warn}`);
  }, [imgA, imgB]);

  const handleJsonFile = useCallback(async (file) => {
    try {
      const text = await file.text();
      const parsed = parsePointsJson(text);
      if (imgA && imgB) {
        pendingJsonRef.current = null;
        applyParsed(parsed);
      } else {
        // изображений ещё нет — применим, когда придёт результат детекта
        pendingJsonRef.current = parsed;
        setNotice('JSON принят — загрузите обе картинки, чтобы увидеть точки');
      }
    } catch (e) {
      setNotice('Ошибка JSON: ' + e.message);
    }
  }, [imgA, imgB, applyParsed]);

  const handleSaveJson = useCallback(() => {
    if (!kpsA.length && !kpsB.length) { setNotice('Нет точек для сохранения'); return; }
    const text = serializePointsJson({
      kpsA, kpsB, matches,
      imageA: imgA ? imgA.name : null,
      imageB: imgB ? imgB.name : null,
    });
    const base = (imgA ? imgA.name : 'keypoints').replace(/\.[^.]+$/, '') || 'keypoints';
    downloadJson(text, `${base}_points.json`);
    const ua = unmatchedIdx(kpsA, matches, 'a').length;
    const ub = unmatchedIdx(kpsB, matches, 'b').length;
    setNotice(`Сохранено: ${matches.length} пар · ${ua}/${ub} непарных`);
  }, [kpsA, kpsB, matches, imgA, imgB]);

  const toggleConnect = useCallback(() => {
    setConnectMode(v => !v);
    setConnectFrom(null);
  }, []);

  return (
    <div className="app">
      <Toolbar
        method={method}
        setMethod={handleMethodChange}
        methodsMeta={methodsMeta}
        onReset={reset}
        onFiles={handleFiles}
        status={notice || status}
        cvReady={cvReady}
        busy={busy}
        showAllLines={showAllLines}
        onToggleAllLines={() => setShowAllLines(v => !v)}
        onJsonFile={handleJsonFile}
        onSaveJson={handleSaveJson}
        connectMode={connectMode}
        onToggleConnect={toggleConnect}
      />

      <div className="panels">
        <Panel
          side="a" img={imgA} kps={kpsA} tf={tf} panelRef={panelARef}
          onWheel={makeWheelHandler(() => panelARef.current)}
          onMouseDown={onMouseDown} onPanelClick={handlePanelClick}
          onPanelDblClick={handlePanelDblClick}
          hovered={hovered} setHovered={setHovered}
          selected={selected}
          colorMap={matchColorIdx.a} isActive={isActive} onFiles={handleFiles}
          pairNo={pairNo.a}
          connectFrom={connectFrom}
          onPointClick={handlePointClick}
          onStartDrag={startPointDrag}
        />
        <Panel
          side="b" img={imgB} kps={kpsB} tf={tf} panelRef={panelBRef}
          onWheel={makeWheelHandler(() => panelBRef.current)}
          onMouseDown={onMouseDown} onPanelClick={handlePanelClick}
          onPanelDblClick={handlePanelDblClick}
          hovered={hovered} setHovered={setHovered}
          selected={selected}
          colorMap={matchColorIdx.b} isActive={isActive} onFiles={handleFiles}
          pairNo={pairNo.b}
          connectFrom={connectFrom}
          onPointClick={handlePointClick}
          onStartDrag={startPointDrag}
        />

        <MatchList
          matches={matches} kpsA={kpsA} kpsB={kpsB}
          selected={selected} setSelected={setSelected}
          setHovered={setHovered}
        />
      </div>

      <ConnectionLine lines={allLines} line={line} />
    </div>
  );
}
