import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { Toolbar } from './components/Toolbar';
import { Panel } from './components/Panel';
import { PointsPanel } from './components/PointsPanel';
import { ConnectionLine } from './components/ConnectionLine';
import { useOpenCV } from './hooks/useOpenCV';
import { usePanZoom } from './hooks/usePanZoom';
import { useMatcher } from './hooks/useMatcher';
import { useClassFilter } from './hooks/useClassFilter';
import { loadImage, screenPoint } from './cv/loadImage';
import { fetchMethods } from './cv/methods';
import { fetchSegmentation } from './cv/segment';
import { pairColor } from './constants';
import { clampKp, removePoint, setPair, unmatchedIdx } from './points';
import { parsePointsJson, serializePointsJson, downloadJson } from './jsonPoints';
import { computeHiddenSets } from './filterPoints';
import {
  computeImportanceRange, normalize, buildPairArrays,
  MIN_POINTS_FOR_IMPORTANCE,
} from './importance';
import {
  selectGreenPairs, greenPairCount, warpRightToLeft,
  compositeOverlay, downloadCanvas,
} from './warp';

export default function App() {
  const cvReady = useOpenCV();
  const { tf, setTf, onMouseDown, makeWheelHandler, wasDrag, reset } = usePanZoom();

  const [imgA, setImgA] = useState(null);
  const [imgB, setImgB] = useState(null);

  const [methodsMeta, setMethodsMeta] = useState({});
  const [method, setMethod] = useState('orb_cv');
  const [params] = useState({});   // задел под слайдеры параметров

  const [hovered, setHovered] = useState(null);
  const [selected, setSelected] = useState(null);
  const [line, setLine] = useState(null);
  const [showAllLines, setShowAllLines] = useState(false);
  const [allLines, setAllLines] = useState([]);

  // --- редактируемое состояние точек/пар
  const [kpsA, setKpsA] = useState([]);
  const [kpsB, setKpsB] = useState([]);
  const [matches, setMatches] = useState([]);
  // откуда точки: 'det' (детект) | 'json' (загружены из файла) | null
  const [pointsSource, setPointsSource] = useState(null);

  // --- явный запуск детекта (кнопка «Сгенерировать точки»)
  const [genTrigger, setGenTrigger] = useState(0);

  // --- режим соединения
  const [connectMode, setConnectMode] = useState(false);
  const [connectFrom, setConnectFrom] = useState(null);

  // --- правая панель: два независимых сайдбара
  const [pairsOpen, setPairsOpen] = useState(false);
  const [impOpen, setImpOpen] = useState(false);

  // --- деформируемость точек (триангуляция Делоне, левое — эталон)
  const [importance, setImportance] = useState(null);   // {values, norms} по индексам matches
  const [importanceBusy, setImportanceBusy] = useState(false);
  const impGenRef = useRef(0);

  const panelARef = useRef(null);
  const panelBRef = useRef(null);
  const panelsMainRef = useRef(null);

  const lastJsonLoadAtRef = useRef(0);
  const kpDragRef = useRef(null);
  const kpDragMoved = useRef(false);
  const tfRef = useRef(tf);
  useEffect(() => { tfRef.current = tf; });

  // --- текущее перетаскивание точки {side, idx}: пока тянешь —
  // ховер соседних точек игнорируется, а перетаскиваемая рисуется сверху
  const [dragging, setDragging] = useState(null);

  // --- наложение правого изображения на левое (по зелёным парам)
  const [overlayVisible, setOverlayVisible] = useState(false);
  const [overlayAlpha, setOverlayAlpha] = useState(0.5);
  const [overlay, setOverlay] = useState(null);        // { warped, green, total }
  const overlayLayerRef = useRef(null);
  const overlayGenRef = useRef(0);

  // --- маски Mask2Former поверх правой картинки
  const [maskVisible, setMaskVisible] = useState(false);
  const [maskBusy, setMaskBusy] = useState(false);
  const [maskError, setMaskError] = useState(null);
  const [maskAlpha, setMaskAlpha] = useState(0.75);
  const [maskData, setMaskData] = useState(null);   // { imgB, el, boundary, instances, elapsedMs }
  const maskLayerRef = useRef(null);
  const maskGenRef = useRef(0);

  // --- фильтр точек после Mask2Former: границы областей ∪ важные для деформации
  const [filterOn, setFilterOn] = useState(false);
  const [filterImpThr, setFilterImpThr] = useState(0.5);   // порог нормы деформируемости
  const [filterRadius, setFilterRadius] = useState(4);     // радиус границы, px полного кадра
  const [excludedClasses, setExcludedClasses] = useState([]);
  const [classMargin, setClassMargin] = useState(5);
  const [geometryFilterOn, setGeometryFilterOn] = useState(false);

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

  // --- пайплайн (запускается только кнопкой генерации)
  const det = useMatcher({ cvReady, imgA, imgB, method, params, trigger: genTrigger });
  const { busy } = det;

  // --- применение результата детекта
  useEffect(() => {
    if (!det.runId) return;
    if (det.startedAt <= lastJsonLoadAtRef.current) return;   // JSON загружен во время детекта
    if (det.runImgA !== imgA || det.runImgB !== imgB) {       // картинки сменили за время детекта
      console.warn('Изображения изменились — результат детекта не применён');
      return;
    }
    setKpsA(det.kpsA);
    setKpsB(det.kpsB);
    setMatches(det.matches);
    setPointsSource('det');
    setSelected(null);
    setConnectFrom(null);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [det.runId]);

  const handleGenerate = useCallback(() => setGenTrigger(t => t + 1), []);

  // --- расчёт деформируемости: дебаунс 300мс + чанками, чтобы не блокировать UI.
  // Стейт понимает только по длине: пока matches не изменился — старые значения
  // остаются валидными (правки координат делают их лишь слегка устаревшими).
  useEffect(() => {
    const gen = ++impGenRef.current;
    const pairs = matches.length >= MIN_POINTS_FOR_IMPORTANCE
      ? buildPairArrays(matches, kpsA, kpsB, imgA)
      : null;
    if (!pairs) {
      setImportance(null);
      setImportanceBusy(false);
      return;
    }
    setImportanceBusy(true);
    const timer = setTimeout(async () => {
      const { left, right, corners } = pairs;
      const M = left.length;
      const values = new Float64Array(M);
      const CH = 20;
      for (let s = 0; s < M; s += CH) {
        if (impGenRef.current !== gen) return;          // устарело — новая правка
        computeImportanceRange(left, right, corners, s, Math.min(M, s + CH), values);
        await new Promise(r => setTimeout(r, 0));       // отдых главному потоку
      }
      if (impGenRef.current !== gen) return;
      setImportance({ values, norms: normalize(values) });
      setImportanceBusy(false);
    }, 300);
    return () => clearTimeout(timer);
  }, [matches, kpsA, kpsB, imgA]);

  // значение деформируемости согласовано с текущими matches
  const impReady = !!(importance && importance.values.length === matches.length);

  // --- наложение: доступные «зелёные» пары (деформируемость ≤ GREEN_MAX_NORM)
  const greenCount = useMemo(
    () => greenPairCount(matches, impReady ? importance : null),
    [matches, impReady, importance],
  );
  const canOverlay = !!(imgA && imgB && greenCount >= 3);

  // --- расчёт warp (дебаунс 300мс), пока наложение видимо;
  // скрыли наложение — устаревший результат не храним.
  useEffect(() => {
    if (!overlayVisible || !imgA || !imgB) {
      setOverlay(null);
      return;
    }
    const gen = ++overlayGenRef.current;
    const pairs = selectGreenPairs(matches, kpsA, kpsB, impReady ? importance : null);
    if (!pairs) {
      setOverlay({ warped: null, green: 0, total: matches.length });
      return;
    }
    const timer = setTimeout(() => {
      if (overlayGenRef.current !== gen) return;
      const warped = warpRightToLeft(imgA, imgB, pairs.left, pairs.right);
      if (overlayGenRef.current !== gen) return;
      setOverlay({ warped, green: pairs.left.length, total: matches.length });
    }, 300);
    return () => clearTimeout(timer);
  }, [overlayVisible, imgA, imgB, kpsA, kpsB, matches, impReady, importance]);

  // --- отрисовка слоя наложения в левой панели
  const overlayWarped = overlay ? overlay.warped : null;
  useEffect(() => {
    const c = overlayLayerRef.current;
    if (!c) return;
    if (!overlayWarped) { c.width = 0; c.height = 0; return; }
    c.width = overlayWarped.width;
    c.height = overlayWarped.height;
    const ctx = c.getContext('2d');
    if (ctx) ctx.drawImage(overlayWarped, 0, 0);
  }, [overlayWarped]);

  const toggleOverlay = useCallback(() => setOverlayVisible(v => !v), []);

  // --- Mask2Former: смена правой картинки обесценивает результат
  useEffect(() => {
    maskGenRef.current++;                 // отменить запрос в полёте
    setMaskData(null);
    setMaskError(null);
    setMaskBusy(false);
    setFilterOn(false);                   // фильтруем по границам — без масок он бессмысленен
  }, [imgB]);

  const runMask = useCallback(async () => {
    if (!imgB) return;
    const gen = ++maskGenRef.current;
    setMaskBusy(true);
    setMaskError(null);
    try {
      const res = await fetchSegmentation(imgB);
      if (maskGenRef.current !== gen) return;
      setMaskData({ imgB, ...res });
      console.info(`Mask2Former: ${res.instances.length} областей за ${Math.round(res.elapsedMs)} мс`);
    } catch (e) {
      if (maskGenRef.current !== gen) return;
      console.error('Mask2Former:', e);
      setMaskError(String(e?.message || e));
    } finally {
      if (maskGenRef.current === gen) setMaskBusy(false);
    }
  }, [imgB]);

  const toggleMask = useCallback(() => {
    if (maskVisible) {
      setMaskVisible(false);
      setMaskError(null);                 // выключение сбрасывает ошибку — включение повторит запрос
      return;
    }
    setMaskVisible(true);
  }, [maskVisible]);

  // включили маски (или сменили картинку) — запустить расчёт, если результата нет
  useEffect(() => {
    if (maskVisible && imgB && (!maskData || !maskData.classMapPng) && !maskBusy && !maskError) runMask();
  }, [maskVisible, imgB, maskData, maskBusy, maskError, runMask]);

  // отрисовка слоя масок в правой панели (растягиваем на исходный кадр)
  const maskEl = maskData && maskData.imgB === imgB ? maskData.el : null;
  useEffect(() => {
    const c = maskLayerRef.current;
    if (!c) return;
    if (!maskEl || !imgB || !maskVisible) { c.width = 0; c.height = 0; return; }
    c.width = imgB.width;
    c.height = imgB.height;
    const ctx = c.getContext('2d');
    if (ctx) ctx.drawImage(maskEl, 0, 0, c.width, c.height);
  }, [maskEl, imgB, maskVisible]);

  const maskCount = maskData ? maskData.instances.length : 0;

  // --- фильтр точек: границы областей Mask2Former ∪ важные для деформации
  const maskBoundary = maskData && maskData.imgB === imgB ? maskData.boundary : null;
  const canFilter = !!maskBoundary;
  const classFilter = useClassFilter({
    enabled: filterOn, classMap: maskData?.imgB === imgB ? maskData.classMapPng : null,
    points: kpsB, excludedClasses, margin: classMargin,
  });

  const filterSets = useMemo(() => {
    if (!filterOn || !maskBoundary) return null;
    return computeHiddenSets({
      kpsA, kpsB, matches,
      importance: impReady ? importance : null,
      boundary: maskBoundary, imgB,
      impThr: filterImpThr, radius: filterRadius,
      classKeep: classFilter.keep, geometryOn: geometryFilterOn,
    });
  }, [filterOn, maskBoundary, kpsA, kpsB, matches,
      impReady, importance, imgB, filterImpThr, filterRadius, classFilter.keep, geometryFilterOn]);

  const toggleFilter = useCallback(() => {
    if (!canFilter) return;
    setFilterOn(v => !v);
  }, [canFilter]);

  const canSave = !!(overlay && overlay.warped && imgA);

  const handleSaveImage = useCallback((kind) => {
    if (!overlay || !overlay.warped || !imgA) {
      console.warn('Нет рассчитанного наложения — сначала «Наложить правое на левое»');
      return;
    }
    const base = (imgA.name || 'overlay').replace(/\.[^.]+$/, '') || 'overlay';
    if (kind === 'warped') {
      downloadCanvas(overlay.warped, `${base}_warped.png`);
      console.info(`Сохранено деформированное правое: ${overlay.green} зелёных пар`);
    } else {
      downloadCanvas(compositeOverlay(imgA, overlay.warped, overlayAlpha), `${base}_overlay.png`);
      console.info(`Сохранён композит: ${overlay.green} зелёных пар, α=${overlayAlpha}`);
    }
  }, [overlay, imgA, overlayAlpha]);

  // --- нормировка для цвета деформируемости точек правой картинки
  const bNormMap = useMemo(() => {
    const map = new Map();
    if (impReady) {
      matches.forEach((m, i) => map.set(m.bIdx, importance.norms[i] ?? 0));
    }
    return map;
  }, [impReady, importance, matches]);

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
    if (dragging) return true;                 // во время перетаскивания ничего не гасим
    if (!hovered) return true;
    if (hovered.side === side && hovered.idx === idx) return true;
    const m = findMatch(hovered.side, hovered.idx);
    if (!m) return false;
    const otherIdx = hovered.side === 'a' ? m.bIdx : m.aIdx;
    return side !== hovered.side && idx === otherIdx;
  }, [hovered, findMatch, dragging]);

  // --- подгонка вида под текущие размеры панелей
  const fitView = useCallback(() => {
    if (!imgA || !imgB) return;
    const pa = panelARef.current, pb = panelBRef.current;
    if (!pa || !pb) return;
    const s = Math.min(pa.clientWidth / imgA.width, pb.clientWidth / imgB.width) * 0.92;
    setTf({ s, tx: 10, ty: 10 });
  }, [imgA, imgB, setTf]);

  // --- авто-фит при загрузке
  useEffect(() => {
    if (!imgA || !imgB) return;
    fitView();
    setSelected(null);
  }, [imgA, imgB, fitView]);

  // --- рефит при открытии/закрытии сайдбаров: картинки не должны обрезаться
  useEffect(() => { fitView(); }, [pairsOpen, impOpen, fitView]);

  // --- координаты относительно .panels-main: линии рисуются только внутри картинок
  const toMain = useCallback((p) => {
    const pm = panelsMainRef.current;
    if (!pm) return p;
    const r = pm.getBoundingClientRect();
    return { x: p.x - r.left, y: p.y - r.top };
  }, []);

  // --- линия между выбранной точкой и её парой
  useEffect(() => {
    if (!selected) { setLine(null); return; }
    const m = findMatch(selected.side, selected.idx);
    if (!m) { setLine(null); return; }

    const update = () => {
      const pa = panelARef.current, pb = panelBRef.current;
      if (!pa || !pb) return;
      if (filterSets && (filterSets.hiddenA.has(m.aIdx) || filterSets.hiddenB.has(m.bIdx))) {
        setLine(null);                       // пара скрыта фильтром — линию не рисуем
        return;
      }
      const ka = kpsA[m.aIdx], kb = kpsB[m.bIdx];
      if (!ka || !kb) return;
      const p1 = toMain(screenPoint(pa, tf, ka));
      const p2 = toMain(screenPoint(pb, tf, kb));
      setLine({ x1: p1.x, y1: p1.y, x2: p2.x, y2: p2.y });
    };
    update();
    window.addEventListener('resize', update);
    return () => window.removeEventListener('resize', update);
  }, [selected, tf, matches, kpsA, kpsB, findMatch, toMain, pairsOpen, impOpen, filterSets]);

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
    setDragging({ side, idx });
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
      setDragging(null);
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

  // --- Delete/Backspace — удалить выбранную точку; Escape — отмена пары / закрыть панель
  useEffect(() => {
    const onKey = (e) => {
      if (e.key === 'Escape') {
        setConnectFrom(null);
        setPairsOpen(false);
        setImpOpen(false);
        return;
      }
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
        if (filterSets && (filterSets.hiddenA.has(m.aIdx) || filterSets.hiddenB.has(m.bIdx))) return;
        const ka = kpsA[m.aIdx], kb = kpsB[m.bIdx];
        if (!ka || !kb) return;
        const p1 = toMain(screenPoint(pa, tf, ka));
        const p2 = toMain(screenPoint(pb, tf, kb));
        out.push({ x1: p1.x, y1: p1.y, x2: p2.x, y2: p2.y, color: pairColor(i) });
      });
      setAllLines(out);
    };
    update();
    window.addEventListener('resize', update);
    return () => window.removeEventListener('resize', update);
  }, [showAllLines, tf, matches, kpsA, kpsB, toMain, pairsOpen, impOpen, filterSets]);

  // --- загрузка одной картинки в конкретную рамку
  const handleFile = useCallback(async (file, side) => {
    if (!file) return;
    let img;
    try { img = await loadImage(file); }
    catch (e) { console.error('Не удалось открыть изображение:', e); return; }

    // точки детекта относились к прежним картинкам — убрать; JSON остаётся
    if (pointsSource === 'det') {
      setKpsA([]); setKpsB([]); setMatches([]);
      setPointsSource(null);
      setSelected(null);
      setConnectFrom(null);
    }
    if (side === 'a') setImgA(img);
    else setImgB(img);
  }, [pointsSource]);

  // --- JSON: применить распарсенные точки
  const applyParsed = useCallback((p, prefix = '') => {
    let warn = '';
    if (imgA && p.imageA && imgA.name !== p.imageA) warn += ` · A: «${p.imageA}» ≠ «${imgA.name}»`;
    if (imgB && p.imageB && imgB.name !== p.imageB) warn += ` · B: «${p.imageB}» ≠ «${imgB.name}»`;
    if (warn) console.warn('Загруженный JSON:' + warn);

    setKpsA(p.kpsA);
    setKpsB(p.kpsB);
    setMatches(p.matches);
    setPointsSource('json');
    setSelected(null);
    setConnectFrom(null);
    lastJsonLoadAtRef.current = Date.now();
    console.info(`${prefix}${p.counts.pairs} пар · ${p.counts.ua}/${p.counts.ub} непарных`);
  }, [imgA, imgB]);

  const handleJsonFile = useCallback(async (file) => {
    try {
      const text = await file.text();
      applyParsed(parsePointsJson(text));
    } catch (e) {
      console.error('Ошибка JSON:', e);
    }
  }, [applyParsed]);

  const handleSaveJson = useCallback(() => {
    if (!kpsA.length && !kpsB.length) { console.warn('Нет точек для сохранения'); return; }
    const text = serializePointsJson({
      kpsA, kpsB, matches,
      imageA: imgA ? imgA.name : null,
      imageB: imgB ? imgB.name : null,
    });
    const base = (imgA ? imgA.name : 'keypoints').replace(/\.[^.]+$/, '') || 'keypoints';
    downloadJson(text, `${base}_points.json`);
    const ua = unmatchedIdx(kpsA, matches, 'a').length;
    const ub = unmatchedIdx(kpsB, matches, 'b').length;
    console.info(`Сохранено: ${matches.length} пар · ${ua}/${ub} непарных`);
  }, [kpsA, kpsB, matches, imgA, imgB]);

  const toggleConnect = useCallback(() => {
    setConnectMode(v => !v);
    setConnectFrom(null);
  }, []);

  const canGenerate = !!(cvReady && imgA && imgB);

  return (
    <div className="app">
      <Toolbar
        method={method}
        setMethod={setMethod}
        methodsMeta={methodsMeta}
        onReset={reset}
        busy={busy}
        showAllLines={showAllLines}
        onToggleAllLines={() => setShowAllLines(v => !v)}
        onJsonFile={handleJsonFile}
        onSaveJson={handleSaveJson}
        connectMode={connectMode}
        onToggleConnect={toggleConnect}
        onGenerate={handleGenerate}
        canGenerate={canGenerate}
        canOverlay={canOverlay}
        overlayVisible={overlayVisible}
        onToggleOverlay={toggleOverlay}
        overlayAlpha={overlayAlpha}
        setOverlayAlpha={setOverlayAlpha}
        greenCount={greenCount}
        pairCount={matches.length}
        onSaveImage={handleSaveImage}
        canSave={canSave}
        maskVisible={maskVisible}
        onToggleMask={toggleMask}
        maskBusy={maskBusy}
        maskError={maskError}
        canMask={!!imgB}
        maskAlpha={maskAlpha}
        setMaskAlpha={setMaskAlpha}
        maskCount={maskCount}
        maskMs={maskData ? maskData.elapsedMs : null}
        filterOn={filterOn}
        onToggleFilter={toggleFilter}
        canFilter={canFilter}
        filterInfo={filterSets}
        filterImpThr={filterImpThr}
        setFilterImpThr={setFilterImpThr}
        filterRadius={filterRadius}
        setFilterRadius={setFilterRadius}
        filterReady={impReady}
        semanticClasses={maskData?.classes || []}
        excludedClasses={excludedClasses}
        setExcludedClasses={setExcludedClasses}
        classMargin={classMargin}
        setClassMargin={setClassMargin}
        geometryFilterOn={geometryFilterOn}
        setGeometryFilterOn={setGeometryFilterOn}
        classFilterBusy={classFilter.busy}
        classFilterError={classFilter.error}
        marginSavedCount={classFilter.savedCount}
      />

      <div className="panels">
        <div className="panels-main" ref={panelsMainRef}>
          <Panel
            side="a" img={imgA} kps={kpsA} tf={tf} panelRef={panelARef}
            onWheel={makeWheelHandler(() => panelARef.current)}
            onMouseDown={onMouseDown} onPanelClick={handlePanelClick}
            onPanelDblClick={handlePanelDblClick}
            hovered={hovered} setHovered={setHovered}
            selected={selected}
            colorMap={matchColorIdx.a} isActive={isActive} onFile={handleFile}
            pairNo={pairNo.a}
            connectFrom={connectFrom}
            onPointClick={handlePointClick}
            onStartDrag={startPointDrag}
            dragging={dragging}
            overlayLayerRef={overlayLayerRef}
            overlayAlpha={overlayAlpha}
            hidden={filterSets ? filterSets.hiddenA : null}
          />
          <Panel
            side="b" img={imgB} kps={kpsB} tf={tf} panelRef={panelBRef}
            onWheel={makeWheelHandler(() => panelBRef.current)}
            onMouseDown={onMouseDown} onPanelClick={handlePanelClick}
            onPanelDblClick={handlePanelDblClick}
            hovered={hovered} setHovered={setHovered}
            selected={selected}
            colorMap={matchColorIdx.b} isActive={isActive} onFile={handleFile}
            pairNo={pairNo.b}
            normMap={bNormMap}
            connectFrom={connectFrom}
            onPointClick={handlePointClick}
            onStartDrag={startPointDrag}
            dragging={dragging}
            maskLayerRef={maskLayerRef}
            maskAlpha={maskAlpha}
            maskVisible={maskVisible}
            hidden={filterSets ? filterSets.hiddenB : null}
          />
          <ConnectionLine lines={allLines} line={line} />
        </div>

        <PointsPanel
          pairsOpen={pairsOpen} setPairsOpen={setPairsOpen}
          impOpen={impOpen} setImpOpen={setImpOpen}
          matches={matches} kpsA={kpsA} kpsB={kpsB}
          selected={selected} setSelected={setSelected}
          setHovered={setHovered}
          importance={impReady ? importance : null}
          importanceBusy={importanceBusy}
          hiddenA={filterSets ? filterSets.hiddenA : null}
          hiddenB={filterSets ? filterSets.hiddenB : null}
        />
      </div>
    </div>
  );
}
