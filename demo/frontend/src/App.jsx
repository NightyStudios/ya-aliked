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
import { fetchSimilarity } from './cv/similarity';
import { pairColor } from './constants';
import { clampKp, removePoint, setPair, unmatchedIdx } from './points';
import { parsePointsJson, serializePointsJson, downloadJson } from './jsonPoints';
import { computeHiddenSets } from './filterPoints';
import {
  computeImportanceRange, normalize, buildPairArrays,
  MIN_POINTS_FOR_IMPORTANCE,
} from './importance';
import {
  selectWarpPairs, warpPairCount, warpWithMesh,
  compositeOverlay, downloadCanvas,
} from './warp';
import { readSlot, writeSlot } from './snapshots';
import { removePair, restoreEntries, argminNorms } from './prune';

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
  // --- режим деформации: клик по точке — исключить её пару из триангуляции
  // (мягкое исключение, флаг ex на точке; пары и JSON не меняются)
  const [deformMode, setDeformMode] = useState(false);
  // --- режим редактирования: выбранная точка мельчает и тускнеет,
  // координаты в HUD, движение стрелками ±1px (Shift ±10px)
  const [editMode, setEditMode] = useState(false);
  // --- solo: визуально выключены все точки, кроме выбранной пары
  const [soloMode, setSoloMode] = useState(false);

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
  const [overlay, setOverlay] = useState(null);        // { warped, edges, green, total }
  const overlayLayerRef = useRef(null);
  const overlayGenRef = useRef(0);

  // --- то же наложение поверх правой (неизменённой) картинки + сетка триангуляции
  const [rwarpVisible, setRWarpVisible] = useState(false);
  const [rwarpAlpha, setRWarpAlpha] = useState(0.5);
  const rwarpLayerRef = useRef(null);
  const rwarpMeshRef = useRef(null);

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
  // снапшот параметров при нажатии «Применить»: пока совпадает — панель свёрнута
  const [appliedFilterParams, setAppliedFilterParams] = useState(null);

  // --- проверка с эталоном: метрики сходства начального и наложенного
  const [refImage, setRefImage] = useState(null);         // { name, dataUrl } — эталон на сессию
  const [checkBusy, setCheckBusy] = useState(false);
  const [checkError, setCheckError] = useState(null);
  const [checkData, setCheckData] = useState(null);   // { refName, ref_size, compared_size, initial, warped, ... }
  const [checkOpen, setCheckOpen] = useState(false);
  const checkGenRef = useRef(0);

  // --- prune: итеративное удаление пар с наименьшей деформируемостью
  const [pruneTarget, setPruneTarget] = useState(0);   // значение ползунка (сколько удалить)
  const [pruneStack, setPruneStack] = useState([]);    // записи удалённого (для возврата)
  const [pruneBusy, setPruneBusy] = useState(false);
  const [pruneProgress, setPruneProgress] = useState(null); // { i, n }
  const pruneGenRef = useRef(0);

  // --- снимки точек: initial (база) и actual (Ctrl+S / тихий сейв / автосейв)
  const [snapshotMeta, setSnapshotMeta] = useState({ initial: null, actual: null });
  const [autoSave, setAutoSave] = useState(() => {
    try { return globalThis.localStorage?.getItem('aliked.autosave') === '1'; }
    catch { return false; }
  });
  useEffect(() => {
    try { globalThis.localStorage?.setItem('aliked.autosave', autoSave ? '1' : '0'); }
    catch { /* ignore */ }
  }, [autoSave]);

  // канонический текст текущего состояния — база для dirty-сравнения и записи слотов
  const currentJsonText = useMemo(() => {
    if (!kpsA.length && !kpsB.length) return '';
    return serializePointsJson({
      kpsA, kpsB, matches,
      imageA: imgA ? imgA.name : null,
      imageB: imgB ? imgB.name : null,
    });
  }, [kpsA, kpsB, matches, imgA, imgB]);
  const snapshotDirty = !!(currentJsonText && snapshotMeta.actual?.text !== currentJsonText);

  const saveActual = useCallback((log = false) => {
    if (!currentJsonText) return false;
    const slot = {
      text: currentJsonText,
      savedAt: Date.now(),
      imageA: imgA ? imgA.name : null,
      imageB: imgB ? imgB.name : null,
    };
    if (!writeSlot(null, 'actual', slot)) return false;
    setSnapshotMeta(m => ({ ...m, actual: slot }));
    if (log) console.info('Снимок сохранён (актуальный слот)');
    return true;
  }, [currentJsonText, imgA, imgB]);

  // базовая линия: новый JSON или результат детекта → оба слота, dirty=false
  const setBaseSnapshots = useCallback((text, imageA, imageB) => {
    const slot = { text, savedAt: Date.now(), imageA, imageB };
    writeSlot(null, 'initial', slot);
    writeSlot(null, 'actual', slot);
    setSnapshotMeta({ initial: slot, actual: slot });
  }, []);

  // тихий сейв перед переключением режима/фильтра — только если есть правки
  const quietSave = useCallback(() => { if (snapshotDirty) saveActual(); },
    [snapshotDirty, saveActual]);

  // отмена prune-цикла и сброс стека при замене точек
  const resetPrune = useCallback(() => {
    pruneGenRef.current++;
    setPruneStack([]);
    setPruneTarget(0);
    setPruneBusy(false);
    setPruneProgress(null);
  }, []);

  // стейт-зеркало для prune-цикла: эффект зависит только от pruneTarget,
  // поэтому цикл читает свежие массивы из ref, а не из замыкания
  const stateRef = useRef({ kpsA, kpsB, matches, pruneStack });
  useEffect(() => { stateRef.current = { kpsA, kpsB, matches, pruneStack }; });

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
    setSoloMode(false);
    resetPrune();
    setBaseSnapshots(
      serializePointsJson({
        kpsA: det.kpsA, kpsB: det.kpsB, matches: det.matches,
        imageA: imgA ? imgA.name : null, imageB: imgB ? imgB.name : null,
      }),
      imgA ? imgA.name : null, imgB ? imgB.name : null,
    );
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

  // --- фильтр точек: границы областей Mask2Former ∪ важные для деформации.
  // Объявлен до warpCount/эффекта наложения: warp строится по показываемым.
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

  // панель настроек фильтра видна, пока параметры отличаются от применённых
  const filterDirty = !appliedFilterParams
    || filterImpThr !== appliedFilterParams.filterImpThr
    || filterRadius !== appliedFilterParams.filterRadius
    || classMargin !== appliedFilterParams.classMargin
    || geometryFilterOn !== appliedFilterParams.geometryFilterOn
    || pruneTarget !== appliedFilterParams.pruneTarget
    || excludedClasses.length !== appliedFilterParams.excludedClasses.length
    || excludedClasses.some((id, i) => id !== appliedFilterParams.excludedClasses[i]);

  const onApplyFilter = useCallback(() => {
    setAppliedFilterParams({
      filterImpThr, filterRadius, classMargin, geometryFilterOn,
      pruneTarget, excludedClasses: [...excludedClasses],
    });
  }, [filterImpThr, filterRadius, classMargin, geometryFilterOn,
      pruneTarget, excludedClasses]);

  const onExpandFilter = useCallback(() => setAppliedFilterParams(null), []);

  const toggleFilter = useCallback(() => {
    if (!canFilter) return;
    quietSave();
    setFilterOn(v => !v);
    // включение всегда раскрывает панель (выключение снапшот не трогает —
    // ctxbar и так скрыт, а при следующем включении панель будет видна)
    setAppliedFilterParams(null);
  }, [canFilter, quietSave]);

  // --- prune: цикл удаления/восстановления пар по значению ползунка.
  // Зависит только от pruneTarget: свежие массивы читает из stateRef,
  // поэтому промежуточные коммиты не перезапускают цикл.
  const pruneMax = Math.max(0, matches.length + pruneStack.length - MIN_POINTS_FOR_IMPORTANCE);
  useEffect(() => {
    const { kpsA, kpsB, matches, pruneStack } = stateRef.current;
    const gen = ++pruneGenRef.current;

    // возврат: снять верхние записи стека до целевого размера (LIFO)
    if (pruneTarget < pruneStack.length) {
      const n = pruneStack.length - pruneTarget;
      const r = restoreEntries(kpsA, kpsB, matches, pruneStack.slice(-n));
      if (r.failed) {
        console.warn(`Восстановление prune: ${r.failed} записей возвращены в конец ` +
          '(внешняя правка сдвинула индексы)');
      }
      setKpsA(r.kpsA); setKpsB(r.kpsB); setMatches(r.matches);
      setPruneStack(s => s.slice(0, s.length - n));
      return;
    }
    if (pruneTarget === pruneStack.length) return;

    // удаление: после каждой пары — полный пересчёт важности
    const n = pruneTarget - pruneStack.length;
    let curA = kpsA, curB = kpsB, curM = matches;
    let stack = pruneStack;
    setPruneBusy(true);
    setPruneProgress({ i: 0, n });
    (async () => {
      try {
        for (let i = 0; i < n; i++) {
          if (pruneGenRef.current !== gen) return;
          if (curM.length < MIN_POINTS_FOR_IMPORTANCE) {
            console.warn('prune: пар осталось меньше минимума — остановились');
            break;
          }
          const pairs = buildPairArrays(curM, curA, curB, imgA);
          if (!pairs) break;
          const { left, right, corners } = pairs;
          const M = left.length;
          const values = new Float64Array(M);
          const CH = 20;
          for (let s = 0; s < M; s += CH) {
            if (pruneGenRef.current !== gen) return;
            computeImportanceRange(left, right, corners, s, Math.min(M, s + CH), values);
            await new Promise(r => setTimeout(r, 0));
          }
          if (pruneGenRef.current !== gen) return;
          const idx = argminNorms(normalize(values));
          if (idx == null) break;
          const res = removePair(curA, curB, curM, idx);
          if (!res) break;
          curA = res.kpsA; curB = res.kpsB; curM = res.matches;
          stack = [...stack, res.entry];
          // коммит после каждого удаления — UI живой, debounce-эффект важности
          // схлопнется после финального коммита (300 мс > шага цикла не будут
          // гоняться — генератор impGenRef их и так отменяет)
          setKpsA(curA); setKpsB(curB); setMatches(curM); setPruneStack(stack);
          setPruneProgress({ i: i + 1, n });
        }
        // ползунок мог цеплять больше, чем разрешили пары — вернуть согласованность
        if (pruneGenRef.current === gen && stack.length !== pruneTarget) {
          setPruneTarget(stack.length);
        }
      } finally {
        if (pruneGenRef.current === gen) {
          setPruneBusy(false);
          setPruneProgress(null);
        }
      }
    })();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [pruneTarget]);

  // --- наложение: пары в триангуляции — все, кроме исключённых кликом (ex)
  // и скрытых визуальным фильтром (классы/важность). Solo в warp не входит:
  // это режим просмотра, из 1 пары меш не построить.
  const warpHiddenA = filterSets ? filterSets.hiddenA : null;
  const warpHiddenB = filterSets ? filterSets.hiddenB : null;
  const warpCount = useMemo(
    () => warpPairCount(matches, kpsA, kpsB, warpHiddenA, warpHiddenB),
    [matches, kpsA, kpsB, warpHiddenA, warpHiddenB],
  );
  const canOverlay = !!(imgA && imgB && warpCount >= 3);

  // --- расчёт warp (дебаунс 300мс), пока любое из наложений видимо;
  // скрыли оба — устаревший результат не храним.
  // Пары считаются синхронно (без ожидания деформируемости): во время
  // дебаунса висит прежний warp, поэтому окна гашения при правке пар
  // или исключении точек нет.
  useEffect(() => {
    if ((!overlayVisible && !rwarpVisible) || !imgA || !imgB) {
      setOverlay(null);
      return;
    }
    const gen = ++overlayGenRef.current;
    const pairs = selectWarpPairs(matches, kpsA, kpsB, warpHiddenA, warpHiddenB);
    if (!pairs) {
      setOverlay({ warped: null, edges: [], pairs: 0, total: matches.length });
      return;
    }
    const timer = setTimeout(() => {
      if (overlayGenRef.current !== gen) return;
      const res = warpWithMesh(imgA, imgB, pairs.left, pairs.right);
      if (overlayGenRef.current !== gen) return;
      setOverlay({
        warped: res ? res.canvas : null,
        edges: res ? res.edges : [],
        pairs: pairs.left.length,
        total: matches.length,
      });
    }, 300);
    return () => clearTimeout(timer);
  }, [overlayVisible, rwarpVisible, imgA, imgB, kpsA, kpsB, matches,
      warpHiddenA, warpHiddenB]);

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

  // --- отрисовка того же warp поверх правой (неизменённой) картинки + сетка
  const rwarpOn = !!(rwarpVisible && overlay && overlay.warped);
  useEffect(() => {
    const c = rwarpLayerRef.current;
    const m = rwarpMeshRef.current;
    if (!c || !m) return;
    const w = (rwarpOn && overlay) ? overlay.warped : null;
    if (!w) {
      c.width = 0; c.height = 0;
      m.width = 0; m.height = 0;
      return;
    }
    c.width = w.width; c.height = w.height;
    m.width = w.width; m.height = w.height;
    const ctx = c.getContext('2d');
    if (ctx) ctx.drawImage(w, 0, 0);
    const g = m.getContext('2d');
    const edges = overlay.edges;
    if (g && edges && edges.length) {
      g.strokeStyle = 'rgba(255,255,255,0.9)';
      g.lineWidth = 1;
      g.beginPath();
      for (const [x1, y1, x2, y2] of edges) { g.moveTo(x1, y1); g.lineTo(x2, y2); }
      g.stroke();
    }
  }, [rwarpOn, overlay]);

  const toggleOverlay = useCallback(() => setOverlayVisible(v => !v), []);
  const toggleRWarp = useCallback(() => setRWarpVisible(v => !v), []);

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

  // --- solo: скрыть все точки, кроме выбранной пары (или самой точки без пары)
  const soloHidden = useMemo(() => {
    if (!soloMode) return null;
    const visA = new Set(), visB = new Set();
    if (selected) {
      const m = matches.find(mm =>
        (selected.side === 'a' ? mm.aIdx : mm.bIdx) === selected.idx);
      if (m) { visA.add(m.aIdx); visB.add(m.bIdx); }
      else if (selected.side === 'a') visA.add(selected.idx);
      else visB.add(selected.idx);
    }
    const hA = new Set(), hB = new Set();
    kpsA.forEach((_, i) => { if (!visA.has(i)) hA.add(i); });
    kpsB.forEach((_, i) => { if (!visB.has(i)) hB.add(i); });
    return { hiddenA: hA, hiddenB: hB };
  }, [soloMode, selected, matches, kpsA, kpsB]);

  const unionHidden = (a, b) => {
    if (!a && !b) return null;
    if (!a) return b;
    if (!b) return a;
    const out = new Set(a);
    b.forEach(i => out.add(i));
    return out;
  };
  const effectiveHiddenA = unionHidden(filterSets ? filterSets.hiddenA : null,
    soloHidden ? soloHidden.hiddenA : null);
  const effectiveHiddenB = unionHidden(filterSets ? filterSets.hiddenB : null,
    soloHidden ? soloHidden.hiddenB : null);

  // HUD координат — только в режиме правки, в панели выбранной точки
  const editHudA = editMode && selected && selected.side === 'a' && kpsA[selected.idx]
    ? kpsA[selected.idx] : null;
  const editHudB = editMode && selected && selected.side === 'b' && kpsB[selected.idx]
    ? kpsB[selected.idx] : null;

  // --- проверка с эталоном: кнопка активна всегда, эталон запоминается
  // на сессию; расчёт запускает эффект ниже, как только есть наложение.

  // выбор файла лишь запоминает эталон — расчёт запускает эффект ниже
  const handleCheckFile = useCallback(async (file) => {
    if (!file) return;
    let refImg;
    try {
      refImg = await loadImage(file);
    } catch (e) {
      console.error('Не удалось открыть эталон:', e);
      setCheckError(String(e?.message || e));
      return;
    }
    setRefImage(refImg);
    setCheckError(null);
    setCheckOpen(true);
  }, []);

  // авто-пересчёт метрик при любом изменении наложения (правка пар, prune, фильтр)
  const warpedForCheck = overlay ? overlay.warped : null;
  useEffect(() => {
    if (!refImage || !imgB || !warpedForCheck) return;
    const gen = ++checkGenRef.current;
    setCheckBusy(true);
    setCheckError(null);
    // debounce поверх warp-debounce — prune-цикл коммитит пачками
    const timer = setTimeout(async () => {
      try {
        const warpedUrl = warpedForCheck.toDataURL('image/png');
        const res = await fetchSimilarity(refImage.dataUrl, imgB.dataUrl, warpedUrl);
        if (checkGenRef.current !== gen) return;
        setCheckData({ refName: refImage.name || 'эталон', ...res });
      } catch (e) {
        if (checkGenRef.current !== gen) return;
        console.error('Проверка с эталоном:', e);
        setCheckError(String(e?.message || e));
      } finally {
        if (checkGenRef.current === gen) setCheckBusy(false);
      }
    }, 400);
    return () => clearTimeout(timer);
  }, [refImage, imgB, warpedForCheck]);

  // смена правого изображения — новый сценарий: результат проверки бессмысленен,
  // но эталон сессионный и переживает смену снимка (метрики пересчитает авто-эффект)
  useEffect(() => {
    checkGenRef.current++;
    setCheckData(null);
    setCheckError(null);
    setCheckBusy(false);
    setCheckOpen(false);
  }, [imgB]);

  const canSave = !!(overlay && overlay.warped && imgA);

  const handleSaveImage = useCallback((kind) => {
    if (!overlay || !overlay.warped || !imgA) {
      console.warn('Нет рассчитанного наложения — сначала «Наложить правое на левое»');
      return;
    }
    const base = (imgA.name || 'overlay').replace(/\.[^.]+$/, '') || 'overlay';
    if (kind === 'warped') {
      downloadCanvas(overlay.warped, `${base}_warped.png`);
      console.info(`Сохранено деформированное правое: ${overlay.pairs} пар в триангуляции`);
    } else {
      downloadCanvas(compositeOverlay(imgA, overlay.warped, overlayAlpha), `${base}_overlay.png`);
      console.info(`Сохранён композит: ${overlay.pairs} пар в триангуляции, α=${overlayAlpha}`);
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
  useEffect(() => { fitView(); }, [pairsOpen, impOpen, checkOpen, fitView]);

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
  }, [selected, tf, matches, kpsA, kpsB, findMatch, toMain, pairsOpen, impOpen, checkOpen, filterSets]);

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

  // --- клик по точке: режим деформации (исключить/вернуть), «Соединить» или выбор
  const handlePointClick = useCallback((side, idx) => {
    if (consumePointDrag()) return;

    if (deformMode) {
      // мягкое исключение: только флаг ex на точке — пары и JSON не меняются
      const flip = (arr) => arr.map((k, i) => (i === idx ? { ...k, ex: !k.ex } : k));
      if (side === 'a') setKpsA(flip); else setKpsB(flip);
      return;
    }

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
  }, [deformMode, connectMode, connectFrom]);

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

  // --- клавиатура: Delete — удалить выбранную точку; Escape — выйти из режимов;
  // Ctrl/Cmd+S — сохранить актуальный снимок; стрелки — правка выбранной точки
  useEffect(() => {
    const onKey = (e) => {
      const typing = e.target && (
        e.target.tagName === 'INPUT' || e.target.tagName === 'TEXTAREA'
        || e.target.isContentEditable);

      if ((e.ctrlKey || e.metaKey) && (e.key === 's' || e.key === 'S' || e.key === 'ы')) {
        e.preventDefault();                        // не открывать «Сохранить страницу»
        saveActual(true);
        return;
      }
      if (e.key === 'Escape') {
        setConnectFrom(null);
        setDeformMode(false);
        setEditMode(false);
        setSoloMode(false);
        setPairsOpen(false);
        setImpOpen(false);
        setCheckOpen(false);
        return;
      }

      // стрелки: движение выбранной точки ±1px (Shift ±10px) в режиме правки
      if (editMode && selected && !typing
          && (e.key === 'ArrowUp' || e.key === 'ArrowDown'
              || e.key === 'ArrowLeft' || e.key === 'ArrowRight')) {
        e.preventDefault();
        const step = e.shiftKey ? 10 : 1;
        const dx = e.key === 'ArrowRight' ? step : e.key === 'ArrowLeft' ? -step : 0;
        const dy = e.key === 'ArrowDown' ? step : e.key === 'ArrowUp' ? -step : 0;
        const { side, idx } = selected;
        const kps = side === 'a' ? kpsA : kpsB;
        const img = side === 'a' ? imgA : imgB;
        const kp = kps[idx];
        if (!kp || !img) return;
        const np = clampKp({ x: kp.x + dx, y: kp.y + dy }, img);
        if (side === 'a') setKpsA(prev => prev.map((k, i) => (i === idx ? np : k)));
        else setKpsB(prev => prev.map((k, i) => (i === idx ? np : k)));
        return;
      }

      if (e.key !== 'Delete' && e.key !== 'Backspace') return;
      if (typing) return;                          // Backspace в поле ввода не трогает точки
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
  }, [selected, kpsA, kpsB, matches, editMode, imgA, imgB, saveActual]);

  // --- все линии матчей (тумблер в тулбаре)
  useEffect(() => {
    if (!showAllLines) { setAllLines([]); return; }

    const update = () => {
      const pa = panelARef.current, pb = panelBRef.current;
      if (!pa || !pb) return;
      const out = [];
      matches.forEach((m, i) => {
        if (filterSets && (filterSets.hiddenA.has(m.aIdx) || filterSets.hiddenB.has(m.bIdx))) return;
        if (soloHidden && (soloHidden.hiddenA.has(m.aIdx) || soloHidden.hiddenB.has(m.bIdx))) return;
        const ka = kpsA[m.aIdx], kb = kpsB[m.bIdx];
        if (!ka || !kb) return;
        const p1 = toMain(screenPoint(pa, tf, ka));
        const p2 = toMain(screenPoint(pb, tf, kb));
        out.push({
          x1: p1.x, y1: p1.y, x2: p2.x, y2: p2.y,
          color: pairColor(i),
          dim: !!(ka.ex || kb.ex),           // пара вне триангуляции
        });
      });
      setAllLines(out);
    };
    update();
    window.addEventListener('resize', update);
    return () => window.removeEventListener('resize', update);
  }, [showAllLines, tf, matches, kpsA, kpsB, toMain, pairsOpen, impOpen, checkOpen, filterSets, soloHidden]);

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
  const applyParsed = useCallback((p, prefix = '', isBaseline = false) => {
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
    setSoloMode(false);
    resetPrune();
    lastJsonLoadAtRef.current = Date.now();
    if (isBaseline) {
      setBaseSnapshots(
        serializePointsJson({
          kpsA: p.kpsA, kpsB: p.kpsB, matches: p.matches,
          imageA: p.imageA, imageB: p.imageB,
        }),
        p.imageA, p.imageB,
      );
    }
    console.info(`${prefix}${p.counts.pairs} пар · ${p.counts.ua}/${p.counts.ub} непарных`);
  }, [imgA, imgB, resetPrune, setBaseSnapshots]);

  const handleJsonFile = useCallback(async (file) => {
    try {
      const text = await file.text();
      applyParsed(parsePointsJson(text), '', true);
    } catch (e) {
      console.error('Ошибка JSON:', e);
    }
  }, [applyParsed]);

  // восстановление из слота (не трогает слоты — иначе lost baseline/actual)
  const restoreSnapshot = useCallback((slot) => {
    const s = readSlot(null, slot);
    if (!s) { console.warn(`Слот «${slot}» пуст`); return; }
    try {
      applyParsed(parsePointsJson(s.text),
        `Восстановлено из слота ${slot} (${new Date(s.savedAt).toLocaleTimeString()}): `);
    } catch (e) {
      console.error('Ошибка восстановления снимка:', e);
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

  // режимы взаимоисключимы: включили один — остальные выключаются
  const toggleConnect = useCallback(() => {
    quietSave();
    const on = !connectMode;
    setConnectMode(on);
    setConnectFrom(null);
    if (on) { setDeformMode(false); setEditMode(false); }
  }, [connectMode, quietSave]);

  const toggleDeform = useCallback(() => {
    quietSave();
    const on = !deformMode;
    setDeformMode(on);
    if (on) { setConnectMode(false); setConnectFrom(null); setEditMode(false); }
  }, [deformMode, quietSave]);

  const toggleEdit = useCallback(() => {
    quietSave();
    const on = !editMode;
    setEditMode(on);
    if (on) { setConnectMode(false); setDeformMode(false); setConnectFrom(null); }
  }, [editMode, quietSave]);

  const toggleSolo = useCallback(() => setSoloMode(v => !v), []);

  // автосейв: включён — записывать актуальный снимок после каждой правки (debounce)
  useEffect(() => {
    if (!autoSave) return;
    const t = setTimeout(() => { saveActual(); }, 500);
    return () => clearTimeout(t);
  }, [autoSave, currentJsonText, saveActual]);

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
        deformMode={deformMode}
        onToggleDeform={toggleDeform}
        editMode={editMode}
        onToggleEdit={toggleEdit}
        soloMode={soloMode}
        onToggleSolo={toggleSolo}
        onGenerate={handleGenerate}
        canGenerate={canGenerate}
        canOverlay={canOverlay}
        overlayVisible={overlayVisible}
        onToggleOverlay={toggleOverlay}
        overlayAlpha={overlayAlpha}
        setOverlayAlpha={setOverlayAlpha}
        canRWarp={canOverlay}
        rwarpVisible={rwarpVisible}
        onToggleRWarp={toggleRWarp}
        rwarpAlpha={rwarpAlpha}
        setRWarpAlpha={setRWarpAlpha}
        warpCount={warpCount}
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
        filterDirty={filterDirty}
        onApplyFilter={onApplyFilter}
        onExpandFilter={onExpandFilter}
        filterImpThr={filterImpThr}
        setFilterImpThr={setFilterImpThr}
        filterRadius={filterRadius}
        setFilterRadius={setFilterRadius}
        filterReady={impReady}
        pruneTarget={pruneTarget}
        setPruneTarget={setPruneTarget}
        pruneMax={pruneMax}
        pruneBusy={pruneBusy}
        pruneProgress={pruneProgress}
        pruneRemoved={pruneStack.length}
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
        onCheckFile={handleCheckFile}
        checkBusy={checkBusy}
        checkError={checkError}
        refName={refImage?.name || null}
        snapshotDirty={snapshotDirty}
        snapshotInitialAt={snapshotMeta.initial?.savedAt ?? null}
        snapshotActualAt={snapshotMeta.actual?.savedAt ?? null}
        onSaveSnapshot={() => saveActual(true)}
        onRestoreSnapshot={restoreSnapshot}
        autoSave={autoSave}
        setAutoSave={setAutoSave}
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
            hidden={effectiveHiddenA}
            editMode={editMode}
            editHud={editHudA}
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
            rwarpLayerRef={rwarpLayerRef}
            rwarpMeshRef={rwarpMeshRef}
            rwarpAlpha={rwarpAlpha}
            rwarpVisible={rwarpVisible}
            hidden={effectiveHiddenB}
            editMode={editMode}
            editHud={editHudB}
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
          checkOpen={checkOpen} setCheckOpen={setCheckOpen}
          checkData={checkData} checkBusy={checkBusy} checkError={checkError}
          refReady={!!refImage} refName={refImage?.name || null}
        />
      </div>
    </div>
  );
}
