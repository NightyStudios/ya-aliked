import { useEffect, useRef, useState } from 'react';
import { DETECTORS } from '../cv/detectors';
import { matchDescriptors } from '../cv/matchers';

// Детект выполняется ТОЛЬКО по изменению trigger (кнопка «Сгенерировать точки»).
// trigger === 0 — ни один запуск ещё не делался.
export function useMatcher({ cvReady, imgA, imgB, method, params = {}, trigger = 0 }) {
  const [state, setState] = useState({
    kpsA: [], kpsB: [], matches: [], busy: false,
    runId: 0, startedAt: 0, runImgA: null, runImgB: null,
  });

  const detRef = useRef({ a: null, b: null });
  const runIdRef = useRef(0);

  useEffect(() => {
    if (!trigger) return;

    if (!imgA || !imgB || !cvReady) {
      setState(s => ({ ...s, busy: false }));
      return;
    }

    const det = DETECTORS[method];
    if (!det) {
      console.error(`Неизвестный метод: ${method}`);
      setState(s => ({
        ...s, busy: false,
        runId: ++runIdRef.current, startedAt: Date.now(),
        runImgA: imgA, runImgB: imgB,
      }));
      return;
    }

    let cancelled = false;
    const startedAt = Date.now();
    const runId = ++runIdRef.current;

    (async () => {
      setState(s => ({ ...s, busy: true }));
      try {
        const [a, b] = await Promise.all([
          det.detect(imgA, params),
          det.detect(imgB, params),
        ]);
        if (cancelled) return;

        const ms = await matchDescriptors(a, b, { crossCheck: true });
        if (cancelled) return;

        const prev = detRef.current;
        if (prev?.a?.desc?.delete) prev.a.desc.delete();
        if (prev?.b?.desc?.delete) prev.b.desc.delete();
        detRef.current = { a, b };

        setState({
          kpsA: a.points,
          kpsB: b.points,
          matches: ms,
          busy: false,
          runId, startedAt,
          runImgA: imgA, runImgB: imgB,
        });
      } catch (e) {
        console.error(e);
        if (!cancelled) {
          setState({
            kpsA: [], kpsB: [], matches: [],
            busy: false,
            runId, startedAt,
            runImgA: imgA, runImgB: imgB,
          });
        }
      }
    })();

    return () => { cancelled = true; };
    // imgA/imgB сознательно не в deps: смена картинок не должна сама запускать детект;
    // актуальность результата проверяется по runImgA/runImgB в App.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [trigger]);

  // cleanup при размонтировании
  useEffect(() => () => {
    const d = detRef.current;
    if (d?.a?.desc?.delete) d.a.desc.delete();
    if (d?.b?.desc?.delete) d.b.desc.delete();
  }, []);

  return { ...state, detections: detRef.current };
}
