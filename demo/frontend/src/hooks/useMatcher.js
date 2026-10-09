import { useEffect, useRef, useState } from 'react';
import { DETECTORS } from '../cv/detectors';
import { matchDescriptors } from '../cv/matchers';

export function useMatcher({ cvReady, imgA, imgB, method, params = {} }) {
  const [state, setState] = useState({
    kpsA: [], kpsB: [], matches: [], status: '', busy: false,
    runId: 0, startedAt: 0,
  });

  const detRef = useRef({ a: null, b: null });
  const runIdRef = useRef(0);

  useEffect(() => {
    if (!imgA || !imgB) return;
    if (!cvReady) return;

    const det = DETECTORS[method];
    if (!det) {
      setState({
        kpsA: [], kpsB: [], matches: [], busy: false,
        runId: ++runIdRef.current, startedAt: Date.now(),
        status: `Неизвестный метод: ${method}`,
      });
      return;
    }

    let cancelled = false;
    const startedAt = Date.now();
    const runId = ++runIdRef.current;

    (async () => {
      setState(s => ({ ...s, busy: true, status: 'Detecting…' }));
      try {
        const [a, b] = await Promise.all([
          det.detect(imgA, params),
          det.detect(imgB, params),
        ]);
        if (cancelled) return;

        setState(s => ({ ...s, status: 'Matching…' }));

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
          status: `${a.points.length} / ${b.points.length} kp · ${ms.length} matches`,
        });
      } catch (e) {
        console.error(e);
        if (!cancelled) {
          setState({
            kpsA: [], kpsB: [], matches: [],
            busy: false,
            runId, startedAt,
            status: 'Ошибка: ' + e.message,
          });
        }
      }
    })();

    return () => { cancelled = true; };
  }, [cvReady, imgA, imgB, method, params]);

  // cleanup при размонтировании
  useEffect(() => () => {
    const d = detRef.current;
    if (d?.a?.desc?.delete) d.a.desc.delete();
    if (d?.b?.desc?.delete) d.b.desc.delete();
  }, []);

  return { ...state, detections: detRef.current };
}
