import { useEffect, useState } from 'react';

// Class/margin changes reuse the class map and never rerun Mask2Former.
export function useClassFilter({ enabled, classMap, points, excludedClasses, margin }) {
  const [state, setState] = useState({ result: null, busy: false, error: null });
  useEffect(() => {
    if (!enabled || !classMap || !excludedClasses.length || !points.length) {
      setState({ result: null, busy: false, error: null });
      return;
    }
    const controller = new AbortController();
    setState({ result: null, busy: true, error: null });
    const timer = setTimeout(async () => {
      try {
        const r = await fetch('/api/filter-keypoints', {
          method: 'POST', headers: { 'Content-Type': 'application/json' },
          signal: controller.signal,
          body: JSON.stringify({ class_map_png_b64: classMap,
            points: points.map(p => [p.x, p.y]), excluded_classes: excludedClasses, margin }),
        });
        if (!r.ok) throw new Error(`Фильтр классов: ${r.status} ${await r.text()}`);
        const data = await r.json();
        if (!controller.signal.aborted) setState({ busy: false, error: null,
          result: { ...data, points, classMap, excludedClasses, margin } });
      } catch (e) {
        if (!controller.signal.aborted) setState({ result: null, busy: false, error: e.message });
      }
    }, 180);
    return () => { clearTimeout(timer); controller.abort(); };
  }, [enabled, classMap, points, excludedClasses, margin]);
  const r = state.result;
  const current = r && r.points === points && r.classMap === classMap
    && r.excludedClasses === excludedClasses && r.margin === margin;
  return { keep: enabled && current ? r.keep : null,
    savedCount: enabled && current ? r.saved_by_margin.filter(Boolean).length : 0,
    busy: enabled && state.busy, error: enabled ? state.error : null };
}
