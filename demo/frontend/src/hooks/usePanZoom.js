import { useCallback, useEffect, useRef, useState } from 'react';

const MIN_SCALE = 0.05;
const MAX_SCALE = 30;

export function usePanZoom() {
  const [tf, setTf] = useState({ s: 1, tx: 0, ty: 0 });
  const dragRef    = useRef(null);
  const dragMoved  = useRef(false);

  useEffect(() => {
    const onMove = (e) => {
      if (!dragRef.current) return;
      const dx = e.clientX - dragRef.current.x;
      const dy = e.clientY - dragRef.current.y;
      if (Math.abs(dx) > 3 || Math.abs(dy) > 3) dragMoved.current = true;
      setTf(prev => ({ ...prev, tx: dragRef.current.tx + dx, ty: dragRef.current.ty + dy }));
    };
    const onUp = () => { dragRef.current = null; };
    window.addEventListener('mousemove', onMove);
    window.addEventListener('mouseup', onUp);
    return () => {
      window.removeEventListener('mousemove', onMove);
      window.removeEventListener('mouseup', onUp);
    };
  }, []);

  const onMouseDown = useCallback((e) => {
    if (e.button !== 0) return;
    dragMoved.current = false;
    dragRef.current = { x: e.clientX, y: e.clientY, tx: tf.tx, ty: tf.ty };
  }, [tf.tx, tf.ty]);

  const makeWheelHandler = useCallback((getPanelEl) => (e) => {
    e.preventDefault();
    const panel = getPanelEl();
    if (!panel) return;
    const rect = panel.getBoundingClientRect();
    const px = e.clientX - rect.left;
    const py = e.clientY - rect.top;
    const factor = Math.exp(-e.deltaY * 0.0015);
    setTf(prev => {
      const s2 = Math.max(MIN_SCALE, Math.min(MAX_SCALE, prev.s * factor));
      const k  = s2 / prev.s;
      return { s: s2, tx: px - (px - prev.tx) * k, ty: py - (py - prev.ty) * k };
    });
  }, []);

  const wasDrag = () => dragMoved.current;
  const reset   = useCallback(() => setTf({ s: 1, tx: 0, ty: 0 }), []);

  return { tf, setTf, onMouseDown, makeWheelHandler, wasDrag, reset };
}