import {
  HIT_RADIUS, POINT_RADIUS, POINT_RADIUS_HOVER, HALO_RADIUS,
  DIM_OPACITY, COLOR_UNMATCHED, COLOR_LINE, pairColor, importanceColor,
} from '../constants';

export function KeypointOverlay({
  side, kps, tf, colorMap, pairNo, normMap, hidden,
  hovered, setHovered, selected, isActive,
  connectFrom, onPointClick, onStartDrag, dragging,
}) {
  const stop = (e) => e.stopPropagation();

  const dragIdx = dragging && dragging.side === side ? dragging.idx : null;
  const isDragged = (i) => dragIdx === i;

  // перетаскиваемая точка рисуется последней — не уходит под соседей
  let order = kps.map((_, i) => i);
  if (dragIdx != null && dragIdx < order.length) {
    order = [...order.filter(i => i !== dragIdx), dragIdx];
  }

  // во время перетаскивания ховер соседних точек игнорируется
  const enter = (i) => () => {
    if (dragging) return;
    setHovered({ side, idx: i });
  };
  const leave = (i) => () => {
    if (dragging) return;
    setHovered(h => (h && h.side === side && h.idx === i) ? null : h);
  };

  return (
    <svg width="100%" height="100%"
         style={{ position: 'absolute', top: 0, left: 0, overflow: 'visible' }}>
      {order.map((i) => {
        const kp = kps[i];
        if (!kp) return null;
        if (hidden && hidden.has(i)) return null;   // точка отсеяна фильтром после масок
        const active = isActive(side, i);
        const excluded = !!kp.ex;               // вне триангуляции (режим деформации)
        const mIdx   = colorMap.get(i);
        // цвет по деформируемости — только правая картинка (левая — эталон)
        const norm   = normMap && mIdx != null ? normMap.get(i) : undefined;
        const color  = mIdx != null
          ? (norm !== undefined ? importanceColor(norm) : pairColor(mIdx))
          : COLOR_UNMATCHED;

        const isHovered  = !dragging && hovered && hovered.side === side && hovered.idx === i;
        const isSelected = selected && selected.side === side && selected.idx === i;
        const isPending  = connectFrom && connectFrom.side === side && connectFrom.idx === i;
        const highlight  = isHovered || isSelected || isPending || isDragged(i);

        const r = (highlight ? POINT_RADIUS_HOVER : POINT_RADIUS) / tf.s;
        const xs = (POINT_RADIUS + 3) / tf.s;    // половина стороны ×
        const no = pairNo ? pairNo.get(i) : undefined;

        return (
          <g key={i} opacity={excluded ? 0.45 : (active ? 1 : DIM_OPACITY)}>
            {highlight && (
              <circle cx={kp.x} cy={kp.y} r={HALO_RADIUS / tf.s}
                      fill="none"
                      stroke={isPending ? COLOR_LINE : color}
                      strokeWidth={(isPending ? 2 : 1.5) / tf.s} />
            )}
            <circle cx={kp.x} cy={kp.y} r={r}
                    fill={isPending ? COLOR_LINE : color}
                    stroke={isSelected || isDragged(i) ? '#fff' : 'rgba(0,0,0,.55)'}
                    strokeWidth={(isSelected || isDragged(i) ? 1.5 : 0.5) / tf.s} />

            {excluded && (
              <path d={`M${kp.x - xs} ${kp.y - xs}L${kp.x + xs} ${kp.y + xs}M${kp.x + xs} ${kp.y - xs}L${kp.x - xs} ${kp.y + xs}`}
                    stroke="#e05c5c" strokeWidth={1.6 / tf.s}
                    strokeLinecap="round"
                    style={{ pointerEvents: 'none' }} />
            )}

            {no != null && (
              <text x={kp.x + (POINT_RADIUS_HOVER + 4) / tf.s}
                    y={kp.y - (POINT_RADIUS_HOVER + 2) / tf.s}
                    fontSize={11 / tf.s}
                    fontFamily="ui-sans-serif, system-ui, sans-serif"
                    fontWeight={600}
                    fill={color}
                    stroke="rgba(0,0,0,.8)"
                    strokeWidth={2.5 / tf.s}
                    strokeLinejoin="round"
                    style={{ paintOrder: 'stroke', pointerEvents: 'none' }}
                    dominantBaseline="middle">
                {no}
              </text>
            )}

            {/* Прозрачный хитбокс — не зависит от зума */}
            <circle cx={kp.x} cy={kp.y} r={HIT_RADIUS / tf.s}
                    fill="transparent"
                    style={{ pointerEvents: 'all', cursor: 'pointer' }}
                    onMouseEnter={enter(i)}
                    onMouseLeave={leave(i)}
                    onMouseDown={(e) => {
                      e.stopPropagation();
                      onStartDrag(side, i, e);
                    }}
                    onClick={(e) => {
                      e.stopPropagation();
                      onPointClick(side, i);
                    }}
                    onDoubleClick={stop} />
          </g>
        );
      })}
    </svg>
  );
}
