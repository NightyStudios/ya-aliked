import {
  HIT_RADIUS, POINT_RADIUS, POINT_RADIUS_HOVER, HALO_RADIUS,
  DIM_OPACITY, COLOR_UNMATCHED, COLOR_LINE, pairColor,
} from '../constants';

export function KeypointOverlay({
  side, kps, tf, colorMap, pairNo,
  hovered, setHovered, selected, isActive,
  connectFrom, onPointClick, onStartDrag,
}) {
  const stop = (e) => e.stopPropagation();

  return (
    <svg width="100%" height="100%"
         style={{ position: 'absolute', top: 0, left: 0, overflow: 'visible' }}>
      {kps.map((kp, i) => {
        const active = isActive(side, i);
        const mIdx   = colorMap.get(i);
        const color  = mIdx != null ? pairColor(mIdx) : COLOR_UNMATCHED;

        const isHovered  = hovered  && hovered.side  === side && hovered.idx  === i;
        const isSelected = selected && selected.side === side && selected.idx === i;
        const isPending  = connectFrom && connectFrom.side === side && connectFrom.idx === i;

        const r = (isHovered || isSelected || isPending
          ? POINT_RADIUS_HOVER : POINT_RADIUS) / tf.s;
        const no = pairNo ? pairNo.get(i) : undefined;

        return (
          <g key={i} opacity={active ? 1 : DIM_OPACITY}>
            {(isHovered || isSelected || isPending) && (
              <circle cx={kp.x} cy={kp.y} r={HALO_RADIUS / tf.s}
                      fill="none"
                      stroke={isPending ? COLOR_LINE : color}
                      strokeWidth={(isPending ? 2 : 1.5) / tf.s} />
            )}
            <circle cx={kp.x} cy={kp.y} r={r}
                    fill={isPending ? COLOR_LINE : color}
                    stroke={isSelected ? '#fff' : 'rgba(0,0,0,.55)'}
                    strokeWidth={(isSelected ? 1.5 : 0.5) / tf.s} />

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
                    onMouseEnter={() => setHovered({ side, idx: i })}
                    onMouseLeave={() =>
                      setHovered(h => (h && h.side === side && h.idx === i) ? null : h)}
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
