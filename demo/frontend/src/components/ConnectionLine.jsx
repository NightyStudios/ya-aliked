import { COLOR_LINE } from '../constants';

export function ConnectionLine({ lines = [], line }) {
  const any = lines.length || line;
  if (!any) return null;
  return (
    <svg className="lines">
      {lines.map((l, i) => (
        <g key={i} opacity="0.55">
          <line x1={l.x1} y1={l.y1} x2={l.x2} y2={l.y2}
                stroke={l.color || COLOR_LINE} strokeWidth="1.5" />
        </g>
      ))}
      {line && (
        <g>
          <line x1={line.x1} y1={line.y1} x2={line.x2} y2={line.y2}
                stroke={COLOR_LINE} strokeWidth="2" strokeDasharray="6 4" />
          <circle cx={line.x1} cy={line.y1} r="6" fill="none" stroke={COLOR_LINE} strokeWidth="2" />
          <circle cx={line.x2} cy={line.y2} r="6" fill="none" stroke={COLOR_LINE} strokeWidth="2" />
        </g>
      )}
    </svg>
  );
}
