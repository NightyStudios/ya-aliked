import { pairColor, COLOR_UNMATCHED } from '../constants';
import { unmatchedIdx } from '../points';

const fmt = (v) => (Math.round(v * 10) / 10).toString();

export function MatchList({
  matches, kpsA, kpsB, selected, setSelected, setHovered,
}) {
  const isActive = (side, idx) =>
    selected && selected.side === side && selected.idx === idx;

  const handlePair = (m) => {
    setSelected(s => (s && s.side === 'a' && s.idx === m.aIdx)
      ? null : { side: 'a', idx: m.aIdx });
  };

  const handleUnp = (side, idx) => {
    setSelected(s => (s && s.side === side && s.idx === idx)
      ? null : { side, idx });
  };

  const ua = unmatchedIdx(kpsA, matches, 'a');
  const ub = unmatchedIdx(kpsB, matches, 'b');
  const hasUnp = ua.length || ub.length;

  const unpSection = (side, idxs, kps, label) => (
    <div className="unp-section">
      <div className="matchlist-sub">
        <span>{label}</span>
        <span className="matchlist-count">{idxs.length}</span>
      </div>
      {idxs.map(idx => {
        const kp = kps[idx];
        if (!kp) return null;
        return (
          <button
            key={idx}
            className={'matchrow unp' + (isActive(side, idx) ? ' active' : '')}
            style={{ '--pair': COLOR_UNMATCHED }}
            onClick={() => handleUnp(side, idx)}
            onMouseEnter={() => setHovered({ side, idx })}
            onMouseLeave={() => setHovered(h =>
              (h && h.side === side && h.idx === idx) ? null : h)}
          >
            <span className="matchrow-coords">
              <span className={'coord coord-' + side}>
                <i className="dot" />{side.toUpperCase()} ({fmt(kp.x)}, {fmt(kp.y)})
              </span>
            </span>
          </button>
        );
      })}
    </div>
  );

  return (
    <aside className="matchlist">
      <div className="matchlist-head">
        <span className="matchlist-title">Пары</span>
        <span className="matchlist-count">{matches.length}</span>
      </div>

      <div className="matchlist-body">
        {!matches.length && <div className="matchlist-empty">Нет сопоставлений</div>}

        {matches.map((m, i) => {
          const ka = kpsA[m.aIdx], kb = kpsB[m.bIdx];
          if (!ka || !kb) return null;
          const color = pairColor(i);

          return (
            <button
              key={i}
              className={'matchrow' + (isActive('a', m.aIdx) ? ' active' : '')}
              style={{ '--pair': color }}
              onClick={() => handlePair(m)}
              onMouseEnter={() => setHovered({ side: 'a', idx: m.aIdx })}
              onMouseLeave={() => setHovered(h =>
                (h && h.side === 'a' && h.idx === m.aIdx) ? null : h)}
            >
              <span className="matchrow-no">{i + 1}</span>
              <span className="matchrow-coords">
                <span className="coord coord-a">
                  <i className="dot" />A ({fmt(ka.x)}, {fmt(ka.y)})
                </span>
                <span className="coord coord-b">
                  <i className="dot" />B ({fmt(kb.x)}, {fmt(kb.y)})
                </span>
              </span>
              <span className="matchrow-dist">{m.dist ?? '—'}</span>
            </button>
          );
        })}

        {hasUnp > 0 && (
          <>
            <div className="matchlist-divider" />
            {unpSection('a', ua, kpsA, 'Непарные A')}
            {unpSection('b', ub, kpsB, 'Непарные B')}
          </>
        )}
      </div>
    </aside>
  );
}
