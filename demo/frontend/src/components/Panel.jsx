import { useState } from 'react';
import { KeypointOverlay } from './KeypointOverlay';

export function Panel({
  side, img, kps, tf, panelRef, onWheel, onMouseDown, onPanelClick, onPanelDblClick,
  hovered, setHovered, selected, colorMap, isActive, onFiles,
  pairNo, connectFrom, onPointClick, onStartDrag,
}) {
  const [over, setOver] = useState(false);

  const handleDrop = (e) => {
    e.preventDefault(); setOver(false);
    if (e.dataTransfer.files?.length) onFiles(e.dataTransfer.files);
  };

  if (!img) {
    return (
      <div ref={panelRef}
           className={'panel empty' + (over ? ' over' : '')}
           onDragOver={(e) => { e.preventDefault(); setOver(true); }}
           onDragLeave={() => setOver(false)}
           onDrop={handleDrop}>
        <div>
          Перетащите изображение сюда<br />
          <span className="hint">или используйте «Загрузить 2 изображения»</span>
        </div>
      </div>
    );
  }

  const { url, width, height } = img;

  return (
    <div ref={panelRef} className="panel"
         onWheel={onWheel} onMouseDown={onMouseDown}
         onClick={onPanelClick} onDoubleClick={(e) => onPanelDblClick(side, e)}>
      <div className="stage"
           style={{
             transform: `translate(${tf.tx}px, ${tf.ty}px) scale(${tf.s})`,
             transformOrigin: '0 0',
           }}>
        <img src={url} width={width} height={height} draggable={false}
             style={{ display: 'block', pointerEvents: 'none' }} />
        <KeypointOverlay
          side={side} kps={kps} tf={tf} colorMap={colorMap} pairNo={pairNo}
          hovered={hovered} setHovered={setHovered}
          selected={selected} isActive={isActive}
          connectFrom={connectFrom}
          onPointClick={onPointClick}
          onStartDrag={onStartDrag}
        />
      </div>
    </div>
  );
}
