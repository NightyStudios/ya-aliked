import { useRef, useState } from 'react';
import { KeypointOverlay } from './KeypointOverlay';

const ACCEPT = 'image/*,.ppm,.pgm,.pnm';

export function Panel({
  side, img, kps, tf, panelRef, onWheel, onMouseDown, onPanelClick, onPanelDblClick,
  hovered, setHovered, selected, colorMap, isActive, onFile,
  pairNo, normMap, connectFrom, onPointClick, onStartDrag,
  dragging, overlayLayerRef, overlayAlpha,
  maskLayerRef, maskAlpha, maskVisible, hidden,
  rwarpLayerRef, rwarpMeshRef, rwarpAlpha, rwarpVisible,
  editMode, editHud,
}) {
  const [over, setOver] = useState(false);
  const inputRef = useRef(null);

  const pickFile = () => inputRef.current?.click();

  const handleInput = (e) => {
    const f = e.target.files?.[0];
    if (f) onFile(f, side);
    e.target.value = '';
  };

  const handleDrop = (e) => {
    e.preventDefault(); setOver(false);
    const f = Array.from(e.dataTransfer.files || [])
      .find(x => x.type.startsWith('image/'));
    if (f) onFile(f, side);
  };

  const input = (
    <input ref={inputRef} type="file" accept={ACCEPT} hidden onChange={handleInput} />
  );

  if (!img) {
    return (
      <div ref={panelRef}
           className={'panel empty' + (over ? ' over' : '')}
           onClick={pickFile}
           onDragOver={(e) => { e.preventDefault(); setOver(true); }}
           onDragLeave={() => setOver(false)}
           onDrop={handleDrop}>
        {input}
        <div>
          Кликните или перетащите изображение сюда<br />
          <span className="hint">{side === 'a' ? 'левое' : 'правое'} изображение</span>
        </div>
      </div>
    );
  }

  const { url, width, height } = img;

  return (
    <div ref={panelRef} className={'panel' + (over ? ' over' : '')}
         onWheel={onWheel} onMouseDown={onMouseDown}
         onClick={onPanelClick} onDoubleClick={(e) => onPanelDblClick(side, e)}
         onDragOver={(e) => { e.preventDefault(); setOver(true); }}
         onDragLeave={() => setOver(false)}
         onDrop={handleDrop}>
      {input}
      {editHud && (
        <div className="coord-hud" role="status">
          x {editHud.x.toFixed(1)} · y {editHud.y.toFixed(1)}
        </div>
      )}
      <div className="stage"
           style={{
             transform: `translate(${tf.tx}px, ${tf.ty}px) scale(${tf.s})`,
             transformOrigin: '0 0',
           }}>
        <img src={url} width={width} height={height} draggable={false}
             style={{ display: 'block', pointerEvents: 'none' }} />
        {overlayLayerRef && (
          <canvas ref={overlayLayerRef} className="overlay-layer"
                  style={{ opacity: overlayAlpha ?? 0.5 }} />
        )}
        {maskLayerRef && (
          <canvas ref={maskLayerRef} className="mask-layer"
                  style={{ opacity: maskVisible ? (maskAlpha ?? 0.75) : 0 }} />
        )}
        {rwarpLayerRef && (
          <>
            <canvas ref={rwarpLayerRef} className="rwarp-layer"
                    style={{ opacity: rwarpVisible ? (rwarpAlpha ?? 0.5) : 0 }} />
            <canvas ref={rwarpMeshRef} className="rwarp-mesh"
                    style={{ opacity: rwarpVisible ? 0.9 : 0 }} />
          </>
        )}
        <KeypointOverlay
          side={side} kps={kps} tf={tf} colorMap={colorMap} pairNo={pairNo}
          normMap={normMap} hidden={hidden}
          hovered={hovered} setHovered={setHovered}
          selected={selected} isActive={isActive}
          connectFrom={connectFrom}
          onPointClick={onPointClick}
          onStartDrag={onStartDrag}
          dragging={dragging}
          editMode={editMode}
        />
      </div>
      {over && <div className="drop-hint">Заменить изображение</div>}
    </div>
  );
}
