import threading
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from . import detectors  # noqa: F401
from . import matchers   # noqa: F401
from . import registry
from . import segment
from .keypoint_filter import decode_class_map, filter_by_mask
from . import similarity
from .ransac import ransac_filter_matches
from .image_io import decode_image_b64, decode_image_rgb_b64
from .schemas import (
    DetectRequest, DetectResponse, Detection,
    Match, MethodsResponse, DetectorMeta,
    PipelineRequest, PipelineResponse,
    SegmentRequest, SegmentResponse, SegmentInstance,
    KeypointFilterRequest, KeypointFilterResponse,
    SimilarityRequest, SimilarityResponse,
)


@asynccontextmanager
async def lifespan(_: FastAPI):
    # подгрузить Mask2Former в фоне, чтобы первый запрос не ждал скачивание весов
    threading.Thread(target=segment.warmup, daemon=True).start()
    yield


app = FastAPI(title="ALIKE Visualizer API", version="0.1.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/api/health")
def health():
    return {"ok": True}


@app.get("/api/methods", response_model=MethodsResponse)
def methods():
    return MethodsResponse(
        detectors=list(registry.DETECTORS.keys()),
        matchers=list(registry.MATCHERS.keys()),
        detector_meta={
            name: DetectorMeta(
                available=meta["available"],
                description=meta["description"],
                hidden=meta["hidden"],
                group=meta["group"],
            )
            for name, meta in registry.DETECTORS.items()
        },
    )


@app.post("/api/detect", response_model=DetectResponse)
def detect(req: DetectRequest):
    try:
        fn = registry.get_detector(req.method)
    except (KeyError, NotImplementedError) as e:
        raise HTTPException(status_code=400, detail=str(e))

    img = decode_image_b64(req.image.image_b64)

    t0 = time.perf_counter()
    pts, desc, norm = fn(img, **req.params)
    dt_ms = (time.perf_counter() - t0) * 1000.0

    return DetectResponse(
        detection=Detection.from_arrays(pts, desc, norm),
        elapsed_ms=dt_ms,
    )


@app.post("/api/pipeline", response_model=PipelineResponse)
def pipeline(req: PipelineRequest):
    try:
        detect_fn = registry.get_detector(req.method)
        match_fn = registry.get_matcher(req.matcher)
    except (KeyError, NotImplementedError) as e:
        raise HTTPException(status_code=400, detail=str(e))

    img_a = decode_image_b64(req.image_a.image_b64)
    img_b = decode_image_b64(req.image_b.image_b64)

    t0 = time.perf_counter()

    pts_a, desc_a, norm_a = detect_fn(img_a, **req.params)
    pts_b, desc_b, norm_b = detect_fn(img_b, **req.params)

    if norm_a != norm_b:
        raise HTTPException(400, "Descriptor norms mismatch between A and B")

    try:
        raw = match_fn(
            desc_a, desc_b, norm_a,
            pts_a=pts_a, pts_b=pts_b,
            image_size_a=(img_a.shape[1], img_a.shape[0]),
            image_size_b=(img_b.shape[1], img_b.shape[0]),
            **req.matcher_params,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    if req.ransac:
        raw = ransac_filter_matches(raw, pts_a, pts_b, thresh=req.ransac_thresh)

    dt_ms = (time.perf_counter() - t0) * 1000.0

    return PipelineResponse(
        det_a=Detection.from_arrays(pts_a, desc_a, norm_a),
        det_b=Detection.from_arrays(pts_b, desc_b, norm_b),
        matches=[Match(a_idx=a, b_idx=b, dist=d) for a, b, d in raw],
        elapsed_ms=dt_ms,
        method=req.method,
        matcher=req.matcher,
    )


@app.post("/api/segment", response_model=SegmentResponse)
def segment_image_endpoint(req: SegmentRequest):
    try:
        img = decode_image_b64(req.image.image_b64)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Bad image: {e}")

    try:
        out = segment.segment_image(img)
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Segmentation failed: {e}")

    return SegmentResponse(
        overlay_png_b64=out["overlay_png_b64"],
        boundary_png_b64=out["boundary_png_b64"],
        class_map_png_b64=out["class_map_png_b64"],
        classes=out["classes"],
        instances=[SegmentInstance(**i) for i in out["instances"]],
        width=out["width"],
        height=out["height"],
        model=out["model"],
        mode=out["mode"],
        threshold=out["threshold"],
        elapsed_ms=out["elapsed_ms"],
    )


@app.post("/api/filter-keypoints", response_model=KeypointFilterResponse)
def filter_keypoints(req: KeypointFilterRequest):
    try:
        return filter_by_mask(decode_class_map(req.class_map_png_b64), req.points,
                              req.excluded_classes, req.margin)
    except (ValueError, TypeError) as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/api/similarity", response_model=SimilarityResponse)
def similarity_endpoint(req: SimilarityRequest):
    """Сравнивает эталон с начальным и с наложенным изображением."""
    try:
        ref = decode_image_rgb_b64(req.image_ref.image_b64)
        initial = decode_image_rgb_b64(req.image_initial.image_b64)
        warped = decode_image_rgb_b64(req.image_warped.image_b64)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Bad image: {e}")

    try:
        out = similarity.compare_all(ref, initial, warped, tile=req.tile)
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Similarity failed: {e}")

    return SimilarityResponse(**out)

'''
@app.post("/api/neighbors", response_model=NeighborsResponse)
def neighbors(req: NeighborsRequest):
    fn = registry.get_detector(req.method)
    img_a = decode_image_b64(req.image_a.image_b64)
    img_b = decode_image_b64(req.image_b.image_b64)
    _, desc_a, norm = fn(img_a, **req.params)
    pts_b, desc_b, _ = fn(img_b, **req.params)

    if req.kp_a_idx >= desc_a.shape[0]:
        raise HTTPException(400, "kp_a_idx out of range")

    q = desc_a[req.kp_a_idx:req.kp_a_idx + 1].astype(np.int32)
    # Hamming distance для uint8 ORB, L2 — для float
    if norm == "hamming":
        d = np.unpackbits(desc_b, axis=1).sum(axis=1) + \
            np.unpackbits(q, axis=1).sum(axis=1)[0] - \
            2 * np.unpackbits(desc_b, axis=1) @ np.unpackbits(q, axis=1)[0]
    else:
        d = np.linalg.norm(desc_b - q, axis=1)

    order = np.argsort(d)[:req.top_k]
    return NeighborsResponse(
        distances=[float(x) for x in d[order]],
        b_indices=[int(x) for x in order],
        det_b_points=[(float(x), float(y)) for x, y in pts_b[order]],
    )
'''
