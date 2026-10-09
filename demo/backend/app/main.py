import time

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from . import detectors  # noqa: F401
from . import matchers   # noqa: F401
from . import registry
from .image_io import decode_image_b64
from .schemas import (
    DetectRequest, DetectResponse, Detection,
    Match, MethodsResponse, DetectorMeta,
    PipelineRequest, PipelineResponse,
)

app = FastAPI(title="ALIKE Visualizer API", version="0.1.0")

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

    raw = match_fn(desc_a, desc_b, norm_a, **req.matcher_params)
    dt_ms = (time.perf_counter() - t0) * 1000.0

    return PipelineResponse(
        det_a=Detection.from_arrays(pts_a, desc_a, norm_a),
        det_b=Detection.from_arrays(pts_b, desc_b, norm_b),
        matches=[Match(a_idx=a, b_idx=b, dist=d) for a, b, d in raw],
        elapsed_ms=dt_ms,
        method=req.method,
        matcher=req.matcher,
    )

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