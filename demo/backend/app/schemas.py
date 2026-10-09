import base64
from typing import Literal, List, Tuple, Dict, Any

import numpy as np
from pydantic import BaseModel, Field


Norm = Literal["hamming", "l2"]
DescDtype = Literal["uint8", "float32"]


class ImagePayload(BaseModel):
    image_b64: str
    filename: str | None = None


class Detection(BaseModel):
    points: List[Tuple[float, float]]
    desc_dtype: DescDtype
    desc_shape: Tuple[int, int]        # (N, D)
    desc_b64: str                      # base64 от desc.tobytes()
    norm: Norm

    @classmethod
    def from_arrays(cls, points: np.ndarray, desc: np.ndarray, norm: Norm) -> "Detection":
        pts = np.asarray(points, dtype=np.float32).reshape(-1, 2)
        desc = np.asarray(desc)

        if desc.size == 0:
            desc = np.zeros((0, 32 if norm == "hamming" else 128),
                            dtype=np.uint8 if norm == "hamming" else np.float32)
        if norm == "hamming":
            desc = desc.astype(np.uint8, copy=False)
            dtype_name: DescDtype = "uint8"
        else:
            desc = desc.astype(np.float32, copy=False)
            dtype_name = "float32"

        return cls(
            points=[(float(x), float(y)) for x, y in pts],
            desc_dtype=dtype_name,
            desc_shape=(int(desc.shape[0]), int(desc.shape[1])),
            desc_b64=base64.b64encode(desc.tobytes()).decode("ascii"),
            norm=norm,
        )


class DetectRequest(BaseModel):
    method: str
    image: ImagePayload
    params: Dict[str, Any] = Field(default_factory=dict)


class DetectResponse(BaseModel):
    detection: Detection
    elapsed_ms: float


class Match(BaseModel):
    a_idx: int
    b_idx: int
    dist: float


class PipelineRequest(BaseModel):
    method: str
    matcher: str = "bf"
    image_a: ImagePayload
    image_b: ImagePayload
    params: Dict[str, Any] = Field(default_factory=dict)
    matcher_params: Dict[str, Any] = Field(default_factory=dict)


class PipelineResponse(BaseModel):
    det_a: Detection
    det_b: Detection
    matches: List[Match]
    elapsed_ms: float
    method: str
    matcher: str


class DetectorMeta(BaseModel):
    available: bool
    description: str = ""
    hidden: bool = False
    group: str = "ORB"


class MethodsResponse(BaseModel):
    detectors: List[str]
    matchers: List[str]
    detector_meta: Dict[str, DetectorMeta]

'''
# backend/app/schemas.py (добавить)
class NeighborsRequest(BaseModel):
    method: str
    image_a: ImagePayload
    image_b: ImagePayload
    kp_a_idx: int
    top_k: int = 64
    params: Dict[str, Any] = Field(default_factory=dict)

class NeighborsResponse(BaseModel):
    distances: List[float]        # отсортированы по возрастанию, len <= top_k
    b_indices: List[int]          # соответствие indices в det_b
    det_b_points: List[Tuple[float, float]]
'''