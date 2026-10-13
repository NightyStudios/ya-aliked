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


class SegmentRequest(BaseModel):
    image: ImagePayload
    # Совместимость со старым API; семантическая сегментация не использует порог.
    threshold: float | None = Field(default=None, gt=0.0, le=1.0)


class SegmentInstance(BaseModel):
    id: int                          # id связной семантической области
    label_id: int
    label: str
    score: float | None = None
    area: int                      # площадь в пикселях оверлея
    bbox: List[int]                  # [x0, y0, x1, y1] в координатах оверлея


class SegmentResponse(BaseModel):
    overlay_png_b64: str           # RGBA-PNG с масками (прозрачность заложена в канал A)
    boundary_png_b64: str          # серый PNG карты границ областей (255 = граница)
    class_map_png_b64: str         # ID класса в каждом пикселе исходного изображения
    classes: List["SegmentClass"]
    instances: List[SegmentInstance]
    width: int                     # размер оверлея — фронт растянет на исходный кадр
    height: int
    model: str
    mode: Literal["semantic"] = "semantic"
    threshold: float | None = None
    elapsed_ms: float


class SegmentClass(BaseModel):
    id: int
    label: str
    color: str
    area: int


class KeypointFilterRequest(BaseModel):
    class_map_png_b64: str
    points: List[Tuple[float, float]]
    excluded_classes: List[int] = Field(default_factory=list)
    margin: float = Field(default=0, ge=0, allow_inf_nan=False)


class KeypointFilterResponse(BaseModel):
    keep: List[bool]
    saved_by_margin: List[bool]


class SimilarityRequest(BaseModel):
    image_ref: ImagePayload           # эталон
    image_initial: ImagePayload       # начальное (правое) изображение
    image_warped: ImagePayload        # наложенное (деформированное правое)
    tile: int = Field(default=128, ge=8, le=512)   # размер куска для локальных сдвигов


class SimilarityMetrics(BaseModel):
    # None — метрика не определена (нечерные поля, вырожденный кусок и т.п.)
    mae: float | None = None                  # средняя абсолютная разница, 0–255
    psnr: float | None = None                 # дБ
    ssim: float | None = None
    ncc: float | None = None                  # корреляция яркостей
    dx: float | None = None                   # общий сдвиг, px
    dy: float | None = None
    tiles: int = 0                            # оцененных кусков tile×tile
    tile_median: float | None = None          # медиана |сдвига| кусков, px
    tiles_under_1px: float | None = None      # доля кусков со сдвигом < 1 px (0..1)


class SimilarityResponse(BaseModel):
    ref_size: Tuple[int, int]                 # (w, h) эталона до обрезки
    compared_size: Tuple[int, int]            # (w, h) общей области сравнения
    initial: SimilarityMetrics
    warped: SimilarityMetrics
    elapsed_ms: float


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
