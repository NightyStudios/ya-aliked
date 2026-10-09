import base64
import cv2
import numpy as np


def decode_image_b64(image_b64: str) -> np.ndarray:
    """Accepts raw base64 or a data-URL. Returns BGR uint8 image."""
    if "," in image_b64:
        image_b64 = image_b64.split(",", 1)[1]
    raw = base64.b64decode(image_b64)
    arr = np.frombuffer(raw, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Failed to decode image from base64")
    return img