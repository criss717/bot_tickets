"""OpenCV image preprocessor for ticket OCR."""

from __future__ import annotations

import cv2
import numpy as np


def preprocess_image(image_bytes: bytes) -> bytes:
    """Preprocess a ticket image for better OCR results.

    Steps:
        1. Decode image from bytes
        2. Convert to grayscale
        3. Apply bilateral filter (noise reduction while keeping edges)
        4. Apply adaptive threshold (binarization)
        5. Encode back to JPEG bytes

    Args:
        image_bytes: Raw image bytes (JPEG, PNG, etc.)

    Returns:
        Preprocessed image as JPEG bytes.

    Raises:
        ValueError: If the image cannot be decoded.
    """
    # Decode
    if not image_bytes:
        raise ValueError("Cannot decode image: empty input")
    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Cannot decode image")

    # Grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Noise reduction
    denoised = cv2.bilateralFilter(gray, 11, 17, 17)

    # Adaptive threshold for binarization
    thresh = cv2.adaptiveThreshold(
        denoised, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        11, 2,
    )

    # Encode back to JPEG
    _, buffer = cv2.imencode(".jpg", thresh)
    return buffer.tobytes()


def resize_if_large(image_bytes: bytes, max_width: int = 1024) -> bytes:
    """Resize image if its width or height exceeds max_width, maintaining aspect ratio.

    Large images waste API tokens and memory when sent to Vision APIs.
    This keeps the image readable while reducing token/processing cost.

    Args:
        image_bytes: Raw image bytes.
        max_width: Maximum dimension (width or height) in pixels.

    Returns:
        Resized image bytes (or original if already small enough).
    """
    if not image_bytes:
        raise ValueError("Cannot decode image")
    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Cannot decode image")

    height, width = img.shape[:2]
    max_dim = max(width, height)
    if max_dim <= max_width:
        return image_bytes

    scale = max_width / max_dim
    new_width = int(width * scale)
    new_height = int(height * scale)
    resized = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_AREA)

    _, buffer = cv2.imencode(".jpg", resized)
    return buffer.tobytes()
