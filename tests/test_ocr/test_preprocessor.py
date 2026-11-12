"""Tests for OCR preprocessor."""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from ocr.preprocessor import preprocess_image, resize_if_large


def _make_test_image_bytes(width: int = 200, height: int = 100) -> bytes:
    """Create a simple test image as JPEG bytes."""
    img = np.zeros((height, width, 3), dtype=np.uint8)
    # Add some white text-like rectangles
    cv2.rectangle(img, (10, 10), (100, 30), (255, 255, 255), -1)
    cv2.rectangle(img, (10, 40), (150, 60), (255, 255, 255), -1)
    _, buffer = cv2.imencode(".jpg", img)
    return buffer.tobytes()


class TestPreprocessImage:
    """Tests for image preprocessing."""

    def test_preprocess_returns_bytes(self):
        img_bytes = _make_test_image_bytes()
        result = preprocess_image(img_bytes)
        assert isinstance(result, bytes)
        assert len(result) > 0

    def test_preprocess_valid_jpeg_output(self):
        img_bytes = _make_test_image_bytes()
        result = preprocess_image(img_bytes)
        # Should be decodable as JPEG
        nparr = np.frombuffer(result, np.uint8)
        decoded = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)
        assert decoded is not None

    def test_preprocess_invalid_image_raises(self):
        with pytest.raises(ValueError, match="Cannot decode"):
            preprocess_image(b"not an image")

    def test_preprocess_empty_bytes_raises(self):
        with pytest.raises(ValueError, match="empty input"):
            preprocess_image(b"")

    def test_preprocess_png_input(self):
        """Should handle PNG input too."""
        img = np.zeros((100, 200, 3), dtype=np.uint8)
        _, buffer = cv2.imencode(".png", img)
        result = preprocess_image(buffer.tobytes())
        assert isinstance(result, bytes)
        assert len(result) > 0


class TestResizeIfLarge:
    """Tests for image resizing."""

    def test_small_image_unchanged(self):
        img_bytes = _make_test_image_bytes(width=200, height=100)
        result = resize_if_large(img_bytes, max_width=1024)
        # Should be the same bytes (no resize needed)
        assert result == img_bytes

    def test_large_image_resized(self):
        img_bytes = _make_test_image_bytes(width=2000, height=1000)
        result = resize_if_large(img_bytes, max_width=1024)
        # Decode and check width
        nparr = np.frombuffer(result, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        assert img.shape[1] <= 1024

    def test_resize_preserves_aspect_ratio(self):
        img_bytes = _make_test_image_bytes(width=2000, height=500)
        result = resize_if_large(img_bytes, max_width=1000)
        nparr = np.frombuffer(result, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        # Original ratio: 2000/500 = 4:1
        # Resized: 1000/250 = 4:1
        assert img.shape[1] == 1000
        assert img.shape[0] == 250

    def test_invalid_image_raises(self):
        with pytest.raises(ValueError, match="Cannot decode"):
            resize_if_large(b"garbage")

    def test_exact_max_width_unchanged(self):
        img_bytes = _make_test_image_bytes(width=1024, height=500)
        result = resize_if_large(img_bytes, max_width=1024)
        assert result == img_bytes
