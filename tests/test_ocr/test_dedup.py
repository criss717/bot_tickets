"""Tests for image fingerprinting and semantic ticket deduplication."""

from __future__ import annotations

import cv2
import numpy as np

from ocr.dedup import (
    companies_match,
    compute_dhash,
    compute_sha256,
    hamming_distance,
    normalize_company,
)


def _make_sample_jpeg(pattern: str = "receipt_a", quality: int = 92, width: int = 400) -> bytes:
    """Generate distinct synthetic images encoded as JPEG bytes."""
    img = np.full((300, width, 3), 245, dtype=np.uint8)
    if pattern == "receipt_a":
        cv2.rectangle(img, (20, 20), (width - 20, 70), (20, 20, 20), -1)
        cv2.putText(img, "MONTEMAYOR SL", (35, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.rectangle(img, (30, 120), (width - 30, 240), (60, 60, 60), 2)
        cv2.putText(img, "TOTAL 28.40 EUR", (45, 190), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (10, 10, 10), 2)
    else:
        cv2.circle(img, (width // 2, 150), 90, (30, 30, 30), -1)
        cv2.line(img, (0, 0), (width, 300), (0, 0, 0), 8)
    ok, buf = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    assert ok
    return buf.tobytes()


class TestImageHashes:
    def test_sha256_exact_match_and_diff(self):
        img1 = _make_sample_jpeg("receipt_a", quality=92)
        img2 = _make_sample_jpeg("receipt_a", quality=70)
        assert compute_sha256(img1) == compute_sha256(img1)
        assert len(compute_sha256(img1)) == 64
        assert compute_sha256(img1) != compute_sha256(img2)
        assert compute_sha256(b"") == ""

    def test_dhash_tolerates_recompression_and_resizing(self):
        orig = _make_sample_jpeg("receipt_a", quality=95, width=600)
        # Simular el redimensionado y recompresión que aplica Telegram sobre la misma imagen
        arr = np.frombuffer(orig, dtype=np.uint8)
        decoded = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        assert decoded is not None
        resized = cv2.resize(decoded, (320, 160), interpolation=cv2.INTER_AREA)
        ok, buf = cv2.imencode(".jpg", resized, [int(cv2.IMWRITE_JPEG_QUALITY), 45])
        assert ok
        recompressed = buf.tobytes()

        h1 = compute_dhash(orig)
        h2 = compute_dhash(recompressed)
        assert len(h1) == 16
        assert len(h2) == 16
        assert hamming_distance(h1, h2) <= 4

    def test_dhash_distinguishes_unrelated_images(self):
        img_a = _make_sample_jpeg("receipt_a")
        img_b = _make_sample_jpeg("circle_pattern")
        assert hamming_distance(compute_dhash(img_a), compute_dhash(img_b)) > 8

    def test_dhash_handles_invalid_bytes(self):
        assert compute_dhash(b"not-an-image") == ""
        assert compute_dhash(b"") == ""
        assert hamming_distance("invalid", "0" * 16) == 999


class TestCompanyMatching:
    def test_normalize_and_match_variations(self):
        assert normalize_company("Café Bar El Rincón S.L.") == "el rincon"
        assert companies_match("MONTEMAYOR ARAHAL S.L.", "Restaurante Montemayor") is True
        assert companies_match("Café Bar Casa Pepe", "Casa Pepe") is True
        assert companies_match("Bar", "Bar") is True
        assert companies_match("Repsol", "Carrefour") is False
        assert companies_match("Desconocido", "Desconocido") is False
        assert companies_match("", "Repsol") is False
