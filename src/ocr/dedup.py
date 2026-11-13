"""Image fingerprinting and semantic ticket deduplication helpers."""

from __future__ import annotations

import hashlib
import re
import unicodedata

import cv2
import numpy as np


def compute_sha256(image_bytes: bytes) -> str:
    """Compute exact SHA-256 hex digest for raw image bytes."""
    if not image_bytes:
        return ""
    return hashlib.sha256(image_bytes).hexdigest()


def compute_dhash(image_bytes: bytes, hash_size: int = 8) -> str:
    """Compute a 64-bit perceptual difference hash (dHash) as a 16-char hex string.

    Robust against Telegram JPEG recompression and minor resizing/brightness changes.
    Returns empty string if bytes cannot be decoded as an image.
    """
    if not image_bytes:
        return ""
    try:
        np_arr = np.frombuffer(image_bytes, np.uint8)
        gray = cv2.imdecode(np_arr, cv2.IMREAD_GRAYSCALE)
        if gray is None or gray.size == 0:
            return ""
        resized = cv2.resize(gray, (hash_size + 1, hash_size), interpolation=cv2.INTER_AREA)
        diff = resized[:, 1:] > resized[:, :-1]
        bits = 0
        for val in diff.flatten():
            bits = (bits << 1) | int(bool(val))
        hex_width = (hash_size * hash_size) // 4
        return f"{bits:0{hex_width}x}"
    except Exception:
        return ""


def hamming_distance(hash_a: str, hash_b: str) -> int:
    """Compute bit-level Hamming distance between two hex dHashes."""
    if not hash_a or not hash_b or len(hash_a) != len(hash_b):
        return 999
    try:
        return (int(hash_a, 16) ^ int(hash_b, 16)).bit_count()
    except ValueError:
        return 999


def normalize_company(name: str) -> str:
    """Normalize store/vendor name for fuzzy comparison across OCR runs."""
    if not name:
        return ""
    nfkd = unicodedata.normalize("NFKD", name)
    ascii_text = "".join(ch for ch in nfkd if not unicodedata.combining(ch)).lower()
    # Colapsar siglas societarias con puntos (p. ej. "s.l.", "s.a.", "s.l.u.") antes de limpiar signos
    ascii_text = re.sub(r"\bs\.\s*l\.\s*u\.?\b", " slu ", ascii_text)
    ascii_text = re.sub(r"\bs\.\s*l\.\s*l\.?\b", " sll ", ascii_text)
    ascii_text = re.sub(r"\bs\.\s*l\.?\b", " sl ", ascii_text)
    ascii_text = re.sub(r"\bs\.\s*a\.?\b", " sa ", ascii_text)
    cleaned = re.sub(r"[^a-z0-9\s]", " ", ascii_text).strip()
    tokens = [
        tok for tok in cleaned.split()
        if tok not in {"sl", "sa", "slu", "sll", "cb", "sc", "bar", "cafe", "cafeteria", "restaurante"}
    ]
    return " ".join(tokens).strip() or re.sub(r"\s+", " ", cleaned)


def companies_match(empresa_a: str, empresa_b: str) -> bool:
    """Return True if two vendor names refer to the same business."""
    norm_a = normalize_company(empresa_a)
    norm_b = normalize_company(empresa_b)
    if not norm_a or not norm_b:
        return False
    if "desconocido" in (norm_a, norm_b):
        return False
    if norm_a == norm_b:
        return True
    if min(len(norm_a), len(norm_b)) >= 4 and (norm_a in norm_b or norm_b in norm_a):
        return True
    return False
