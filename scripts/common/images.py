"""Image handling: URL, base64 download/cache, and fallback."""

import base64
import hashlib
import os
import tempfile
from pathlib import Path

import httpx

# Non-descriptive URL — no content hints in the name
DEFAULT_IMAGE_URL = "https://picsum.photos/id/274/800/600"

_CACHE_DIR = Path(tempfile.gettempdir()) / "llm_api_simple_tests_image_cache"
_ASSETS_DIR = Path(__file__).resolve().parent.parent.parent / "assets"


def get_image_url() -> str:
    """Return the test image URL (env override or default)."""
    return os.environ.get("TEST_IMAGE_URL", DEFAULT_IMAGE_URL)


def _cache_path(url: str) -> Path:
    url_hash = hashlib.sha256(url.encode()).hexdigest()
    return _CACHE_DIR / url_hash


def download_image_as_base64(url: str | None = None) -> tuple[str, str]:
    """Download image and return (base64_data, mime_type).

    Uses disk cache. Falls back to assets/test_image.jpg on failure.
    """
    if url is None:
        url = get_image_url()

    # Check cache
    cached_data = _cache_path(url).with_suffix(".data")
    cached_meta = _cache_path(url).with_suffix(".meta")
    if cached_data.exists() and cached_meta.exists():
        b64 = base64.b64encode(cached_data.read_bytes()).decode()
        mime = cached_meta.read_text().strip()
        return b64, mime

    # Download
    try:
        resp = httpx.get(url, follow_redirects=True, timeout=30.0)
        resp.raise_for_status()
        raw = resp.content
        mime = resp.headers.get("content-type", "image/jpeg").split(";")[0].strip()

        # Save to cache
        _CACHE_DIR.mkdir(parents=True, exist_ok=True)
        cached_data.write_bytes(raw)
        cached_meta.write_text(mime)

        return base64.b64encode(raw).decode(), mime
    except Exception:
        return load_fallback_image_base64()


def load_fallback_image_base64() -> tuple[str, str]:
    """Load assets/test_image.jpg as base64."""
    fallback = _ASSETS_DIR / "test_image.jpg"
    if not fallback.exists():
        raise FileNotFoundError(f"Fallback image not found: {fallback}")
    raw = fallback.read_bytes()
    return base64.b64encode(raw).decode(), "image/jpeg"
