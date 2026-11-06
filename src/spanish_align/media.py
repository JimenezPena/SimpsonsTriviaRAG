"""Local video download and file management.

One-off downloads with yt-dlp (cached: never downloaded twice), deletion
after processing so no copyrighted videos pile up on disk, and utilities
to manage the media folder.
"""

import subprocess
from pathlib import Path

from .config import MEDIA_DIR, ensure_dirs


def shortcode_from_url(url: str) -> str:
    """Extract the shortcode: .../reel/CXYZ123/ -> CXYZ123"""
    tail = url.rstrip("/").rsplit("/", 1)[-1]
    return tail or "unknown"


def local_path(url: str) -> Path:
    ensure_dirs()
    return MEDIA_DIR / f"{shortcode_from_url(url)}.mp4"


def has_local(url: str) -> bool:
    return local_path(url).exists()


def download_clip(url: str) -> Path:
    """Download the clip unless it is already cached. Returns local path."""
    out = local_path(url)
    if out.exists():
        return out
    subprocess.run(
        ["yt-dlp", "-f", "mp4", "-o", str(out), url],
        check=True,
    )
    return out


def delete_clip(url: str) -> None:
    """Delete the local video (e.g. after transcription; the transcript
    stays cached as JSON, the video is no longer needed)."""
    p = local_path(url)
    if p.exists():
        p.unlink()


def media_stats() -> dict:
    """Total size of the video cache, for periodic cleanup."""
    files = list(MEDIA_DIR.glob("*.mp4")) if MEDIA_DIR.exists() else []
    return {
        "count": len(files),
        "total_mb": round(sum(f.stat().st_size for f in files) / 1e6, 1),
    }
