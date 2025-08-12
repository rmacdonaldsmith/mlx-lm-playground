from __future__ import annotations
import mlx.core as mx  # type: ignore


def clear_cache() -> None:
    """Best-effort clear across MLX versions and reset peak memory if available."""
    try:
        mx.clear_cache()  # new
    except Exception:
        try:
            mx.metal.clear_cache()  # old
        except Exception:
            pass
    try:
        mx.reset_peak_memory()
    except Exception:
        pass


def snapshot_bytes() -> dict:
    """Return a dict of active/cache/peak memory in bytes (keys may be zero if not supported)."""
    get_cache = getattr(mx, "get_cache_memory", lambda: 0)
    get_peak = getattr(mx, "get_peak_memory", lambda: 0)
    try:
        active = mx.get_active_memory()
    except Exception:
        active = 0
    try:
        cache = get_cache()
    except Exception:
        cache = 0
    try:
        peak = get_peak()
    except Exception:
        peak = 0
    return {"active": active, "cache": cache, "peak": peak}
