"""Deterministic cache for Stage 6 simulation runs."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Dict, Optional


CACHE_PATH = Path(__file__).resolve().parent / "simulation_cache.json"


def _load_cache() -> Dict[str, Any]:
    if not CACHE_PATH.exists():
        return {}
    return json.loads(CACHE_PATH.read_text(encoding="utf-8"))


def _save_cache(cache: Dict[str, Any]) -> None:
    CACHE_PATH.write_text(json.dumps(cache, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def config_hash(payload: Dict[str, Any]) -> str:
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def get_cached_result(payload: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    return _load_cache().get(config_hash(payload))


def set_cached_result(payload: Dict[str, Any], result: Dict[str, Any]) -> None:
    cache = _load_cache()
    cache[config_hash(payload)] = result
    _save_cache(cache)
