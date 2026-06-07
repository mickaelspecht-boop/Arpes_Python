"""Cache/signature helpers for raw ARPES loading."""
from __future__ import annotations

import copy
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np


RAW_LOAD_CACHE_VERSION = 3


def freeze_cache_value(value: Any) -> Any:
    """Transform a context value into a stable hashable cache key part."""
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return tuple((str(k), freeze_cache_value(v)) for k, v in sorted(value.items(), key=lambda item: str(item[0])))
    if isinstance(value, (list, tuple)):
        return tuple(freeze_cache_value(v) for v in value)
    if isinstance(value, set):
        return tuple(sorted(freeze_cache_value(v) for v in value))
    try:
        hash(value)
        return value
    except TypeError:
        return repr(value)


def clone_loaded_value(value: Any) -> Any:
    """Copy metadata/containers while sharing large numpy arrays read-only."""
    if isinstance(value, np.ndarray):
        return value
    if isinstance(value, dict):
        return {k: clone_loaded_value(v) for k, v in value.items()}
    if isinstance(value, list):
        return [clone_loaded_value(v) for v in value]
    if isinstance(value, tuple):
        return tuple(clone_loaded_value(v) for v in value)
    try:
        return copy.deepcopy(value)
    except Exception:
        return value


def path_signature(path: Path, parent) -> tuple:
    try:
        p = path.resolve()
    except Exception:
        p = path
    cache = getattr(parent, "_path_signature_cache", None)
    cache_key = str(p)
    quick_sig = quick_path_signature(p)
    if cache is not None and not p.is_dir():
        cached = cache.get(cache_key)
        if cached is not None and cached[0] == quick_sig:
            cache.move_to_end(cache_key)
            parent._last_path_signature_cache_hit = True
            return cached[1]
    parent._last_path_signature_cache_hit = False

    if p.is_dir():
        items = []
        try:
            for child in sorted(p.rglob("*")):
                if not child.is_file():
                    continue
                rel_parts = child.relative_to(p).parts
                if rel_parts and rel_parts[0] in {".arpes_cache", ".arpes_theory_cache"}:
                    continue
                st = child.stat()
                items.append(("/".join(rel_parts), int(st.st_size), int(st.st_mtime_ns)))
        except Exception:
            try:
                st = p.stat()
                items.append((".", int(st.st_size), int(st.st_mtime_ns)))
            except Exception:
                items.append((".", -1, -1))
        return ("dir", str(p), tuple(items))
    signature = file_signature_with_sidecars(p)
    if cache is not None:
        cache[cache_key] = (quick_sig, signature)
        max_items = int(getattr(parent, "_path_signature_cache_max", 128) or 128)
        while len(cache) > max_items:
            cache.popitem(last=False)
    return signature


def quick_path_signature(path: Path) -> tuple:
    if path.is_file():
        return file_signature_with_sidecars(path)
    try:
        st = path.stat()
        return (
            "dir" if path.is_dir() else "file",
            str(path),
            int(st.st_size),
            int(st.st_mtime_ns),
        )
    except Exception:
        return ("missing", str(path))


def file_signature_with_sidecars(path: Path) -> tuple:
    files = [path]
    cls_param = path.parent / f"{path.name}_param.txt"
    if cls_param.exists():
        files.append(cls_param)
    items = []
    for item in files:
        try:
            st = item.stat()
            items.append((item.name, int(st.st_size), int(st.st_mtime_ns)))
        except Exception:
            items.append((item.name, -1, -1))
    return ("file", str(path), tuple(items))


def entry_state_token(entry) -> str:
    try:
        payload = asdict(entry)
    except Exception:
        payload = getattr(entry, "__dict__", {})
    try:
        return json.dumps(freeze_cache_value(payload), sort_keys=True)
    except Exception:
        return repr(payload)
