"""Small on-disk artifact cache for loaded ARPES datasets."""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import threading
import warnings
from pathlib import Path
from typing import Any

import numpy as np


ARTIFACT_VERSION = 1
DEFAULT_QUOTA_MB = 250


def raw_artifact_root(path: str | Path, session_folder: str | Path | None = None) -> Path:
    if session_folder:
        return Path(session_folder) / ".arpes_cache" / "raw_artifacts"
    p = Path(path)
    base = p if p.is_dir() else p.parent
    return base / ".arpes_cache" / "raw_artifacts"


def raw_artifact_path(path: str | Path, cache_key: tuple, session_folder: str | Path | None = None) -> Path:
    digest = hashlib.sha256(_stable_json(cache_key).encode("utf-8")).hexdigest()
    return raw_artifact_root(path, session_folder) / f"raw_{digest[:32]}.npz"


def load_raw_artifact(path: str | Path, cache_key: tuple, session_folder: str | Path | None = None):
    cache_path = raw_artifact_path(path, cache_key, session_folder)
    if not cache_path.exists():
        return None
    try:
        with np.load(cache_path, allow_pickle=False) as npz:
            manifest = json.loads(str(npz["__manifest__"].item()))
            if manifest.get("version") != ARTIFACT_VERSION:
                return None
            if manifest.get("cache_key") != _jsonable(cache_key):
                return None
            arrays = {name: npz[name] for name in npz.files if name != "__manifest__"}
            data = _restore_payload(manifest["data"], arrays)
            offsets = _restore_payload(manifest.get("angle_offsets", {}), arrays)
            return data, offsets
    except Exception as exc:
        # Cache illisible → cache miss + signal au lieu d'avaler l'erreur.
        warnings.warn(
            f"load_raw_artifact: cache illisible ({exc}); re-calcul requis.",
            RuntimeWarning, stacklevel=2,
        )
        return None


def save_raw_artifact(
    path: str | Path,
    cache_key: tuple,
    data: dict,
    angle_offsets: dict | None = None,
    session_folder: str | Path | None = None,
) -> None:
    cache_path = raw_artifact_path(path, cache_key, session_folder)
    arrays: dict[str, np.ndarray] = {}
    manifest = {
        "version": ARTIFACT_VERSION,
        "cache_key": _jsonable(cache_key),
        "data": _freeze_payload(data, arrays, "data"),
        "angle_offsets": _freeze_payload(angle_offsets or {}, arrays, "angle_offsets"),
    }
    tmp_path = cache_path.with_suffix(cache_path.suffix + ".tmp")
    try:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(tmp_path, __manifest__=np.array(json.dumps(manifest)), **arrays)
        npz_tmp = tmp_path if tmp_path.exists() else tmp_path.with_suffix(tmp_path.suffix + ".npz")
        os.replace(npz_tmp, cache_path)
    except Exception as exc:
        # Sauvegarde échouée → données NON persistées : surface l'erreur
        # plutôt que perdre silencieusement.
        warnings.warn(
            f"save_raw_artifact: échec écriture {cache_path} ({exc}); "
            f"cache disque non mis à jour.",
            RuntimeWarning, stacklevel=2,
        )
        try:
            tmp_path.unlink(missing_ok=True)
            tmp_path.with_suffix(tmp_path.suffix + ".npz").unlink(missing_ok=True)
        except Exception:
            pass


def _stable_json(value: Any) -> str:
    return json.dumps(_jsonable(value), sort_keys=True, separators=(",", ":"))


def _jsonable(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in sorted(value.items(), key=lambda item: str(item[0]))}
    if isinstance(value, tuple):
        return [_jsonable(v) for v in value]
    if isinstance(value, list):
        return [_jsonable(v) for v in value]
    if isinstance(value, set):
        return sorted(_jsonable(v) for v in value)
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return repr(value)


def _freeze_payload(value: Any, arrays: dict[str, np.ndarray], prefix: str) -> Any:
    if isinstance(value, np.ndarray):
        name = f"arr_{len(arrays)}"
        arrays[name] = value
        return {"__ndarray__": name}
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): _freeze_payload(v, arrays, f"{prefix}.{k}") for k, v in value.items()}
    if isinstance(value, tuple):
        return {"__tuple__": [_freeze_payload(v, arrays, f"{prefix}.{i}") for i, v in enumerate(value)]}
    if isinstance(value, list):
        return [_freeze_payload(v, arrays, f"{prefix}.{i}") for i, v in enumerate(value)]
    try:
        json.dumps(value)
        return value
    except TypeError:
        raise TypeError(f"Unsupported cache payload value at {prefix}: {type(value).__name__}")


def _restore_payload(value: Any, arrays: dict[str, np.ndarray]) -> Any:
    if isinstance(value, dict):
        if "__ndarray__" in value:
            return arrays[value["__ndarray__"]]
        if "__tuple__" in value:
            return tuple(_restore_payload(v, arrays) for v in value["__tuple__"])
        return {k: _restore_payload(v, arrays) for k, v in value.items()}
    if isinstance(value, list):
        return [_restore_payload(v, arrays) for v in value]
    return value


def save_raw_artifact_async(
    path: str | Path,
    cache_key: tuple,
    data: dict,
    angle_offsets: dict | None = None,
    session_folder: str | Path | None = None,
    quota_mb: float = DEFAULT_QUOTA_MB,
) -> threading.Thread:
    """Écrit l'artefact en arrière-plan (daemon thread). Ne bloque pas l'UI.

    Effectue aussi un prune éventuel pour rester sous `quota_mb`.
    """
    def _worker():
        try:
            save_raw_artifact(path, cache_key, data, angle_offsets, session_folder)
            prune_cache_folder(session_folder if session_folder else Path(path).parent,
                               max_mb=quota_mb)
        except Exception as exc:
            warnings.warn(
                f"save_raw_artifact_async: thread cache crashé ({exc}); "
                f"sauvegarde disque non garantie.",
                RuntimeWarning, stacklevel=2,
            )

    t = threading.Thread(target=_worker, daemon=True)
    t.start()
    return t


def clear_cache_folder(session_folder: str | Path | None) -> tuple[int, int]:
    """Supprime `.arpes_cache/raw_artifacts/`. Retourne (n_files, total_bytes)."""
    root = raw_artifact_root(session_folder or Path.cwd(), session_folder)
    if not root.exists():
        return (0, 0)
    n = 0
    total = 0
    for f in root.glob("*.npz"):
        try:
            total += f.stat().st_size
        except OSError:
            pass
        n += 1
    try:
        shutil.rmtree(root, ignore_errors=True)
    except Exception:
        pass
    return (n, total)


def prune_cache_folder(session_folder: str | Path | None, *, max_mb: float = DEFAULT_QUOTA_MB) -> int:
    """Évince les artefacts les plus anciens si la taille totale > max_mb. Retourne n_evicted."""
    if session_folder is None:
        return 0
    root = raw_artifact_root(session_folder, session_folder)
    if not root.exists():
        return 0
    quota_bytes = int(float(max_mb) * 1024 * 1024)
    files: list[tuple[float, int, Path]] = []
    total = 0
    for f in root.glob("*.npz"):
        try:
            st = f.stat()
            files.append((st.st_mtime, st.st_size, f))
            total += st.st_size
        except OSError:
            continue
    if total <= quota_bytes:
        return 0
    files.sort()  # plus anciens d'abord
    evicted = 0
    for mtime, size, f in files:
        if total <= quota_bytes:
            break
        try:
            f.unlink()
            total -= size
            evicted += 1
        except OSError:
            continue
    return evicted


def cache_size_mb(session_folder: str | Path | None) -> float:
    if session_folder is None:
        return 0.0
    root = raw_artifact_root(session_folder, session_folder)
    if not root.exists():
        return 0.0
    total = 0
    for f in root.glob("*.npz"):
        try:
            total += f.stat().st_size
        except OSError:
            continue
    return total / (1024 * 1024)
