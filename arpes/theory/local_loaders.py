"""Load local DFT band files into the common theory overlay model."""
from __future__ import annotations

import ast
import json
from pathlib import Path
from typing import Any

import numpy as np

from arpes.theory.models import TheoryBandData, bandstructure_to_theory_data


def load_local_band_data(path: str | Path) -> TheoryBandData:
    """Dispatch local DFT import by file extension."""
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix in {".yaml", ".yml", ".json"}:
        return load_yaml_bands(path)
    if suffix == ".xml":
        return load_vasprun(path)
    if suffix in {".dat", ".txt"}:
        return load_qe_bands(path)
    raise ValueError(f"Unsupported local DFT format: {suffix or path.name}")


def load_vasprun(path: str | Path) -> TheoryBandData:
    """Load a VASP ``vasprun.xml`` through pymatgen when available."""
    path = Path(path)
    try:
        from pymatgen.io.vasp.outputs import Vasprun
    except Exception as exc:  # pragma: no cover - depends on optional install
        raise RuntimeError("VASP import unavailable: pymatgen is required.") from exc

    try:
        bandstructure = Vasprun(str(path), parse_projected_eigen=False).get_band_structure(line_mode=True)
    except Exception as exc:
        raise ValueError(f"Cannot read vasprun.xml: {exc}") from exc
    return bandstructure_to_theory_data(
        bandstructure,
        material_id=f"local:{path.stem}",
        source="local_vasp",
        path_type="local_vasp",
    )


def load_qe_bands(path: str | Path) -> TheoryBandData:
    """Load a simple Quantum Espresso bands text table.

    Supported rows are either ``k E1 E2 ...`` or ``kx ky kz E1 E2 ...``.
    QE-specific headers are ignored when they do not parse as floats.
    """
    path = Path(path)
    rows: list[list[float]] = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith(("#", "!", "@")):
            continue
        try:
            rows.append([float(part) for part in line.replace(",", " ").split()])
        except ValueError:
            continue
    if not rows:
        raise ValueError("QE file is empty or has no numeric rows.")
    width = len(rows[0])
    if width < 2 or any(len(row) != width for row in rows):
        raise ValueError("Invalid QE table: incompatible row widths.")

    arr = np.asarray(rows, dtype=float)
    if width >= 5 and _looks_like_fractional_kpoints(arr[:, :3]):
        k_distance = _cumulative_distance(arr[:, :3])
        energies = arr[:, 3:]
    else:
        k_distance = arr[:, 0]
        energies = arr[:, 1:]
    if energies.shape[1] == 0:
        raise ValueError("Invalid QE table: no band detected.")
    return TheoryBandData(
        source="local_qe",
        material_id=f"local:{path.stem}",
        k_distance=[float(x) for x in k_distance],
        bands=energies.T.astype(float).tolist(),
        labels=[],
        path_type="local_qe",
    )


def load_yaml_bands(path: str | Path) -> TheoryBandData:
    """Load the minimal custom YAML/JSON schema used by the UI importer."""
    path = Path(path)
    data = _read_mapping(path)
    efermi = _as_float(data.get("efermi", 0.0), "efermi")
    bands = np.asarray(data.get("bands"), dtype=float)
    if bands.ndim != 2 or bands.shape[0] == 0 or bands.shape[1] == 0:
        raise ValueError("Invalid local DFT schema: 'bands' must be a band x k-point matrix.")
    k_distance = _k_axis_from_mapping(data, bands.shape[1])
    labels = _labels_from_mapping(data.get("labels"), k_distance)
    return TheoryBandData(
        source="local_yaml",
        material_id=str(data.get("material_id") or f"local:{path.stem}"),
        formula=str(data.get("formula") or ""),
        efermi=efermi,
        k_distance=[float(x) for x in k_distance],
        bands=(bands - efermi).astype(float).tolist(),
        labels=labels,
        path_type=str(data.get("path_type") or "local"),
        warning=str(data.get("warning") or ""),
    )


def _read_mapping(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8")
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        try:
            import yaml
        except Exception:
            try:
                data = ast.literal_eval(text)
            except Exception as exc:
                raise RuntimeError("Cannot read YAML: install PyYAML or provide JSON.") from exc
        else:
            data = yaml.safe_load(text)
    if not isinstance(data, dict):
        raise ValueError("Invalid local DFT schema: root must be a mapping.")
    return data


def _k_axis_from_mapping(data: dict[str, Any], n_k: int) -> np.ndarray:
    if "k_distance" in data:
        k_distance = np.asarray(data["k_distance"], dtype=float)
    elif "kpoints" in data:
        kpoints = np.asarray(data["kpoints"], dtype=float)
        if kpoints.ndim != 2:
            raise ValueError("Invalid local DFT schema: 'kpoints' must be a matrix.")
        k_distance = _cumulative_distance(kpoints)
    else:
        k_distance = np.linspace(0.0, 1.0, n_k)
    if k_distance.size != n_k:
        raise ValueError("Invalid local DFT schema: k-axis length is incompatible with bands.")
    if not np.all(np.isfinite(k_distance)):
        raise ValueError("Invalid local DFT schema: k axis is not finite.")
    return k_distance.astype(float)


def _labels_from_mapping(raw: Any, k_distance: np.ndarray) -> list[dict[str, Any]]:
    if not raw:
        return []
    labels: list[dict[str, Any]] = []
    if isinstance(raw, dict):
        iterator = ({"label": label, "k": k} for label, k in raw.items())
    elif isinstance(raw, list):
        iterator = raw
    else:
        raise ValueError("Invalid local DFT schema: 'labels' must be a mapping or a list.")
    for item in iterator:
        if not isinstance(item, dict):
            raise ValueError("Invalid local DFT schema: label is not structured.")
        label = _clean_label(item.get("label", ""))
        if not label:
            continue
        pos = item.get("k", item.get("position", item.get("index")))
        if "index" in item and "k" not in item and "position" not in item:
            idx = int(pos)
            if idx < 0 or idx >= k_distance.size:
                raise ValueError(f"Label {label}: index outside k axis.")
            pos = float(k_distance[idx])
        labels.append({"label": label, "k": _as_float(pos, f"label {label}")})
    labels.sort(key=lambda item: float(item["k"]))
    return labels


def _cumulative_distance(kpoints: np.ndarray) -> np.ndarray:
    coords = np.asarray(kpoints, dtype=float)
    if coords.shape[0] == 0:
        return np.asarray([], dtype=float)
    out = [0.0]
    for prev, cur in zip(coords, coords[1:]):
        out.append(out[-1] + float(np.linalg.norm(cur - prev)))
    return np.asarray(out, dtype=float)


def _looks_like_fractional_kpoints(values: np.ndarray) -> bool:
    if values.ndim != 2 or values.shape[1] != 3:
        return False
    return bool(np.all(np.isfinite(values)) and np.nanmax(np.abs(values)) <= 2.0)


def _as_float(value: Any, field: str) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Non-numeric local DFT field: {field}") from exc
    if not np.isfinite(out):
        raise ValueError(f"Non-finite local DFT field: {field}")
    return out


def _clean_label(label: Any) -> str:
    text = str(label).strip()
    if text.upper() in {"G", "GAMMA", "\\GAMMA"}:
        return "Γ"
    return text.lstrip("\\")
