"""Optional Materials Project import for theoretical band overlays."""
from __future__ import annotations

from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, TimeoutError as _FutTimeout
import json
import os

from .models import TheoryBandData, bandstructure_to_theory_data


# Mapping from MP crystal_system to the Bravais label used by physics/bz.Lattice3D.
_CRYSTAL_SYSTEM_TO_BRAVAIS: dict[str, str] = {
    "cubic": "cubic",
    "tetragonal": "tetragonal",
    "orthorhombic": "orthorhombic",
    "hexagonal": "hexagonal",
    "trigonal": "hexagonal",
    "monoclinic": "monoclinic",
    "triclinic": "triclinic",
}

DEFAULT_MP_TIMEOUT_S = 10.0


class MaterialsProjectUnavailable(RuntimeError):
    pass


def load_materials_project_band_data(
    material_id: str,
    *,
    api_key: str | None = None,
    cache_dir: str | Path | None = None,
    path_type: str = "setyawan_curtarolo",
    force_refresh: bool = False,
    with_projections: bool = False,
) -> TheoryBandData:
    """Fetch and cache a Materials Project band structure as overlay data.

    ``with_projections`` (opt-in): try to fetch orbital projections and infer
    per-band character. Uses a SEPARATE cache (``_proj`` suffix) so the legacy
    cache without projections is never polluted or overwritten.
    """
    mpid = str(material_id or "").strip()
    if not mpid:
        raise ValueError("Empty Materials Project ID.")
    cache_path = _cache_path(cache_dir, mpid, path_type,
                             with_projections=with_projections)
    cached: TheoryBandData | None = None
    if cache_path.exists() and not force_refresh:
        cached = TheoryBandData.from_dict(json.loads(cache_path.read_text()))
        if int(cached.schema_version) >= 3:
            return cached

    try:
        from mp_api.client import MPRester
    except Exception as exc:
        raise MaterialsProjectUnavailable(
            "mp-api unavailable. Install mp-api and define MP_API_KEY."
        ) from exc

    api_key = api_key or os.environ.get("MP_API_KEY") or None
    try:
        with MPRester(api_key) as mpr:
            bs = _get_bandstructure(mpr, mpid, path_type=path_type)
            formula = _get_formula(mpr, mpid)
            crystal_system = _get_crystal_system(mpr, mpid)
    except Exception as exc:
        if cached is not None:
            return cached
        raise RuntimeError(f"Materials Project import failed for {mpid}: {exc}") from exc

    data = bandstructure_to_theory_data(
        bs,
        material_id=mpid,
        formula=formula,
        crystal_system=crystal_system,
        source="materials_project",
        path_type=path_type,
        with_projections=with_projections,
    )
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(json.dumps(data.to_dict(), indent=2))
    return data


def search_by_formula(
    formula: str,
    *,
    api_key: str | None = None,
    max_results: int = 25,
) -> list[dict]:
    """Search Materials Project candidates by chemical formula.

    Return a list of dicts {material_id, formula_pretty, crystal_system,
    spacegroup_symbol, energy_above_hull, is_stable}. Used by the MP search
    dialog to propose an MPID when the user enters a formula.
    """
    formula = str(formula or "").strip()
    if not formula:
        raise ValueError("Empty chemical formula.")
    try:
        from mp_api.client import MPRester
    except Exception as exc:
        raise MaterialsProjectUnavailable(
            "mp-api unavailable. Install mp-api and define MP_API_KEY."
        ) from exc

    api_key = api_key or os.environ.get("MP_API_KEY") or None
    fields = ["material_id", "formula_pretty", "symmetry", "energy_above_hull", "is_stable"]
    try:
        with MPRester(api_key) as mpr:
            docs = mpr.materials.summary.search(formula=formula, fields=fields)
    except Exception as exc:
        raise RuntimeError(f"Materials Project search failed for '{formula}': {exc}") from exc

    out: list[dict] = []
    for d in docs[: int(max_results)]:
        sym = getattr(d, "symmetry", None)
        out.append({
            "material_id": str(getattr(d, "material_id", "") or ""),
            "formula_pretty": str(getattr(d, "formula_pretty", "") or ""),
            "crystal_system": str(getattr(sym, "crystal_system", "") or "") if sym else "",
            "spacegroup_symbol": str(getattr(sym, "symbol", "") or "") if sym else "",
            "energy_above_hull": float(getattr(d, "energy_above_hull", 0.0) or 0.0),
            "is_stable": bool(getattr(d, "is_stable", False)),
        })
    out.sort(key=lambda r: (not r["is_stable"], r["energy_above_hull"]))
    return out


def load_lattice(
    material_id: str,
    *,
    api_key: str | None = None,
    cache_dir: str | Path | None = None,
    force_refresh: bool = False,
    timeout_s: float = DEFAULT_MP_TIMEOUT_S,
):
    """Load lattice parameters for a Materials Project material.

    Return ``physics.bz.Lattice3D`` (imported lazily to avoid a cycle).

    - JSON disk cache: ``<cache_dir>/<mpid>_lattice.json`` (reuses
      default `cache_dir` ``.arpes_theory_cache``).
    - Hard timeout (``ThreadPoolExecutor`` + ``future.result(timeout)``):
      10 s by default. If MP times out, fall back to disk cache; otherwise raise
      ``MaterialsProjectUnavailable``.
    - Mapping from MP crystal_system to Lattice3D bravais (see table).

    Raises:
        ValueError: if ``material_id`` is empty.
        MaterialsProjectUnavailable: if mp-api is absent, timed out, or cache is empty.
    """
    from ..physics.bz import Lattice3D  # lazy import avoids a physics<->theory cycle

    mpid = str(material_id or "").strip()
    if not mpid:
        raise ValueError("Empty Materials Project ID.")

    cache_path = _lattice_cache_path(cache_dir, mpid)
    cached_dict: dict | None = None
    if cache_path.exists():
        try:
            cached_dict = json.loads(cache_path.read_text())
        except Exception:
            cached_dict = None  # unreadable cache, fall back to MP
        if cached_dict is not None and not force_refresh:
            return _lattice_from_dict(cached_dict, Lattice3D)

    try:
        from mp_api.client import MPRester
    except Exception as exc:
        if cached_dict is not None:
            return _lattice_from_dict(cached_dict, Lattice3D)
        raise MaterialsProjectUnavailable(
            "mp-api unavailable. Install mp-api and define MP_API_KEY."
        ) from exc

    api_key_resolved = api_key or os.environ.get("MP_API_KEY") or None

    def _fetch() -> dict:
        with MPRester(api_key_resolved) as mpr:
            structure = _get_structure(mpr, mpid)
            crystal_system = _get_crystal_system(mpr, mpid)
            space_group = _get_space_group(mpr, mpid)
        return _structure_to_dict(structure, mpid, crystal_system, space_group)

    try:
        with ThreadPoolExecutor(max_workers=1) as pool:
            data = pool.submit(_fetch).result(timeout=float(timeout_s))
    except _FutTimeout as exc:
        if cached_dict is not None:
            return _lattice_from_dict(cached_dict, Lattice3D)
        raise MaterialsProjectUnavailable(
            f"Materials Project: timeout exceeded ({timeout_s:.0f} s) for {mpid}."
        ) from exc
    except Exception as exc:
        if cached_dict is not None:
            return _lattice_from_dict(cached_dict, Lattice3D)
        raise MaterialsProjectUnavailable(
            f"Materials Project failed for {mpid}: {exc}"
        ) from exc

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(json.dumps(data, indent=2))
    return _lattice_from_dict(data, Lattice3D)


def _lattice_cache_path(cache_dir: str | Path | None, material_id: str) -> Path:
    root = Path(cache_dir) if cache_dir is not None else Path(".arpes_theory_cache")
    safe = material_id.replace("/", "_")
    return root / f"{safe}_lattice.json"


def _get_structure(mpr, material_id: str):
    if hasattr(mpr, "get_structure_by_material_id"):
        return mpr.get_structure_by_material_id(material_id)
    materials = getattr(mpr, "materials", None)
    if materials is not None and hasattr(materials, "get_structure_by_material_id"):
        return materials.get_structure_by_material_id(material_id)
    docs = mpr.materials.summary.search(material_ids=[material_id], fields=["structure"])
    if not docs:
        raise MaterialsProjectUnavailable(f"No structure for {material_id}.")
    return docs[0].structure


def _get_space_group(mpr, material_id: str) -> str:
    try:
        docs = mpr.materials.summary.search(material_ids=[material_id], fields=["symmetry"])
        if not docs:
            return ""
        sym = getattr(docs[0], "symmetry", None)
        if sym is None:
            return ""
        # symmetry.symbol = "I4/mmm", symmetry.number = 139
        sym_symbol = getattr(sym, "symbol", "") or ""
        sym_number = getattr(sym, "number", "") or ""
        if sym_number:
            return f"{sym_symbol} ({sym_number})" if sym_symbol else str(sym_number)
        return str(sym_symbol)
    except Exception:
        return ""


def _structure_to_dict(structure, mpid: str, crystal_system: str, space_group: str) -> dict:
    lat = structure.lattice
    cs_key = str(crystal_system or "").strip().lower()
    bravais = _CRYSTAL_SYSTEM_TO_BRAVAIS.get(cs_key, "tetragonal")
    return {
        "mp_id": str(mpid),
        "a": float(lat.a),
        "b": float(lat.b),
        "c": float(lat.c),
        "alpha_deg": float(lat.alpha),
        "beta_deg": float(lat.beta),
        "gamma_deg": float(lat.gamma),
        "bravais": bravais,
        "crystal_system": str(crystal_system or ""),
        "space_group": str(space_group or ""),
        "schema_version": 1,
    }


def _lattice_from_dict(data: dict, Lattice3D):
    """Construit Lattice3D depuis dict cache. Robuste aux champs manquants."""
    return Lattice3D(
        a=float(data.get("a", 1.0)),
        b=float(data.get("b", data.get("a", 1.0))),
        c=float(data.get("c", 1.0)),
        alpha_deg=float(data.get("alpha_deg", 90.0)),
        beta_deg=float(data.get("beta_deg", 90.0)),
        gamma_deg=float(data.get("gamma_deg", 90.0)),
        bravais=str(data.get("bravais", "tetragonal")),
        space_group=str(data.get("space_group", "")),
        mp_id=str(data.get("mp_id", "")),
    )


def _cache_path(
    cache_dir: str | Path | None,
    material_id: str,
    path_type: str,
    *,
    with_projections: bool = False,
) -> Path:
    root = Path(cache_dir) if cache_dir is not None else Path(".arpes_theory_cache")
    safe = material_id.replace("/", "_")
    suffix = "_proj" if with_projections else ""
    return root / f"{safe}_{path_type}{suffix}.json"


def _get_bandstructure(mpr, material_id: str, *, path_type: str):
    if hasattr(mpr, "get_bandstructure_by_material_id"):
        return mpr.get_bandstructure_by_material_id(material_id)
    materials = getattr(mpr, "materials", None)
    electronic = getattr(materials, "electronic_structure", None) if materials is not None else None
    band_route = getattr(electronic, "bandstructure", None) if electronic is not None else None
    if band_route is not None and hasattr(band_route, "get_bandstructure_from_material_id"):
        return band_route.get_bandstructure_from_material_id(material_id)
    raise MaterialsProjectUnavailable("Endpoint bandstructure Materials Project introuvable.")


def _get_formula(mpr, material_id: str) -> str:
    try:
        docs = mpr.materials.summary.search(
            material_ids=[material_id],
            fields=["formula_pretty"],
        )
        return str(docs[0].formula_pretty) if docs else ""
    except Exception:
        return ""


def _get_crystal_system(mpr, material_id: str) -> str:
    """MP crystal system (for example 'Tetragonal'). Empty string if unavailable/offline."""
    try:
        docs = mpr.materials.summary.search(
            material_ids=[material_id],
            fields=["symmetry"],
        )
        if not docs:
            return ""
        sym = getattr(docs[0], "symmetry", None)
        cs = getattr(sym, "crystal_system", "") if sym else ""
        return str(cs or "")
    except Exception:
        return ""
