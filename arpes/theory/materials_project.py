"""Optional Materials Project import for theoretical band overlays."""
from __future__ import annotations

from pathlib import Path
import json
import os

from .models import TheoryBandData, bandstructure_to_theory_data


class MaterialsProjectUnavailable(RuntimeError):
    pass


def load_materials_project_band_data(
    material_id: str,
    *,
    api_key: str | None = None,
    cache_dir: str | Path | None = None,
    path_type: str = "setyawan_curtarolo",
    force_refresh: bool = False,
) -> TheoryBandData:
    """Fetch and cache a Materials Project band structure as overlay data."""
    mpid = str(material_id or "").strip()
    if not mpid:
        raise ValueError("Materials Project ID vide.")
    cache_path = _cache_path(cache_dir, mpid, path_type)
    if cache_path.exists() and not force_refresh:
        return TheoryBandData.from_dict(json.loads(cache_path.read_text()))

    try:
        from mp_api.client import MPRester
    except Exception as exc:
        raise MaterialsProjectUnavailable(
            "mp-api indisponible. Installer mp-api et définir MP_API_KEY."
        ) from exc

    api_key = api_key or os.environ.get("MP_API_KEY") or None
    try:
        with MPRester(api_key) as mpr:
            bs = _get_bandstructure(mpr, mpid, path_type=path_type)
            formula = _get_formula(mpr, mpid)
    except Exception as exc:
        raise RuntimeError(f"Import Materials Project échoué pour {mpid}: {exc}") from exc

    data = bandstructure_to_theory_data(
        bs,
        material_id=mpid,
        formula=formula,
        source="materials_project",
        path_type=path_type,
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
    """Recherche les candidats Materials Project par formule chimique.

    Retourne une liste de dicts {material_id, formula_pretty, crystal_system,
    spacegroup_symbol, energy_above_hull, is_stable}. Utilisé par le dialog
    MP search pour proposer un MPID quand l'utilisateur tape une formule.
    """
    formula = str(formula or "").strip()
    if not formula:
        raise ValueError("Formule chimique vide.")
    try:
        from mp_api.client import MPRester
    except Exception as exc:
        raise MaterialsProjectUnavailable(
            "mp-api indisponible. Installer mp-api et définir MP_API_KEY."
        ) from exc

    api_key = api_key or os.environ.get("MP_API_KEY") or None
    fields = ["material_id", "formula_pretty", "symmetry", "energy_above_hull", "is_stable"]
    try:
        with MPRester(api_key) as mpr:
            docs = mpr.materials.summary.search(formula=formula, fields=fields)
    except Exception as exc:
        raise RuntimeError(f"Recherche Materials Project échouée pour '{formula}': {exc}") from exc

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


def _cache_path(cache_dir: str | Path | None, material_id: str, path_type: str) -> Path:
    root = Path(cache_dir) if cache_dir is not None else Path(".arpes_theory_cache")
    safe = material_id.replace("/", "_")
    return root / f"{safe}_{path_type}.json"


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
