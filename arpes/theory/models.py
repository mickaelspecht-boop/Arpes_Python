"""Pure models/helpers for optional DFT band overlays."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any
import math
import re

import numpy as np


@dataclass(frozen=True)
class TheoryBandData:
    source: str
    material_id: str
    formula: str = ""
    efermi: float = 0.0
    k_distance: list[float] = field(default_factory=list)
    bands: list[list[float]] = field(default_factory=list)
    labels: list[dict[str, Any]] = field(default_factory=list)
    path_type: str = "setyawan_curtarolo"
    warning: str = ""
    # Schéma v2 : métadonnées par bande pour la liste cochable + caractère
    # orbital agrégé (projections MP opt-in). Champs OPTIONNELS : un cache
    # legacy (sans ces clés) reste chargeable, from_dict comble par défaut.
    schema_version: int = 2
    band_meta: list[dict[str, Any]] = field(default_factory=list)
    band_character: list[str] = field(default_factory=list)
    # Vrai chemin k parcouru par MP : liste ordonnée
    # {name, start, end} (indices dans k_distance). Vide pour DFT local
    # legacy → fallback ancien comportement par position de label.
    branches: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "source": self.source,
            "material_id": self.material_id,
            "formula": self.formula,
            "efermi": self.efermi,
            "k_distance": self.k_distance,
            "bands": self.bands,
            "labels": self.labels,
            "path_type": self.path_type,
            "warning": self.warning,
            "schema_version": self.schema_version,
            "band_meta": self.band_meta,
            "band_character": self.band_character,
            "branches": self.branches,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "TheoryBandData":
        data = data or {}
        return cls(
            source=str(data.get("source") or ""),
            material_id=str(data.get("material_id") or ""),
            formula=str(data.get("formula") or ""),
            efermi=_finite_float(data.get("efermi"), 0.0),
            k_distance=[float(x) for x in data.get("k_distance", [])],
            bands=[[float(v) for v in row] for row in data.get("bands", [])],
            labels=[{**dict(x), "label": _clean_label(x.get("label", ""))} for x in data.get("labels", [])],
            path_type=str(data.get("path_type") or "setyawan_curtarolo"),
            warning=str(data.get("warning") or ""),
            schema_version=int(_finite_float(data.get("schema_version"), 1)),
            band_meta=[dict(x) for x in (data.get("band_meta") or [])],
            band_character=[str(x) for x in (data.get("band_character") or [])],
            branches=[dict(x) for x in (data.get("branches") or [])],
        )


@dataclass(frozen=True)
class TheoryOverlayConfig:
    enabled: bool = False
    segment: str = ""
    energy_shift: float = 0.0
    k_shift: float = 0.0
    k_scale: float = 1.0
    alpha: float = 0.65
    max_bands: int = 10
    mirror_gamma: bool = False
    band_indices: str = ""
    # ef_window > 0 : ne garder/pré-cocher que les bandes croisant ±ef_window
    # autour de E_F (E=0, efermi déjà soustrait). 0 = désactivé.
    ef_window: float = 0.0
    color_by_band: bool = True

    def to_dict(self) -> dict[str, Any]:
        return {
            "enabled": bool(self.enabled),
            "segment": self.segment,
            "energy_shift": float(self.energy_shift),
            "k_shift": float(self.k_shift),
            "k_scale": float(self.k_scale),
            "alpha": float(self.alpha),
            "max_bands": int(self.max_bands),
            "mirror_gamma": bool(self.mirror_gamma),
            "band_indices": str(self.band_indices),
            "ef_window": float(self.ef_window),
            "color_by_band": bool(self.color_by_band),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "TheoryOverlayConfig":
        data = data or {}
        return cls(
            enabled=bool(data.get("enabled", False)),
            segment=str(data.get("segment") or ""),
            energy_shift=_finite_float(data.get("energy_shift"), 0.0),
            k_shift=_finite_float(data.get("k_shift"), 0.0),
            k_scale=_finite_float(data.get("k_scale"), 1.0) or 1.0,
            alpha=_finite_float(data.get("alpha"), 0.65),
            max_bands=max(1, int(_finite_float(data.get("max_bands"), 10))),
            mirror_gamma=bool(data.get("mirror_gamma", False)),
            band_indices=str(data.get("band_indices") or ""),
            ef_window=max(0.0, _finite_float(data.get("ef_window"), 0.0)),
            color_by_band=bool(data.get("color_by_band", True)),
        )


def _finite_float(value: Any, default: float = 0.0) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return float(default)
    return out if math.isfinite(out) else float(default)


def normalize_direction_label(value: Any) -> str:
    """Normalize common logbook direction spellings without changing meaning."""
    text = "" if value is None else str(value).strip()
    if not text:
        return ""
    text = re.sub(r"(?i)\bgamma\b", "Γ", text)
    text = re.sub(r"(?i)\bg(?=\s*(?:$|[-_/→> ]))", "Γ", text)
    text = text.replace("->", "-").replace("→", "-").replace("_", "-").replace("/", "-")
    text = re.sub(r"\s+", "", text)
    text = re.sub(r"-+", "-", text)
    if re.fullmatch(r"(?i)g[mkxy]", text):
        text = "Γ" + text[1:]
    if len(text) == 2 and text.startswith("Γ"):
        text = f"Γ-{text[1]}"
    return text.upper().replace("Γ", "Γ")


def branch_display_names(branches: list[dict[str, Any]] | None) -> list[str]:
    """Noms affichables des branches MP, ordre du chemin réel.

    Désambiguïse les répétitions (chemin repassant par Γ) en suffixant
    ``(2)``, ``(3)``… à partir de la 2ᵉ occurrence d'un même nom. Le
    résultat reste alignable 1:1 avec ``branches`` (même ordre/longueur).
    """
    out: list[str] = []
    counts: dict[str, int] = {}
    for br in branches or []:
        name = _clean_segment_name(str(br.get("name") or ""))
        if not name:
            name = "?"
        counts[name] = counts.get(name, 0) + 1
        out.append(name if counts[name] == 1 else f"{name} ({counts[name]})")
    return out


def _clean_segment_name(name: str) -> str:
    """`\\Gamma-X` / `GAMMA-X` → `Γ-X` (nettoie chaque extrémité)."""
    if "-" not in name:
        return _clean_label(name)
    a, _, b = name.partition("-")
    return f"{_clean_label(a)}-{_clean_label(b)}"


def _branch_index_for_segment(
    branches: list[dict[str, Any]], segment: str
) -> dict[str, Any] | None:
    """Retrouve la branche dont le nom affiché == segment choisi."""
    names = branch_display_names(branches)
    for disp, br in zip(names, branches):
        if disp == segment:
            return br
    # tolérance : segment sans suffixe → 1ʳᵉ occurrence du nom
    for disp, br in zip(names, branches):
        if disp.split(" (")[0] == segment:
            return br
    return None


def segment_from_direction(
    direction: str,
    labels: list[dict[str, Any]],
    branches: list[dict[str, Any]] | None = None,
) -> str:
    """Return a matching segment name, else ``""``.

    Si ``branches`` (vrai chemin MP) fourni, on cherche d'abord parmi
    elles (orientation indifférente) : c'est le chemin physiquement
    parcouru. Sinon repli sur les paires de labels consécutifs.
    """
    norm = normalize_direction_label(direction)
    if not norm or "-" not in norm:
        return ""
    a, b = [part.strip() for part in norm.split("-", 1)]
    if not a or not b:
        return ""
    if branches:
        disp_names = branch_display_names(branches)
        for disp in disp_names:
            base = disp.split(" (")[0]
            if "-" not in base:
                continue
            la, _, lb = base.partition("-")
            up = lambda s: s.upper().replace("GAMMA", "Γ")
            if (up(la), up(lb)) == (a, b) or (up(lb), up(la)) == (a, b):
                return disp
        return ""
    names = [str(item.get("label") or "") for item in labels]
    pairs = set()
    for left, right in zip(names, names[1:]):
        if left and right:
            pairs.add((left.upper().replace("GAMMA", "Γ"), right.upper().replace("GAMMA", "Γ")))
    if (a, b) in pairs:
        return f"{a}-{b}"
    if (b, a) in pairs:
        return f"{b}-{a}"
    return ""


def available_segments(
    labels: list[dict[str, Any]],
    branches: list[dict[str, Any]] | None = None,
) -> list[str]:
    """Segments DFT proposables dans le menu.

    Si ``branches`` (vrai chemin MP) présent : on ne propose QUE les
    branches réellement parcourues, dans l'ordre du chemin (doublons
    désambiguïsés). On ne fabrique plus de paires arbitraires non
    parcourues — qui produisaient un overlay incohérent avec MP.

    Fallback (DFT local sans branches) : ancienne énumération par
    paires de labels.
    """
    if branches:
        return branch_display_names(branches)
    names = [str(item.get("label") or "") for item in labels if item.get("label")]
    out: list[str] = []
    seen: set[str] = set()

    def push(seg: str) -> None:
        if seg and seg not in seen:
            seen.add(seg)
            out.append(seg)

    if "Γ" in names:
        for label in names:
            if label and label != "Γ":
                push(f"Γ-{label}")
    last = ""
    for label in names:
        if last and label:
            push(f"{last}-{label}")
        last = label
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            if a and b and a != b:
                push(f"{a}-{b}")
    return out


def parse_band_indices(spec: str, n_bands: int) -> list[int]:
    """Parse `'1,3,5-8'` → [1,3,5,6,7,8]. Skip out-of-range. Empty → []."""
    out: list[int] = []
    seen: set[int] = set()
    if not spec:
        return out
    for chunk in spec.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        if "-" in chunk:
            try:
                lo_s, hi_s = chunk.split("-", 1)
                lo, hi = int(lo_s), int(hi_s)
            except ValueError:
                continue
            if lo > hi:
                lo, hi = hi, lo
            for idx in range(lo, hi + 1):
                if 0 <= idx < n_bands and idx not in seen:
                    seen.add(idx)
                    out.append(idx)
        else:
            try:
                idx = int(chunk)
            except ValueError:
                continue
            if 0 <= idx < n_bands and idx not in seen:
                seen.add(idx)
                out.append(idx)
    return out


def select_bands_for_view(
    data: TheoryBandData | dict[str, Any],
    config: TheoryOverlayConfig | dict[str, Any],
    *,
    xlim: tuple[float, float],
    ylim: tuple[float, float],
) -> list[tuple[int, np.ndarray, np.ndarray]]:
    """Comme filter_bands_for_view mais expose l'index de bande source.

    Retourne ``[(band_index, k, band_energies), ...]``. L'index permet une
    couleur stable et une légende. Le miroir Γ réutilise l'index de la
    bande originale.
    """
    data = TheoryBandData.from_dict(data) if isinstance(data, dict) else data
    config = TheoryOverlayConfig.from_dict(config) if isinstance(config, dict) else config
    if not config.enabled or not data.k_distance or not data.bands:
        return []
    k_raw = np.asarray(data.k_distance, dtype=float)
    k = _branch_local_k(data, config, k_raw)
    k = k * float(config.k_scale) + float(config.k_shift)
    bands = np.asarray(data.bands, dtype=float) + float(config.energy_shift)
    if bands.ndim != 2 or bands.shape[1] != k.size:
        return []
    segment_mask = _segment_mask(data, config, k.size)
    x0, x1 = sorted((float(xlim[0]), float(xlim[1])))
    y0, y1 = sorted((float(ylim[0]), float(ylim[1])))
    mask_x = (k >= x0) & (k <= x1) & segment_mask
    if not mask_x.any():
        mask_x = segment_mask
    scored: list[tuple[float, int]] = []
    y_center = 0.5 * (y0 + y1)
    for idx, band in enumerate(bands):
        visible = band[mask_x]
        finite = visible[np.isfinite(visible)]
        if finite.size == 0:
            continue
        overlap = np.mean((finite >= y0) & (finite <= y1))
        distance = float(np.nanmin(np.abs(finite - y_center)))
        scored.append((-float(overlap), distance, idx))
    scored.sort()
    explicit = parse_band_indices(config.band_indices, len(bands))
    if explicit:
        selected = explicit
    else:
        selected = [idx for *_rest, idx in scored[: int(config.max_bands)]]
    # Filtre fenêtre E_F : ne garder que les bandes traversant ±ef_window
    # autour de E=0 (efermi déjà soustrait, energy_shift inclus dans bands).
    win = float(config.ef_window)
    if win > 0.0:
        kept: list[int] = []
        for idx in selected:
            finite = bands[idx][np.isfinite(bands[idx])]
            if finite.size and float(np.nanmin(finite)) <= win and float(np.nanmax(finite)) >= -win:
                kept.append(idx)
        selected = kept
    # Sélection appliquée AVANT le miroir Γ (cf. arpes-redteam).
    curves: list[tuple[int, np.ndarray, np.ndarray]] = []
    for idx in selected:
        band = bands[idx].copy()
        band[~segment_mask] = np.nan
        curves.append((idx, k, band))
        if config.mirror_gamma:
            curves.append((idx, -k, band.copy()))
    return curves


def filter_bands_for_view(
    data: TheoryBandData | dict[str, Any],
    config: TheoryOverlayConfig | dict[str, Any],
    *,
    xlim: tuple[float, float],
    ylim: tuple[float, float],
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Compat : ``[(k, band), ...]`` sans index (anciens appelants/tests)."""
    return [(k, band) for _idx, k, band in select_bands_for_view(
        data, config, xlim=xlim, ylim=ylim)]


def compare_fit_to_theory(
    data: TheoryBandData | dict[str, Any],
    config: TheoryOverlayConfig | dict[str, Any],
    fit_result: dict[str, Any] | None,
    *,
    max_results: int = 6,
    min_points: int = 3,
) -> list[dict[str, Any]]:
    """Score fitted experimental kF branches against DFT bands.

    The score is a vertical residual in displayed coordinates: for each fitted
    point ``(k_exp, E_exp)``, interpolate the DFT band ``E_DFT(k_exp)`` and
    compute RMS(``E_exp - E_DFT``). This is a diagnostic, not an automatic
    physical assignment.
    """
    data = TheoryBandData.from_dict(data) if isinstance(data, dict) else data
    config = TheoryOverlayConfig.from_dict(config) if isinstance(config, dict) else config
    fr = fit_result or {}
    e_exp = np.asarray(fr.get("e_fitted", []), dtype=float)
    if e_exp.size == 0 or not data.k_distance or not data.bands:
        return []
    k_dft = np.asarray(data.k_distance, dtype=float) * float(config.k_scale) + float(config.k_shift)
    bands = np.asarray(data.bands, dtype=float) + float(config.energy_shift)
    if bands.ndim != 2 or bands.shape[1] != k_dft.size:
        return []
    segment_mask = _segment_mask(data, config, k_dft.size)
    order = np.argsort(k_dft)
    k_sorted = k_dft[order]
    valid_segment_sorted = segment_mask[order]
    out: list[dict[str, Any]] = []
    for branch_name in ("kF_minus", "kF_plus"):
        branches = fr.get(branch_name) or []
        for pair_index, k_branch_raw in enumerate(branches):
            k_exp = np.asarray(k_branch_raw, dtype=float)
            n = min(k_exp.size, e_exp.size)
            if n == 0:
                continue
            k_exp_n = k_exp[:n]
            e_exp_n = e_exp[:n]
            valid_exp = np.isfinite(k_exp_n) & np.isfinite(e_exp_n)
            if int(valid_exp.sum()) < int(min_points):
                continue
            for band_index, band in enumerate(bands):
                band_sorted = np.asarray(band, dtype=float)[order]
                valid_band = valid_segment_sorted & np.isfinite(k_sorted) & np.isfinite(band_sorted)
                if int(valid_band.sum()) < 2:
                    continue
                k_ref = k_sorted[valid_band]
                e_ref = band_sorted[valid_band]
                lo, hi = float(np.nanmin(k_ref)), float(np.nanmax(k_ref))
                valid = valid_exp & (k_exp_n >= lo) & (k_exp_n <= hi)
                if int(valid.sum()) < int(min_points):
                    continue
                e_interp = np.interp(k_exp_n[valid], k_ref, e_ref)
                residual = e_exp_n[valid] - e_interp
                rms_e = float(np.sqrt(np.nanmean(residual**2)))
                med_e = float(np.nanmedian(residual))
                out.append({
                    "branch": branch_name,
                    "pair_index": int(pair_index),
                    "band_index": int(band_index),
                    "n_points": int(valid.sum()),
                    "rms_e": rms_e,
                    "median_e": med_e,
                })
    out.sort(key=lambda item: (item["rms_e"], -item["n_points"]))
    return out[: int(max_results)]


def _branch_local_k(
    data: TheoryBandData, config: TheoryOverlayConfig, k_raw: np.ndarray
) -> np.ndarray:
    """Coordonnée k LOCALE à la branche choisie : Γ→0, bord de zone→1
    (en π/a, l'utilisateur applique k_scale). Évite que tout le chemin
    multi-branches reste tassé sur [-1,1] (overlay illisible).

    Sans branches MP ou sans segment : renvoie l'axe global inchangé
    (comportement legacy préservé).
    """
    if not data.branches or not config.segment:
        return k_raw
    br = _branch_index_for_segment(data.branches, config.segment)
    if br is None:
        return k_raw
    try:
        s = int(br.get("start", 0))
        e = int(br.get("end", k_raw.size - 1))
    except (TypeError, ValueError):
        return k_raw
    s, e = max(0, min(s, e)), min(k_raw.size - 1, max(s, e))
    span = max(e - s, 1)
    loc = np.full(k_raw.size, np.nan, dtype=float)
    frac = (np.arange(s, e + 1, dtype=float) - s) / span  # 0..1
    name = _clean_segment_name(str(br.get("name", "")))
    left, _, right = name.partition("-")
    # Γ placé à 0 : si Γ est l'extrémité finale de la branche, on inverse.
    if right.strip() == "Γ" and left.strip() != "Γ":
        frac = 1.0 - frac
    loc[s:e + 1] = frac
    return loc


def _segment_mask(data: TheoryBandData, config: TheoryOverlayConfig, n_k: int) -> np.ndarray:
    if not config.segment:
        return np.ones(n_k, dtype=bool)
    # Chemin MP réel : masque = plage d'indices [start, end] de la branche
    # choisie. Exact, indépendant du rescale de l'axe k (corrige
    # l'incohérence avec les graphes MP en ligne).
    if data.branches:
        br = _branch_index_for_segment(data.branches, config.segment)
        if br is not None:
            try:
                s = int(br.get("start", 0))
                e = int(br.get("end", n_k - 1))
            except (TypeError, ValueError):
                s, e = 0, n_k - 1
            s, e = max(0, min(s, e)), min(n_k - 1, max(s, e))
            mask = np.zeros(n_k, dtype=bool)
            mask[s:e + 1] = True
            return mask
        return np.ones(n_k, dtype=bool)
    if "-" not in config.segment:
        return np.ones(n_k, dtype=bool)
    left, right = [
        x.strip().upper().replace("GAMMA", "Γ")
        for x in config.segment.split("-", 1)
    ]
    label_positions = {
        str(item.get("label") or "").upper().replace("GAMMA", "Γ"): item.get("k")
        for item in data.labels
    }
    if left not in label_positions or right not in label_positions:
        return np.ones(n_k, dtype=bool)
    lo = _finite_float(label_positions[left], float("nan"))
    hi = _finite_float(label_positions[right], float("nan"))
    if not np.isfinite(lo) or not np.isfinite(hi):
        return np.ones(n_k, dtype=bool)
    a, b = sorted((lo, hi))
    raw_k = np.asarray(data.k_distance, dtype=float)
    return (raw_k >= a) & (raw_k <= b)


def bandstructure_to_theory_data(
    bandstructure: Any,
    *,
    material_id: str,
    formula: str = "",
    source: str = "materials_project",
    path_type: str = "setyawan_curtarolo",
    with_projections: bool = False,
) -> TheoryBandData:
    """Convert a pymatgen-like band structure object to JSON-safe arrays."""
    efermi = _finite_float(getattr(bandstructure, "efermi", 0.0), 0.0)
    bands_obj = getattr(bandstructure, "bands", None)
    if isinstance(bands_obj, dict):
        first = next(iter(bands_obj.values()), [])
        bands = np.asarray(first, dtype=float) - efermi
    else:
        bands = np.asarray(bands_obj, dtype=float) - efermi
    if bands.ndim != 2:
        raise ValueError("Band structure invalide: bandes DFT non matricielles.")

    k_distance = _k_distance_from_bandstructure(bandstructure, bands.shape[1])
    labels = _labels_from_bandstructure(bandstructure, k_distance)
    bands_list = bands.astype(float).tolist()
    from .band_select import aggregate_projection_character, compute_band_meta
    band_meta = compute_band_meta(bands_list)
    band_character: list[str] = []
    if with_projections:
        try:
            band_character = aggregate_projection_character(
                getattr(bandstructure, "projections", None),
                _structure_elements(bandstructure),
            )
        except Exception:
            band_character = []  # dégradation gracieuse (arpes-redteam)
    branches = _branches_from_bandstructure(bandstructure, bands.shape[1])
    return TheoryBandData(
        source=source,
        material_id=material_id,
        formula=formula,
        efermi=efermi,
        k_distance=[float(x) for x in k_distance],
        bands=bands_list,
        labels=labels,
        path_type=path_type,
        band_meta=band_meta,
        band_character=band_character,
        branches=branches,
    )


def _branches_from_bandstructure(bandstructure: Any, n_k: int) -> list[dict[str, Any]]:
    """Vrai chemin MP : pymatgen .branches = [{name,start_index,end_index}].

    Renvoie [{name, start, end}] borné à n_k. Vide si absent (DFT local)
    → fallback ancien comportement (arpes-redteam : cache legacy OK).
    """
    raw = getattr(bandstructure, "branches", None)
    if not raw:
        return []
    out: list[dict[str, Any]] = []
    for br in raw:
        try:
            name = str(br.get("name", "") if isinstance(br, dict) else getattr(br, "name", ""))
            s = int(br.get("start_index") if isinstance(br, dict) else getattr(br, "start_index"))
            e = int(br.get("end_index") if isinstance(br, dict) else getattr(br, "end_index"))
        except (TypeError, ValueError, AttributeError):
            continue
        s = max(0, min(s, n_k - 1))
        e = max(0, min(e, n_k - 1))
        if e < s:
            s, e = e, s
        out.append({"name": name, "start": s, "end": e})
    return out


def _structure_elements(bandstructure: Any) -> list[str]:
    """Symbole chimique par ion depuis bs.structure (ordre des sites)."""
    struct = getattr(bandstructure, "structure", None)
    if struct is None:
        return []
    out: list[str] = []
    try:
        for site in struct:
            sp = getattr(site, "specie", None) or getattr(site, "species", None)
            sym = getattr(sp, "symbol", None)
            out.append(str(sym) if sym else str(sp))
    except Exception:
        return []
    return out


def _k_distance_from_bandstructure(bandstructure: Any, n_k: int) -> np.ndarray:
    dist = getattr(bandstructure, "distance", None)
    if dist is not None:
        arr = np.asarray(dist, dtype=float)
        if arr.size == n_k and np.all(np.isfinite(arr)):
            return _scaled_k_axis(arr)
    kpoints = getattr(bandstructure, "kpoints", None) or []
    coords = []
    for kp in kpoints:
        frac = getattr(kp, "frac_coords", None)
        coords.append(np.asarray(frac if frac is not None else kp, dtype=float))
    if len(coords) == n_k:
        out = [0.0]
        for prev, cur in zip(coords, coords[1:]):
            out.append(out[-1] + float(np.linalg.norm(cur - prev)))
        return _scaled_k_axis(np.asarray(out, dtype=float))
    return np.linspace(-1.0, 1.0, n_k)


def _scaled_k_axis(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    if values.size < 2:
        return values
    lo = float(np.nanmin(values))
    hi = float(np.nanmax(values))
    if not np.isfinite(hi - lo) or abs(hi - lo) <= 1e-12:
        return np.linspace(-1.0, 1.0, values.size)
    centered = values - 0.5 * (lo + hi)
    half = max(abs(float(np.nanmin(centered))), abs(float(np.nanmax(centered))), 1e-12)
    return centered / half


def _labels_from_bandstructure(bandstructure: Any, k_distance: np.ndarray) -> list[dict[str, Any]]:
    labels: list[dict[str, Any]] = []
    raw = getattr(bandstructure, "labels_dict", None) or {}
    if isinstance(raw, dict):
        for label, kp in raw.items():
            try:
                coord = np.asarray(getattr(kp, "frac_coords", kp), dtype=float)
                kpoints = getattr(bandstructure, "kpoints", None) or []
                idx = _nearest_kpoint_index(kpoints, coord)
                x = float(k_distance[idx]) if idx is not None else None
            except Exception:
                x = None
            labels.append({"label": _clean_label(label), "k": x})
    labels.sort(key=lambda item: float(item["k"]) if item.get("k") is not None else 1e9)
    return labels


def _nearest_kpoint_index(kpoints: list[Any], coord: np.ndarray) -> int | None:
    if not kpoints:
        return None
    distances = []
    for kp in kpoints:
        frac = np.asarray(getattr(kp, "frac_coords", kp), dtype=float)
        distances.append(float(np.linalg.norm(frac - coord)))
    return int(np.argmin(distances))


_GREEK_MAP = {
    "GAMMA": "Γ", "SIGMA": "Σ", "DELTA": "Δ", "LAMBDA": "Λ",
    "PI": "Π", "OMEGA": "Ω", "PHI": "Φ", "THETA": "Θ", "EPSILON": "Ε",
}

_SUBSCRIPT_MAP = str.maketrans("0123456789+-", "₀₁₂₃₄₅₆₇₈₉₊₋")


def _clean_label(label: Any) -> str:
    """Convertit labels pymatgen (`\\Sigma_1`, `\\Gamma`, etc.) en Unicode.

    Gere : `G`, `GAMMA`, `\\GAMMA` -> `Γ` ; `\\Sigma_1`, `\\Sigma_{1}` -> `Σ₁` ;
    `Y_1` -> `Y₁`. Caractères grecs latex courants traduits via `_GREEK_MAP`.
    """
    text = str(label).strip()
    if not text:
        return ""
    # Cas legacy
    if text.upper() in {"G", "GAMMA", "\\GAMMA"}:
        return "Γ"
    # Strip backslashes leading
    raw = text.lstrip("\\")
    # Split base / subscript
    base, _, sub = raw.partition("_")
    sub = sub.strip("{}")
    # Greek replacement on base
    base_upper = base.upper()
    base_clean = _GREEK_MAP.get(base_upper, base)
    if base_clean == base and len(base_upper) == 1:
        # Lettre simple (X, Y, M, N, P, ...). Garde majuscule.
        base_clean = base_upper
    # Subscript -> unicode
    if sub:
        try:
            sub_clean = sub.translate(_SUBSCRIPT_MAP)
        except Exception:
            sub_clean = sub
        return f"{base_clean}{sub_clean}"
    return base_clean
