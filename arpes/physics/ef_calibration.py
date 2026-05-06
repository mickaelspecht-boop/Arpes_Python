"""Logique de calibration EF — sans PyQt.

Extraite de `arpes_explorer.py`. La couche UI conserve les `QMessageBox`,
les blocs `blockSignals` et l'enchaînement save/reload — toutes les
décisions scientifiques (choix de mode, calcul d'offset, correction
per-file) vivent ici.

Conventions :
- ``ef_offset`` est l'offset scalaire appliqué à l'axe énergie pour
  ramener EF à 0 (mode scalar) ou est laissé à 0 quand un polynôme par
  colonne porte tout le décalage (mode poly) ;
- ``ef_correction`` est un dict décrivant la correction (poly_coefs,
  source, etc.) ; vide en mode scalar pur ;
- ``ref_payload`` est ce qu'on stocke dans ``session.ef_reference`` pour
  pouvoir l'appliquer ensuite à d'autres fichiers du dossier.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def axis_zero_in_kinetic(meta: dict | None) -> float | None:
    """Renvoie l'énergie cinétique correspondant à E−EF=0 AVANT
    application de ``ef_offset`` (cf. doc historique dans `arpes_explorer`).
    """
    if not meta:
        return None
    for key in ("ef_kinetic_nominal", "ef_kinetic_from_hv"):
        v = meta.get(key)
        try:
            vf = float(v) if v is not None else None
        except (TypeError, ValueError):
            continue
        if vf is not None and np.isfinite(vf):
            return vf
    return None


@dataclass
class CalibrationUpdate:
    """Résultat de l'application d'une calibration EF à une entrée."""

    new_ef_offset: float
    ef_correction: dict
    ref_payload: dict
    msg: str


def compute_calibration_update(
    payload: dict,
    *,
    current_ef_offset: float,
    source_meta: dict | None,
    source_path: str,
) -> CalibrationUpdate:
    """Calcule la mise à jour de session pour le résultat d'une calibration EF.

    ``payload`` est le dict produit par `EFCalibrationDialog` (mode scalar
    ou poly). ``current_ef_offset`` est la valeur courante de la spinbox
    `sp_ef`. ``source_meta`` est `_raw_data["metadata"]` du fichier en
    cours de calibration.
    """
    mode = payload["mode"]
    if mode == "scalar":
        new_off = float(current_ef_offset) - float(payload["ef_shift"])
        ef_correction: dict = {}
        ref_payload = {
            "mode": "scalar",
            "ef_shift": float(payload["ef_shift"]),
            "T": float(payload["T"]),
            "fwhm_res": float(payload["fwhm_res"]),
            "source_file": str(source_path),
            "source_ef_kin_nominal": axis_zero_in_kinetic(source_meta),
            "source_energy_reference": str((source_meta or {}).get("energy_reference") or ""),
        }
        msg = (
            f"EF scalaire : Δ={payload['ef_shift'] * 1000:+.1f} meV → "
            f"offset={new_off:.4f} eV"
        )
        return CalibrationUpdate(new_ef_offset=new_off, ef_correction=ef_correction,
                                 ref_payload=ref_payload, msg=msg)

    # mode poly
    new_off = 0.0
    ef_correction = {
        "mode": "poly",
        "poly_coefs": [float(c) for c in payload["poly_coefs"]],
        "k_min": float(payload["k_min"]),
        "k_max": float(payload["k_max"]),
        "T": float(payload["T"]),
        "fwhm_res": float(payload["fwhm_res"]),
        "rms": float(payload["rms"]),
        "n_valid": int(payload["n_valid"]),
        "source": "self",
        "source_file": str(source_path),
    }
    ref_payload = dict(ef_correction)
    msg = (
        f"EF par colonne : {payload['n_valid']} k valides, "
        f"FWHM≈{payload['fwhm_res'] * 1000:.0f} meV, "
        f"rms={payload['rms'] * 1000:.1f} meV"
    )
    return CalibrationUpdate(new_ef_offset=new_off, ef_correction=ef_correction,
                             ref_payload=ref_payload, msg=msg)


@dataclass
class ReferenceApplication:
    """Résultat de l'application d'une référence EF de session à un fichier."""

    new_ef_offset: float
    ef_correction: dict
    msg: str


class ReferenceError(ValueError):
    """Référence EF mal formée (mode inconnu, manquant, etc.)."""


def already_applied(ef_correction: dict) -> bool:
    """Détecte qu'une référence EF a déjà été appliquée à ce fichier
    (gardefou contre la double-soustraction du shift scalaire)."""
    src = (ef_correction or {}).get("source")
    return src in ("reference", "reference_scalar")


def apply_reference_to_target(
    ref: dict,
    *,
    current_ef_offset: float,
    target_meta: dict | None,
    ref_path_str: str,
) -> ReferenceApplication:
    """Applique ``ref`` (session.ef_reference) au fichier cible courant.

    Lève ``ReferenceError`` si le mode n'est pas reconnu.
    """
    mode = ref.get("mode")
    if mode == "poly":
        ef_correction = dict(ref)
        ef_correction["source"] = "reference"
        msg = f"Référence EF poly appliquée (source: {ref_path_str})"
        return ReferenceApplication(new_ef_offset=0.0, ef_correction=ef_correction, msg=msg)

    if mode == "scalar":
        base_shift = float(ref.get("ef_shift", 0.0))
        # Quand chaque fichier a sa propre Center Energy (BESSY) ou son propre
        # hν−φ (Solaris/CLS), un shift scalaire seul est invalide : il faut
        # compenser la différence d'origine cinétique entre source et cible.
        tgt_meta = target_meta or {}
        tgt_ef_kin = axis_zero_in_kinetic(tgt_meta)
        tgt_ref_mode = str(tgt_meta.get("energy_reference") or "")
        src_ef_kin = ref.get("source_ef_kin_nominal")
        src_ref_mode = str(ref.get("source_energy_reference") or "")
        try:
            src_ef_kin = float(src_ef_kin) if src_ef_kin is not None else None
        except (TypeError, ValueError):
            src_ef_kin = None
        kinetic_correction = 0.0
        applied_per_file = False
        modes_match = bool(src_ref_mode) and bool(tgt_ref_mode) and src_ref_mode == tgt_ref_mode
        if src_ef_kin is not None and tgt_ef_kin is not None and modes_match:
            kinetic_correction = src_ef_kin - tgt_ef_kin
            applied_per_file = abs(kinetic_correction) > 1e-6
        effective_shift = base_shift + kinetic_correction
        new_off = float(current_ef_offset) - effective_shift
        ef_correction = {
            "source": "reference_scalar",
            "ref_shift": base_shift,
            "ref_kinetic_correction": float(kinetic_correction),
            "ref_effective_shift": float(effective_shift),
            "ref_source_file": ref.get("source_file", ""),
            "ref_source_ef_kin_nominal": float(src_ef_kin) if src_ef_kin is not None else None,
            "ref_target_ef_kin_nominal": float(tgt_ef_kin) if tgt_ef_kin is not None else None,
        }
        if applied_per_file:
            msg = (
                f"Référence EF scalaire (per-file) : Δsrc={base_shift * 1000:+.1f} meV, "
                f"Δkin={kinetic_correction:+.3f} eV → offset={new_off:.4f} eV "
                f"(source: {ref_path_str})"
            )
        else:
            msg = (
                f"Référence EF scalaire : Δ={base_shift * 1000:+.1f} meV → "
                f"offset={new_off:.4f} eV (source: {ref_path_str})"
            )
            if src_ef_kin is None or tgt_ef_kin is None:
                msg += "  |  Attention: correction per-file impossible (ef_kin_nominal manquant)"
            elif not modes_match:
                msg += (
                    f"  |  Attention: modes énergie différents "
                    f"(src={src_ref_mode or '?'} vs cible={tgt_ref_mode or '?'}) — "
                    f"correction per-file ignorée"
                )
        return ReferenceApplication(new_ef_offset=new_off, ef_correction=ef_correction, msg=msg)

    raise ReferenceError(f"Référence EF mal formée (mode={mode!r})")
