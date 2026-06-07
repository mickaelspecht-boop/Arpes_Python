"""EF calibration logic, without PyQt.

Extracted from `arpes_explorer.py`. The UI layer keeps `QMessageBox`,
`blockSignals` blocks, and the save/reload sequence; all scientific decisions
(mode choice, offset calculation, per-file correction) live here.

Conventions:
- ``ef_offset`` is the scalar offset applied to the energy axis to bring EF to
  0 (scalar mode), or is left at 0 when a per-column polynomial carries the
  full shift (poly mode);
- ``ef_correction`` is a dict describing the correction (poly_coefs, source,
  etc.); empty in pure scalar mode;
- ``ref_payload`` is what is stored in ``session.ef_reference`` so it can be
  applied later to other files in the folder.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def axis_zero_in_kinetic(meta: dict | None) -> float | None:
    """Return the kinetic energy corresponding to E−EF=0 BEFORE applying
    ``ef_offset`` (see historical docs in `arpes_explorer`).
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
    """Result of applying an EF calibration to an entry."""

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
    """Compute the session update for an EF calibration result.

    ``payload`` is the dict produced by `EFCalibrationDialog` (scalar or poly
    mode). ``current_ef_offset`` is the current value of the `sp_ef` spinbox.
    ``source_meta`` is `_raw_data["metadata"]` for the file being calibrated.
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
            f"Scalar EF: Δ={payload['ef_shift'] * 1000:+.1f} meV → "
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
        f"Per-column EF: {payload['n_valid']} valid k values, "
        f"FWHM≈{payload['fwhm_res'] * 1000:.0f} meV, "
        f"rms={payload['rms'] * 1000:.1f} meV"
    )
    return CalibrationUpdate(new_ef_offset=new_off, ef_correction=ef_correction,
                             ref_payload=ref_payload, msg=msg)


@dataclass
class ReferenceApplication:
    """Result of applying a session EF reference to a file."""

    new_ef_offset: float
    ef_correction: dict
    msg: str


class ReferenceError(ValueError):
    """Malformed EF reference (unknown/missing mode, etc.)."""


def already_applied(ef_correction: dict) -> bool:
    """Detect that an EF reference has already been applied to this file
    (guard against double-subtracting the scalar shift)."""
    src = (ef_correction or {}).get("source")
    return src in ("reference", "reference_scalar")


def apply_reference_to_target(
    ref: dict,
    *,
    current_ef_offset: float,
    target_meta: dict | None,
    ref_path_str: str,
) -> ReferenceApplication:
    """Apply ``ref`` (session.ef_reference) to the current target file.

    Raises ``ReferenceError`` if the mode is not recognized.
    """
    mode = ref.get("mode")
    if mode == "poly":
        ef_correction = dict(ref)
        ef_correction["source"] = "reference"
        msg = f"Poly EF reference applied (source: {ref_path_str})"
        return ReferenceApplication(new_ef_offset=0.0, ef_correction=ef_correction, msg=msg)

    if mode == "scalar":
        base_shift = float(ref.get("ef_shift", 0.0))
        # When each file has its own Center Energy (BESSY) or hν−φ
        # (Solaris/CLS), a scalar shift alone is invalid: compensate for the
        # kinetic-origin difference between source and target.
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
                f"Scalar EF reference (per-file): Δsrc={base_shift * 1000:+.1f} meV, "
                f"Δkin={kinetic_correction:+.3f} eV → offset={new_off:.4f} eV "
                f"(source: {ref_path_str})"
            )
        else:
            msg = (
                f"Scalar EF reference: Δ={base_shift * 1000:+.1f} meV → "
                f"offset={new_off:.4f} eV (source: {ref_path_str})"
            )
            if src_ef_kin is None or tgt_ef_kin is None:
                msg += "  |  Warning: per-file correction impossible (ef_kin_nominal missing)"
            elif not modes_match:
                msg += (
                    f"  |  Warning: different energy modes "
                    f"(src={src_ref_mode or '?'} vs target={tgt_ref_mode or '?'}) — "
                    f"per-file correction ignored"
                )
        return ReferenceApplication(new_ef_offset=new_off, ef_correction=ef_correction, msg=msg)

    raise ReferenceError(f"Malformed EF reference (mode={mode!r})")
