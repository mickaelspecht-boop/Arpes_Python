"""Logique Γ / angles ARPES — fonctions pures, sans PyQt.

Extraites de `arpes_explorer.py` pour permettre des tests unitaires sans
lancer l'UI. Les conventions scientifiques (signes d'angles, formules de
projection) sont identiques à celles de l'ancienne classe `ArpesExplorer`.

NE PAS modifier les conventions sans en parler au conseil — toute inversion
de signe ou rotation introduite ici se propagerait silencieusement à tous
les fits CLS.
"""

from __future__ import annotations

import numpy as np

# Constante ARPES (identique à la valeur historique)
C_ARPES = 0.51233          # √(2 m_e) / ħ en unités appropriées
A_LATTICE_DEFAULT = 0.0    # 0 = inconnu ; le caller doit fournir SampleConfig.a


def k_to_angle_offset_deg(
    k_pi_a: float,
    *,
    hv: float,
    work_func: float,
    a_lattice: float = 0.0,
) -> float | None:
    """Convertit un décalage k (en π/a) en offset angulaire (deg) pour CLS.

    Renvoie ``None`` si l'énergie cinétique ``hv − φ`` est non valide.
    """
    try:
        ek = float(hv) - float(work_func)
    except Exception:
        return None
    if not np.isfinite(ek) or ek <= 0:
        return None
    scale = C_ARPES * np.sqrt(ek) * float(a_lattice) / np.pi
    if not np.isfinite(scale) or scale <= 0:
        return None
    arg = float(k_pi_a) / scale
    if abs(arg) > 1.0:
        arg = float(np.clip(arg, -1.0, 1.0))
    return float(np.degrees(np.arcsin(arg)))


def angle_offsets_from_k_center(
    kx: float,
    ky: float = 0.0,
    *,
    hv: float | None,
    work_func: float,
    source: str = "",
    ref_path: str | None = None,
    azi: float | None = None,
    a_lattice: float = 0.0,
) -> dict:
    """Construit le dict d'offsets angulaires à injecter dans le loader CLS.

    Renvoie ``{}`` si la conversion échoue (hv non valide, etc.).
    """
    if hv is None:
        return {}
    theta0 = k_to_angle_offset_deg(kx, hv=hv, work_func=work_func, a_lattice=a_lattice)
    tilt0 = k_to_angle_offset_deg(ky, hv=hv, work_func=work_func, a_lattice=a_lattice)
    if theta0 is None or tilt0 is None:
        return {}
    out = {
        "mode": "cls_angle_offsets",
        "theta0_deg": float(theta0),
        "tilt0_deg": float(tilt0),
        "source": source,
        "ref_path": ref_path or "",
        "hv": float(hv) if np.isfinite(float(hv)) else None,
        "work_func": float(work_func),
        "a_lattice": float(a_lattice),
    }
    if azi is not None:
        out["azi"] = float(azi)
    return out


def project_gamma_by_azi(
    ref: dict,
    azi_target: float | None,
    *,
    on_warn=None,
    warn_label: str = "Γ",
) -> tuple[float, float]:
    """Projette le Γ de référence dans le repère du fichier courant.

    La direction ZDB n'est pas utilisée ici : seule la différence d'azimut
    définit la rotation entre la FS de référence et la donnée cible.

    ``on_warn`` est un callback optionnel (str → None) pour remonter un
    avertissement quand l'azi est inconnu mais ``ky_ref`` non négligeable.
    """
    kx_ref = float(ref.get("kx", np.nan))
    ky_ref = float(ref.get("ky", 0.0) or 0.0)
    if not np.isfinite(kx_ref) or not np.isfinite(ky_ref):
        return float("nan"), float("nan")

    azi_ref = ref.get("azi")
    if azi_ref is None or azi_target is None:
        if abs(ky_ref) > 1e-3 and on_warn is not None:
            on_warn(f"Attention: {warn_label} : azi inconnu — projection non corrigée")
        return kx_ref, ky_ref

    d_azi = np.radians(float(azi_target) - float(azi_ref))
    k_parallel = kx_ref * np.cos(d_azi) + ky_ref * np.sin(d_azi)
    k_perp = -kx_ref * np.sin(d_azi) + ky_ref * np.cos(d_azi)
    return float(k_parallel), float(k_perp)


# Tolérance polar (deg) au-delà de laquelle le transfert FS→BM est refusé.
POLAR_TOLERANCE_DEG = 2.0


def build_gamma_reference(
    *,
    kx: float,
    ky: float,
    metadata: dict | None,
    hv: float | None,
    path: str | None,
    azi: float | None,
    source: str,
    direction: str | None = None,
) -> dict:
    """Construit le dict ``gamma_reference`` à stocker dans la session.

    Centralise la forme du dict utilisée à la fois par `_store_fs_center_reference`
    (clic FS) et `_estimate_gamma_bm` (Auto Γ BM).
    """
    meta = metadata or {}
    ref = {
        "kx": float(kx),
        "ky": float(ky),
        "polar": float(meta.get("polar", 0.0) or 0.0),
        "tilt": float(meta.get("tilt_ref", 0.0) or 0.0),
        "azi": float(azi) if azi is not None else None,
        "hv": hv,
        "path": path,
        "polar_already_applied_to_kx": bool(meta.get("polar_already_applied_to_kx", False)),
        "source": source,
    }
    if direction:
        ref["direction"] = direction
    return ref


def gamma_reference_to_bm_center(
    ref: dict,
    *,
    bm_metadata: dict | None,
    bm_hv: float | None,
    work_func: float,
    bm_azi: float | None,
    on_warn=None,
    polar_tolerance_deg: float = POLAR_TOLERANCE_DEG,
    a_lattice: float = 0.0,
) -> tuple[float, float]:
    """Projette le Γ mesuré sur la FS vers l'axe k de la BM courante.

    Retourne ``(gamma, correction)`` ou ``(NaN, 0.0)`` si :
    - ``bm_metadata`` est absent ;
    - le polar diffère de la référence de plus que ``polar_tolerance_deg`` ;
    - la projection azimutale échoue.

    ``correction`` est le résidu de polar appliqué quand
    ``polar_already_applied_to_kx`` n'est pas vrai des deux côtés.
    """
    if bm_metadata is None:
        return float("nan"), 0.0
    meta = bm_metadata or {}

    p_ref = float(ref.get("polar", 0.0) or 0.0)
    p_bm = float(meta.get("polar", 0.0) or 0.0)
    ref_polar_applied = bool(ref.get("polar_already_applied_to_kx", False))
    bm_polar_applied = bool(meta.get("polar_already_applied_to_kx", False))
    polar_baked_both_sides = ref_polar_applied and bm_polar_applied

    # Tolerance guard only meaningful when polar is NOT baked into kpar on both
    # sides. When both loaders apply polar in the angle→k conversion (e.g. CLS
    # `kx = scale·sin(θ−polar−θ0)`), the polar offset is already absorbed into
    # the kpar axis, so kx_FS at Γ equals kx_BM at Γ regardless of polar diff.
    if not polar_baked_both_sides and abs(p_bm - p_ref) > polar_tolerance_deg:
        if on_warn is not None:
            on_warn(
                f"Attention: Γ FS→BM ignoré : polar diffère de {p_bm - p_ref:+.1f}° "
                f"(>±{polar_tolerance_deg:.0f}°) et polar non absorbé dans k. "
                f"Utilise 'Auto Γ BM'."
            )
        return float("nan"), 0.0

    gamma, _ = project_gamma_by_azi(ref, bm_azi, on_warn=on_warn, warn_label="Γ FS→BM")
    if not np.isfinite(gamma):
        return float("nan"), 0.0

    correction = 0.0
    if not polar_baked_both_sides:
        hv = bm_hv if bm_hv is not None else ref.get("hv")
        if hv is not None and float(hv) > work_func:
            ek = float(hv) - float(work_func)
            correction = (
                C_ARPES * np.sqrt(ek)
                * (np.sin(np.radians(p_bm)) - np.sin(np.radians(p_ref)))
                * a_lattice / np.pi
            )
    return gamma + correction, correction


def apply_bm_gamma_axis_shift(
    raw_data: dict,
    gamma_bm: float,
    *,
    ref: dict | None = None,
    allow_fs: bool = False,
    gamma_ky: float = 0.0,
) -> bool:
    """Recentre l'axe k// d'une BM pour que Γ soit affiché à k//=0.

    Mute ``raw_data["kpar"]`` et ``raw_data["metadata"]`` en place.
    Retourne ``True`` si le shift a été appliqué, ``False`` sinon (FS sans
    `allow_fs`, offsets déjà appliqués, ``gamma_bm`` non fini, kpar vide).
    Si l'axe a déjà été recentré, applique seulement le delta entre l'ancien
    et le nouveau Γ pour permettre l'ajustement manuel sans recharger.

    Si ``allow_fs`` et ``raw_data`` est une FS, décale aussi ``fs_kx`` /
    ``fs_ky`` pour que le panel FS et la vue BM (qui lit ``kpar`` issu d'une
    coupe FS) montrent tous deux Γ à 0.

    NOTE : la mise à jour de l'état UI (sélection MDC `_sel_k`, marker FS)
    reste à la charge de l'appelant — cette fonction est pure côté
    ``raw_data``.
    """
    if not raw_data:
        return False
    meta = raw_data.get("metadata", {}) or {}
    is_fs = meta.get("fs_data") is not None
    if is_fs and not allow_fs:
        return False
    if meta.get("angle_offsets_applied"):
        return False
    if not np.isfinite(gamma_bm):
        return False

    kpar = np.asarray(raw_data.get("kpar"), dtype=float)
    if kpar.size == 0 or not np.isfinite(kpar).any():
        return False

    shift = float(gamma_bm)
    previous_shift = 0.0
    if bool(meta.get("bm_gamma_axis_centered", False)):
        try:
            previous_shift = float(meta.get("bm_gamma_axis_shift", 0.0) or 0.0)
        except Exception:
            previous_shift = 0.0
    delta = shift - previous_shift
    if abs(delta) < 1e-12 and bool(meta.get("bm_gamma_axis_centered", False)):
        return False

    raw_data["kpar"] = kpar - delta
    meta["bm_gamma_axis_centered"] = True
    meta["bm_gamma_axis_shift"] = shift
    meta["bm_gamma_axis_note"] = "kpar_display = kpar_raw - gamma_bm"
    if is_fs:
        ky_shift = float(gamma_ky) if np.isfinite(gamma_ky) else 0.0
        fs_kx = meta.get("fs_kx")
        if fs_kx is not None:
            meta["fs_kx"] = np.asarray(fs_kx, dtype=float) - delta
        fs_ky = meta.get("fs_ky")
        previous_ky_shift = 0.0
        try:
            previous_ky_shift = float(meta.get("fs_gamma_axis_shift_ky", 0.0) or 0.0)
        except Exception:
            previous_ky_shift = 0.0
        ky_delta = ky_shift - previous_ky_shift
        if fs_ky is not None and ky_delta != 0.0:
            meta["fs_ky"] = np.asarray(fs_ky, dtype=float) - ky_delta
        meta["fs_gamma_axis_centered"] = True
        meta["fs_gamma_axis_shift_kx"] = shift
        meta["fs_gamma_axis_shift_ky"] = ky_shift
    if ref:
        meta["bm_gamma_reference_source"] = ref.get("source", "")
        meta["bm_gamma_reference_path"] = ref.get("path", "")
        meta["bm_gamma_reference_azi"] = ref.get("azi")
    raw_data["metadata"] = meta
    return True


def _candidate_key(cfg: dict) -> tuple:
    return (
        round(float(cfg.get("theta0_deg", 0.0)), 8),
        round(float(cfg.get("tilt0_deg", 0.0)), 8),
        cfg.get("candidate", ""),
    )


def angle_offset_candidates_for_load(
    *,
    primary: dict | None,
    is_file: bool,
    ref: dict | None,
    target_geom: dict | None,
    target_azi_fallback: float | None,
    hv: float | None,
    work_func: float,
    a_lattice: float = 0.0,
) -> list[dict]:
    """Génère la liste de configurations d'offsets angulaires à essayer.

    Pour une BM CLS, plusieurs conventions de signes/projections sont
    plausibles selon comment l'azi est défini dans le logbook. On les énumère
    sans doublons (clé arrondie sur theta0/tilt0/label).

    `is_file` doit être ``True`` pour une BM (fichier .ibw / .pxt) — pour
    une FS (dossier) on retourne uniquement le primary.
    """
    if not primary or not is_file:
        return [primary] if primary else []
    candidates: list[dict] = []

    def add(cfg: dict, label: str):
        if not cfg:
            return
        c = dict(cfg)
        c["candidate"] = label
        key = _candidate_key(c)
        for old in candidates:
            if _candidate_key(old) == key:
                return
        candidates.append(c)

    add(primary, "theta0")
    neg = dict(primary)
    neg["theta0_deg"] = -float(neg.get("theta0_deg", 0.0) or 0.0)
    neg["gamma_bm_pi_over_a"] = -float(neg.get("gamma_bm_pi_over_a", 0.0) or 0.0)
    add(neg, "-theta0")

    if not ref:
        return candidates

    geom = target_geom or {}
    p_ref = ref.get("polar")
    p_target = geom.get("polar")
    theta_ref = k_to_angle_offset_deg(
        float(ref.get("kx", 0.0) or 0.0),
        hv=hv if hv is not None else 0.0,
        work_func=work_func,
        a_lattice=a_lattice,
    ) if hv is not None else None

    if p_ref is not None and p_target is not None and theta_ref is not None:
        # Convention "angle brut analyseur" : Γ FS ≈ theta_raw - P_ref ;
        # pour une BM à polar différent, l'offset à appliquer = theta_raw - P_target.
        raw_theta0 = float(theta_ref) + float(p_ref) - float(p_target)
        cfg = dict(primary)
        cfg["theta0_deg"] = raw_theta0
        cfg["tilt0_deg"] = 0.0
        cfg["source"] = "gamma_reference_projected_to_bm_raw_polar"
        cfg["target_polar"] = float(p_target)
        cfg["ref_polar"] = float(p_ref)
        add(cfg, "raw_polar")
        cfg_neg = dict(cfg)
        cfg_neg["theta0_deg"] = -raw_theta0
        add(cfg_neg, "raw_polar_neg")

    azi_ref = ref.get("azi")
    azi_bm = geom.get("azi", target_azi_fallback)
    if azi_ref is None or azi_bm is None or hv is None:
        return candidates
    kx_ref = float(ref.get("kx", float("nan")))
    ky_ref = float(ref.get("ky", 0.0) or 0.0)
    if not (np.isfinite(kx_ref) and np.isfinite(ky_ref)):
        return candidates

    d_azi = np.radians(float(azi_bm) - float(azi_ref))
    for label, gamma_bm in (
        ("azi_plus", kx_ref * np.cos(d_azi) + ky_ref * np.sin(d_azi)),
        ("azi_minus", kx_ref * np.cos(d_azi) - ky_ref * np.sin(d_azi)),
    ):
        cfg = angle_offsets_from_k_center(
            float(gamma_bm), 0.0,
            hv=hv, work_func=work_func, a_lattice=a_lattice,
            source=f"gamma_reference_projected_to_bm_{label}",
            ref_path=ref.get("path"),
            azi=azi_bm,
        )
        if not cfg:
            continue
        cfg["gamma_bm_pi_over_a"] = float(gamma_bm)
        cfg["gamma_ref_source"] = ref.get("source", "")
        add(cfg, label)
        if p_ref is not None and p_target is not None:
            theta_proj = k_to_angle_offset_deg(
                float(gamma_bm), hv=hv, work_func=work_func, a_lattice=a_lattice,
            )
            if theta_proj is not None:
                cfg_raw = dict(cfg)
                cfg_raw["theta0_deg"] = float(theta_proj) + float(p_ref) - float(p_target)
                cfg_raw["source"] = f"gamma_reference_projected_to_bm_{label}_raw_polar"
                cfg_raw["target_polar"] = float(p_target)
                cfg_raw["ref_polar"] = float(p_ref)
                add(cfg_raw, f"{label}_raw_polar")
        cfg_neg = dict(cfg)
        cfg_neg["theta0_deg"] = -float(cfg_neg.get("theta0_deg", 0.0) or 0.0)
        cfg_neg["gamma_bm_pi_over_a"] = -float(cfg_neg.get("gamma_bm_pi_over_a", 0.0) or 0.0)
        add(cfg_neg, f"{label}_neg")

    return candidates


def score_bm_gamma_residual(
    loaded: dict,
    *,
    ev_range: tuple[float, float],
    k_range: tuple[float, float],
    center_window: float,
    smooth_sigma: float,
    estimate_fn,
) -> float:
    """Score petit si la BM chargée est centrée autour de Γ=0.

    ``estimate_fn`` doit avoir la même signature que
    ``arpes_plots.estimate_gamma_bm_mdc``. Cette injection rend la fonction
    testable sans dépendre de `arpes_plots`.
    """
    if estimate_fn is None:
        return float("inf")
    try:
        res = estimate_fn(
            np.asarray(loaded["data"], dtype=float),
            np.asarray(loaded["kpar"], dtype=float),
            np.asarray(loaded["ev_arr"], dtype=float),
            ev_range=ev_range,
            k_range=k_range,
            center_guess=0.0,
            center_window=max(float(center_window), 0.25),
            smooth_sigma=float(smooth_sigma),
            verbose=False,
        )
        gamma = float(res.get("gamma", float("nan")))
        mad = float(res.get("mad", 0.0) or 0.0)
        n = int(res.get("n", 0) or 0)
        if not np.isfinite(gamma) or n < 2:
            return float("inf")
        kpar = np.asarray(loaded["kpar"], dtype=float)
        k_mid = 0.5 * (float(np.nanmin(kpar)) + float(np.nanmax(kpar)))
        return abs(gamma) + 0.25 * mad + 0.10 * abs(k_mid)
    except Exception:
        return float("inf")


def stored_gamma_reference(session_gamma_ref: dict | None) -> dict:
    """Filtre la référence Γ stockée en session : ``{}`` si invalide."""
    ref = session_gamma_ref or {}
    try:
        kx = float(ref.get("kx", np.nan))
        ky = float(ref.get("ky", 0.0) or 0.0)
    except Exception:
        return {}
    if not np.isfinite(kx) or not np.isfinite(ky):
        return {}
    return ref
