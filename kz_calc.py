#!/usr/bin/env python3
"""
kz_calc.py — Calcul et analyse du moment perpendiculaire kz pour BaNi₂As₂

Physique
--------
En ARPES, kz n'est pas conservé à la surface. On suppose un potentiel intérieur V₀ :

    kz [Å⁻¹] = 0.51233 × √( (hν − φ − EB)·cos²θ + V₀ )

À EF (EB=0) et émission normale (θ=0) :
    kz = 0.51233 × √(hν − φ + V₀)

BZ de BaNi₂As₂ (I4/mmm, c=11.65 Å, surface (001)) :
    Période Δkz = 2π/c ≈ 0.539 Å⁻¹
    Γ : kz = n × Δkz       Z : kz = (n+½) × Δkz

Utilisation typique
-------------------
    # 1) Analyser le scan kz et ajuster V₀ visuellement
    python3 kz_calc.py \\
        --kz-scan-dir BaNi2As2_/kz_scan_1 \\
        --out kz_results.csv

    # 2) Avec V₀ ajusté manuellement après inspection du graphique
    python3 kz_calc.py \\
        --kz-scan-dir BaNi2As2_/kz_scan_1 \\
        --v0 13.5 \\
        --out kz_results.csv

    # 3) Fitter V₀ automatiquement à partir de deux hν à Γ
    python3 kz_calc.py \\
        --kz-scan-dir BaNi2As2_/kz_scan_1 \\
        --fit-gamma 68 100 \\
        --out kz_results.csv

    # 4) Juste calculer pour des hν donnés (sans charger les fichiers)
    python3 kz_calc.py --hv 44 61 80 100 --v0 12.0 --no-plot
"""

from __future__ import annotations

import argparse
import io
import re
from pathlib import Path
from typing import List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar

# ─────────────────────────────────────────────────────────────────────────────
# Constantes BaNi₂As₂
# ─────────────────────────────────────────────────────────────────────────────
WORK_FUNC  = 4.0310   # eV — Solaris URANOS
A_LATTICE  = 3.960    # Å
C_LATTICE  = 11.65    # Å
V0_DEFAULT = 12.0     # eV — point de départ raisonnable

_C_ARPES   = 0.51233  # Å⁻¹ eV^(-1/2) = √(2mₑ)/ℏ


# ─────────────────────────────────────────────────────────────────────────────
# Formules kz
# ─────────────────────────────────────────────────────────────────────────────

def kz_ang(hv: float, phi: float, v0: float,
           eb: float = 0.0, theta_deg: float = 0.0) -> float:
    """kz en Å⁻¹. Retourne NaN si énergie cinétique trop faible."""
    ekin = hv - phi - eb
    cos2 = np.cos(np.radians(theta_deg)) ** 2
    arg  = ekin * cos2 + v0
    return _C_ARPES * np.sqrt(max(arg, 0.0))


def bz_period(c: float = C_LATTICE) -> float:
    return 2 * np.pi / c


def nearest_hs(kz: float, c: float = C_LATTICE) -> Tuple[str, float]:
    """Retourne (nom, distance en Å⁻¹) du point Γ ou Z le plus proche."""
    dkz  = bz_period(c)
    frac = (kz / dkz) % 1.0
    d_gamma = min(frac, 1.0 - frac) * dkz
    d_z     = abs(frac - 0.5)       * dkz
    if d_gamma <= d_z:
        n = round(kz / dkz)
        return f"Γ({n})", d_gamma
    else:
        n = int(kz / dkz)
        return f"Z({n})", d_z


def gamma_hvs(phi: float, v0: float, c: float,
              hv_min: float = 20.0, hv_max: float = 200.0) -> np.ndarray:
    """Énergies hν correspondant aux points Γ dans la fenêtre donnée."""
    dkz = bz_period(c)
    out = []
    for n in range(1, 50):
        kz_g  = n * dkz
        ekin  = (kz_g / _C_ARPES) ** 2 - v0
        hv    = ekin + phi
        if hv_min <= hv <= hv_max:
            out.append(hv)
    return np.array(out)


def z_hvs(phi: float, v0: float, c: float,
          hv_min: float = 20.0, hv_max: float = 200.0) -> np.ndarray:
    """Énergies hν correspondant aux points Z."""
    dkz = bz_period(c)
    out = []
    for n in range(0, 50):
        kz_z  = (n + 0.5) * dkz
        ekin  = (kz_z / _C_ARPES) ** 2 - v0
        hv    = ekin + phi
        if hv_min <= hv <= hv_max:
            out.append(hv)
    return np.array(out)


def fit_v0_from_two_gamma(hv1: float, hv2: float,
                          phi: float, c: float) -> Tuple[float, int]:
    """
    Ajuste V₀ sachant que hv1 et hv2 sont tous deux à des Γ.
    Teste N=1,2,3 périodes entre eux, retourne le meilleur (V₀, N).
    """
    dkz = bz_period(c)
    best_v0, best_n, best_err = V0_DEFAULT, 1, np.inf
    for n in range(1, 6):
        def res(v0):
            k1, k2 = kz_ang(hv1, phi, v0), kz_ang(hv2, phi, v0)
            return (abs(k2 - k1) - n * dkz) ** 2
        r = minimize_scalar(res, bounds=(0.5, 40.0), method="bounded")
        if r.fun < best_err and r.x > 0:
            best_err, best_v0, best_n = r.fun, r.x, n
    print(f"  Fit V₀ : hν={hv1}↔{hv2} eV, N={best_n} périodes → V₀={best_v0:.2f} eV")
    return float(best_v0), best_n


# ─────────────────────────────────────────────────────────────────────────────
# Chargement du scan kz (dossier de fichiers .ibw / .pxt)
# ─────────────────────────────────────────────────────────────────────────────

def _hv_from_filename(path: Path) -> Optional[float]:
    """Extrait hν depuis le nom de fichier (ex: kz_80.0.ibw → 80.0)."""
    m = re.search(r"kz[_\s]*([\d.]+)", path.stem, re.IGNORECASE)
    if m:
        return float(m.group(1))
    return None


def load_kz_scan(scan_dir: Path,
                 phi: float, v0: float, c: float,
                 ef_window_ev: float = 0.05,
                 k_window: Optional[Tuple[float, float]] = None
                 ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Charge tous les fichiers .ibw/.pxt du dossier kz_scan.

    Retourne :
        hv_arr   : (N,)   énergies photon triées
        kz_arr   : (N,)   kz calculé [Å⁻¹]
        ev_arr   : (M,)   axe énergie commun [eV]
        edc_map  : (N, M) EDC intégrée en k pour chaque hν (normalisée 0→1)
    """
    try:
        import erlab.io
        erlab.io.set_loader("da30")
    except ImportError:
        raise SystemExit("erlab non trouvé. Active ton environnement conda.")

    files = sorted(
        [p for p in scan_dir.iterdir() if p.suffix.lower() in (".ibw", ".pxt")],
        key=lambda p: _hv_from_filename(p) or 0.0,
    )
    if not files:
        raise FileNotFoundError(f"Aucun fichier .ibw/.pxt dans {scan_dir}")

    records = []
    for f in files:
        hv_file = _hv_from_filename(f)
        if hv_file is None:
            continue
        try:
            da = erlab.io.load(str(f))
            hv_attr = float(da.attrs.get("hv", hv_file))
            hv_used = hv_attr

            # Conversion énergie cinétique → énergie de liaison
            ef_kin = hv_used - phi
            import warnings
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                da_be = da.assign_coords(eV=da.eV - ef_kin, hv=hv_used, xi=0.0)

            ev  = np.asarray(da_be.eV.values, dtype=float)
            dat = np.asarray(da_be.values,    dtype=float)

            # Intégrer sur k en détectant l'axe eV (peut être 0 ou 1 selon erlab)
            if dat.ndim == 2:
                if dat.shape[0] == len(ev):      # (nE, nK)
                    edc = np.nanmean(dat, axis=1)
                elif dat.shape[1] == len(ev):    # (nK, nE)
                    edc = np.nanmean(dat, axis=0)
                else:
                    edc = np.nanmean(dat, axis=0)
            elif dat.ndim == 3:
                # Trouver l'axe qui correspond à eV
                ev_axis = next((i for i, s in enumerate(dat.shape) if s == len(ev)), 0)
                k_axes  = tuple(i for i in range(3) if i != ev_axis)
                edc     = np.nanmean(dat, axis=k_axes)
            else:
                edc = dat.ravel()

            records.append({"hv": hv_used, "ev": ev, "edc": edc})
            print(f"  Chargé {f.name}  hν={hv_used:.1f} eV  shape={dat.shape}")
        except Exception as exc:
            print(f"  ⚠ {f.name} : {exc}")

    if not records:
        raise RuntimeError("Aucun fichier chargé correctement.")

    records.sort(key=lambda r: r["hv"])

    # Axe énergie commun : intersection de tous les axes
    ev_min = max(r["ev"].min() for r in records)
    ev_max = min(r["ev"].max() for r in records)
    n_pts  = min(len(r["ev"]) for r in records)
    ev_common = np.linspace(ev_min, ev_max, n_pts)

    hv_arr  = np.array([r["hv"] for r in records])
    kz_arr  = np.array([kz_ang(hv, phi, v0) for hv in hv_arr])
    edc_map = np.zeros((len(records), len(ev_common)))

    for i, r in enumerate(records):
        edc_interp = np.interp(ev_common, r["ev"], r["edc"])
        # Normalisation 0→1 par fichier
        lo, hi = np.nanpercentile(edc_interp, [2, 98])
        edc_map[i] = (edc_interp - lo) / (hi - lo + 1e-12)

    return hv_arr, kz_arr, ev_common, edc_map


# ─────────────────────────────────────────────────────────────────────────────
# Tableau de résultats
# ─────────────────────────────────────────────────────────────────────────────

def build_table(hv_arr: np.ndarray, phi: float, v0: float, c: float,
                meas_nos: Optional[List] = None,
                modes: Optional[List[str]] = None) -> pd.DataFrame:
    rows = []
    dkz  = bz_period(c)
    for i, hv in enumerate(hv_arr):
        kz  = kz_ang(hv, phi, v0)
        hs_name, hs_dist = nearest_hs(kz, c)
        rows.append({
            "meas_no":         meas_nos[i] if meas_nos else i + 1,
            "mode":            modes[i]    if modes    else "?",
            "hv_eV":           round(hv, 2),
            "kz_Ang":          round(kz,   4),
            "kz_pi_c":         round(kz / (np.pi / c), 4),
            "kz_2pi_c":        round(kz / dkz, 4),
            "nearest_hs":      hs_name,
            "dist_hs_Ang":     round(hs_dist, 4),
            "dist_hs_frac_BZ": round(hs_dist / dkz, 4),
            "phi_eV":          phi,
            "v0_eV":           round(v0, 3),
            "c_Ang":           c,
        })
    return pd.DataFrame(rows)


# ─────────────────────────────────────────────────────────────────────────────
# Graphiques
# ─────────────────────────────────────────────────────────────────────────────

def plot_kz_map(hv_arr: np.ndarray, kz_arr: np.ndarray,
                ev_arr: np.ndarray, edc_map: np.ndarray,
                phi: float, v0: float, c: float,
                out_path: Optional[Path] = None) -> None:
    """
    Carte 2D EDC vs hν (= EDC vs kz).
    Chaque colonne = une EDC à hν fixe. Lignes horizontales = Γ et Z.

    Comment lire :
      - Axe X du haut : hν (eV)    Axe X du bas : kz (Å⁻¹)
      - Axe Y : énergie de liaison (eV), 0 = EF
      - Couleur : intensité ARPES normalisée. Les bandes apparaissent comme
        des traînées verticales qui se déplacent avec kz (dispersion kz).
      - Lignes bleues pleines  = Γ  (bandes à maximum ou minimum)
      - Lignes rouges tiretées = Z  (bandes à l'autre extrémum)
      - Si une bande monte/descend périodiquement → elle disperse avec kz (caractère 3D)
      - Si elle reste plate  → caractère 2D (confiné dans les plans a-b)
    """
    dkz = bz_period(c)
    fig, axes = plt.subplots(1, 2, figsize=(16, 7),
                             gridspec_kw={"width_ratios": [3, 1], "wspace": 0.05})
    ax_map, ax_edc = axes

    # ── Carte EDC vs kz ───────────────────────────────────────────────────
    extent = [kz_arr[0], kz_arr[-1], ev_arr[0], ev_arr[-1]]
    im = ax_map.imshow(
        edc_map.T, origin="lower", aspect="auto",
        extent=extent, cmap="inferno",
        vmin=0, vmax=1,
    )
    plt.colorbar(im, ax=ax_map, label="Intensité normalisée")
    ax_map.axhline(0, color="cyan", lw=0.9, ls="--", label="EF")

    # Lignes Γ et Z
    for n in range(0, 30):
        kg = n * dkz
        kz_ = (n + 0.5) * dkz
        if kz_arr[0] <= kg <= kz_arr[-1]:
            ax_map.axvline(kg,  color="steelblue", lw=1.2, ls="-",  alpha=0.7,
                           label="Γ" if n == 0 else "_")
            ax_map.text(kg, ev_arr[-1] * 0.98, f"Γ({n})",
                        ha="center", va="top", fontsize=8, color="steelblue")
        if kz_arr[0] <= kz_ <= kz_arr[-1]:
            ax_map.axvline(kz_, color="tomato",    lw=1.2, ls="--", alpha=0.7,
                           label="Z" if n == 0 else "_")
            ax_map.text(kz_, ev_arr[-1] * 0.92, f"Z({n})",
                        ha="center", va="top", fontsize=8, color="tomato")

    # Axe hν en haut
    ax_top = ax_map.twiny()
    ax_top.set_xlim(ax_map.get_xlim())
    tick_kz = np.interp(hv_arr[::4], hv_arr, kz_arr)  # sous-échantillonner
    ax_top.set_xticks(kz_arr[::4])
    ax_top.set_xticklabels([f"{h:.0f}" for h in hv_arr[::4]], fontsize=8)
    ax_top.set_xlabel("hν (eV)", fontsize=10)

    ax_map.set_xlabel("kz (Å⁻¹)", fontsize=11)
    ax_map.set_ylabel("E − EF (eV)", fontsize=11)
    ax_map.set_title(
        f"Scan kz BaNi₂As₂  |  φ={phi} eV  V₀={v0:.1f} eV  c={c} Å\n"
        f"Γ (bleu) : bandes à extremum | Z (rouge) : bandes à l'autre extremum",
        fontsize=10,
    )
    ax_map.legend(fontsize=8, loc="lower left")

    # ── EDC moyenne (colonne droite) ──────────────────────────────────────
    edc_mean = np.nanmean(edc_map, axis=0)
    ax_edc.plot(edc_mean, ev_arr, color="steelblue", lw=1.2)
    ax_edc.fill_betweenx(ev_arr, 0, edc_mean, alpha=0.25, color="steelblue")
    ax_edc.axhline(0, color="cyan", lw=0.9, ls="--")
    ax_edc.set_xlabel("Intensité\nmoyenne", fontsize=9)
    ax_edc.set_yticklabels([])
    ax_edc.set_ylim(ev_arr[0], ev_arr[-1])
    ax_edc.set_title("EDC\nmoyennée", fontsize=9)
    ax_edc.set_xlim(left=0)

    plt.tight_layout()
    if out_path:
        fig.savefig(out_path, dpi=130, bbox_inches="tight")
        print(f"Graphique kz map : {out_path}")
    else:
        plt.show()
    plt.close(fig)


def plot_kz_curve(df: pd.DataFrame, phi: float, v0: float, c: float,
                  hv_min: float = 20.0, hv_max: float = 200.0,
                  out_path: Optional[Path] = None) -> None:
    """
    Courbe kz(hν) avec les positions Γ/Z et les mesures disponibles.
    Utile pour savoir à quel hν mesurer pour être sur Γ ou Z.
    """
    dkz      = bz_period(c)
    hv_cont  = np.linspace(hv_min, hv_max, 600)
    kz_cont  = np.array([kz_ang(h, phi, v0) for h in hv_cont])
    g_hvs    = gamma_hvs(phi, v0, c, hv_min, hv_max)
    z_hvs_   = z_hvs(phi, v0, c, hv_min, hv_max)

    fig, ax = plt.subplots(figsize=(11, 5))
    ax2 = ax.twinx()

    ax.plot(hv_cont, kz_cont, "k-", lw=1.5,
            label=f"kz(hν)  φ={phi} eV  V₀={v0:.1f} eV")

    # Positions Γ
    for hv_g in g_hvs:
        kz_g = kz_ang(hv_g, phi, v0)
        ax.axhline(kz_g, color="steelblue", lw=0.7, ls="-", alpha=0.4)
    ax.scatter(g_hvs, [kz_ang(h, phi, v0) for h in g_hvs],
               color="steelblue", marker="^", s=60, zorder=4, label="Γ")

    # Positions Z
    for hv_z in z_hvs_:
        kz_z = kz_ang(hv_z, phi, v0)
        ax.axhline(kz_z, color="tomato", lw=0.7, ls="--", alpha=0.4)
    ax.scatter(z_hvs_, [kz_ang(h, phi, v0) for h in z_hvs_],
               color="tomato", marker="v", s=60, zorder=4, label="Z")

    # Mesures du logbook
    for _, row in df.iterrows():
        hv  = row["hv_eV"]
        kz  = row["kz_Ang"]
        hs  = row["nearest_hs"]
        d   = row["dist_hs_frac_BZ"]
        col = "limegreen" if ("Γ" in hs and d < 0.15) else \
              ("tomato"   if ("Z" in hs and d < 0.15) else "gray")
        ax.scatter(hv, kz, color=col, s=90, zorder=6,
                   edgecolors="k", linewidths=0.6)
        ax.annotate(f"{hv:.0f} eV\n{hs}\n±{d:.2f}BZ",
                    (hv, kz), textcoords="offset points",
                    xytext=(5, 4), fontsize=7)

    kz_all  = kz_cont[np.isfinite(kz_cont)]
    ax.set_ylim(kz_all.min() * 0.97, kz_all.max() * 1.03)
    ax2.set_ylim(ax.get_ylim()[0] / dkz, ax.get_ylim()[1] / dkz)

    ax.set_xlabel("hν (eV)", fontsize=11)
    ax.set_ylabel("kz (Å⁻¹)", fontsize=11)
    ax2.set_ylabel("kz / (2π/c)  [périodes BZ]", fontsize=10)
    ax.set_title(
        f"BaNi₂As₂ — kz(hν)  |  Γ=bleu▲  Z=rouge▼  vert=proche Γ  orange=proche Z\n"
        f"Δkz = 2π/c = {dkz:.4f} Å⁻¹  |  c = {c} Å",
        fontsize=10,
    )
    ax.legend(fontsize=9)
    plt.tight_layout()

    if out_path:
        fig.savefig(out_path, dpi=130, bbox_inches="tight")
        print(f"Courbe kz(hν)   : {out_path}")
    else:
        plt.show()
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# Lecture logbook
# ─────────────────────────────────────────────────────────────────────────────

def read_logbook(path: Path) -> pd.DataFrame:
    raw = path.read_bytes()
    txt = raw.decode("utf-8-sig", errors="replace")
    sep = "\t" if txt.count("\t") > txt.count(";") else ";"
    return pd.read_csv(io.StringIO(txt), sep=sep)


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Calcul kz + analyse scan kz BaNi₂As₂",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--kz-scan-dir", type=Path,
                     help="Dossier contenant les fichiers .ibw/.pxt du scan kz")
    src.add_argument("--logbook", type=Path,
                     help="CSV logbook principal (hν depuis Monochromator energy [eV])")
    src.add_argument("--hv", nargs="+", type=float,
                     help="Valeurs hν directement (eV)")

    p.add_argument("--phi",  type=float, default=WORK_FUNC,
                   help=f"Travail φ [eV] (défaut={WORK_FUNC})")
    p.add_argument("--v0",   type=float, default=None,
                   help=f"Potentiel interne V₀ [eV] (défaut={V0_DEFAULT})")
    p.add_argument("--c",    type=float, default=C_LATTICE,
                   help=f"Paramètre c [Å] (défaut={C_LATTICE})")
    p.add_argument("--fit-gamma", nargs=2, type=float, metavar=("HV1", "HV2"),
                   help="Deux hν à Γ → fit V₀ automatique")
    p.add_argument("--ef-window", type=float, default=0.05,
                   help="Fenêtre autour de EF pour EDC [eV] (défaut=0.05)")
    p.add_argument("--out",  type=Path, default=None,
                   help="CSV de sortie (lu par fs_batch.py)")
    p.add_argument("--no-plot", action="store_true", help="Pas de graphique")
    return p.parse_args()


def main():
    args = parse_args()
    phi = args.phi
    c   = args.c
    dkz = bz_period(c)

    # ── V₀ ───────────────────────────────────────────────────────────────
    if args.fit_gamma:
        hv1, hv2 = sorted(args.fit_gamma)
        v0, _    = fit_v0_from_two_gamma(hv1, hv2, phi, c)
    else:
        v0 = args.v0 if args.v0 is not None else V0_DEFAULT

    print(f"\n{'='*60}")
    print(f"BaNi₂As₂ kz calculator")
    print(f"  φ = {phi} eV  |  V₀ = {v0:.2f} eV  |  c = {c} Å")
    print(f"  Δkz = 2π/c = {dkz:.4f} Å⁻¹")
    print(f"{'='*60}")

    # ── Points Γ et Z dans 20–200 eV ─────────────────────────────────────
    g_hvs = gamma_hvs(phi, v0, c)
    z_hvs_ = z_hvs(phi, v0, c)
    print(f"\nPoints Γ (20–200 eV) : {np.round(g_hvs, 1)} eV")
    print(f"Points Z (20–200 eV) : {np.round(z_hvs_, 1)} eV")

    # ── Charger données kz et construire table ────────────────────────────
    edc_map = None
    if args.kz_scan_dir:
        print(f"\nChargement scan kz depuis {args.kz_scan_dir} ...")
        hv_arr, kz_arr, ev_arr, edc_map = load_kz_scan(
            args.kz_scan_dir, phi, v0, c, ef_window_ev=args.ef_window,
        )
        df = build_table(hv_arr, phi, v0, c)

    elif args.logbook:
        df_lb   = read_logbook(args.logbook)
        hv_col  = "Monochromator energy [eV]"
        hv_arr  = np.sort(df_lb[hv_col].dropna().unique().astype(float))
        kz_arr  = np.array([kz_ang(h, phi, v0) for h in hv_arr])
        meas    = df_lb["Measurement NO"].tolist()
        modes   = df_lb.get("Mode", pd.Series(["?"]*len(df_lb))).tolist()
        df = build_table(
            df_lb[hv_col].astype(float).values, phi, v0, c,
            meas_nos=meas, modes=modes,
        )
    else:
        hv_arr = np.array(sorted(args.hv))
        kz_arr = np.array([kz_ang(h, phi, v0) for h in hv_arr])
        df = build_table(hv_arr, phi, v0, c)

    # ── Affichage table ───────────────────────────────────────────────────
    print(f"\n{df.to_string(index=False)}")

    # ── Sauvegarde CSV ────────────────────────────────────────────────────
    if args.out:
        df.to_csv(args.out, index=False)
        import json
        params = {
            "phi": phi, "v0": round(v0, 3),
            "c_lattice": c, "a_lattice": A_LATTICE,
        }
        json_path = args.out.with_suffix(".json")
        json_path.write_text(json.dumps(params, indent=2))
        print(f"\nCSV sauvegardé : {args.out}")
        print(f"JSON paramètres : {json_path}  ← utilisé par fs_batch.py")

    # ── Graphiques ────────────────────────────────────────────────────────
    if not args.no_plot:
        out_dir = args.out.parent if args.out else Path(".")

        # 1) Carte EDC vs kz (si scan kz chargé)
        if edc_map is not None:
            plot_kz_map(
                hv_arr, kz_arr, ev_arr, edc_map,
                phi, v0, c,
                out_path=out_dir / "kz_map.png" if args.out else None,
            )

        # 2) Courbe kz(hν) avec mesures
        # Utiliser une table simple (une ligne par hν unique)
        df_unique = build_table(
            np.unique(df["hv_eV"].values), phi, v0, c,
        )
        plot_kz_curve(
            df_unique, phi, v0, c,
            hv_min=20, hv_max=200,
            out_path=out_dir / "kz_curve.png" if args.out else None,
        )


if __name__ == "__main__":
    main()
