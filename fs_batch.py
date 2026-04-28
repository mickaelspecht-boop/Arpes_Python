#!/usr/bin/env python3
"""
fs_batch.py — Surfaces de Fermi ARPES pour BaNi₂As₂

Deux modes d'utilisation
------------------------
1) FS 2D kx×ky  (fast maps — deflector scans à hν fixe)
   → Chaque fichier .zip est un fast map complet.
   → Intégration ±window_meV autour de EF → image kx vs ky.
   → On affiche plusieurs hν/kz dans une grille.

2) FS kx×kz  (fixed cuts à hν variables — scan kz)
   → Chaque .ibw est une coupe kx(E) à hν différent.
   → On empile les coupes à EF pour former kx vs kz.

Dans les deux cas, on superpose :
   - Frontière de la zone de Brillouin (carré tétragonal √2×√2 π/a)
   - Points de haute symétrie Γ, X, M (BaNi₂As₂ I4/mmm)
   - Si V₀ fourni : étiquette kz sur chaque carte

Usage
-----
    # FS 2D kx×ky depuis les fast maps, avec paramètres kz
    python3 fs_batch.py \\
        --mode fastmap \\
        --data-dir BaNi2As2_ \\
        --logbook BaNi2As2_/___BANI2AS2__FOLDER_LOGBOOK___.csv \\
        --kz-params kz_results.json \\
        --out-dir fs_out

    # FS kx×kz depuis le scan kz
    python3 fs_batch.py \\
        --mode kxkz \\
        --kz-scan-dir BaNi2As2_/kz_scan_1 \\
        --kz-params kz_results.json \\
        --out-dir fs_out

    # Sans fichier JSON kz (utilise V₀ par défaut)
    python3 fs_batch.py --mode fastmap --data-dir BaNi2As2_ \\
        --logbook BaNi2As2_/___BANI2AS2__FOLDER_LOGBOOK___.csv \\
        --out-dir fs_out
"""

from __future__ import annotations

import argparse
import io
import json
import re
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.collections import LineCollection
import numpy as np
import pandas as pd

try:
    import erlab.io
except ImportError:
    raise SystemExit("erlab non trouvé. Active ton environnement conda.")

# ─────────────────────────────────────────────────────────────────────────────
# Paramètres par défaut BaNi₂As₂
# ─────────────────────────────────────────────────────────────────────────────
WORK_FUNC  = 4.0310   # eV
A_LATTICE  = 3.960    # Å
C_LATTICE  = 11.65    # Å
V0_DEFAULT = 12.0     # eV
_C_ARPES   = 0.51233  # Å⁻¹ eV^(-1/2)


def kz_ang(hv: float, phi: float, v0: float) -> float:
    arg = hv - phi + v0
    return _C_ARPES * np.sqrt(max(arg, 0.0))


def bz_period(c: float) -> float:
    return 2 * np.pi / c


def nearest_hs_label(kz: float, c: float) -> str:
    dkz = bz_period(c)
    n   = round(kz / dkz)
    frac = (kz / dkz) % 1.0
    d_gamma = min(frac, 1.0 - frac)
    d_z     = abs(frac - 0.5)
    if d_gamma <= d_z:
        return f"Γ({n})  ±{d_gamma:.2f}BZ"
    else:
        return f"Z({int(kz/dkz)})  ±{d_z:.2f}BZ"


# ─────────────────────────────────────────────────────────────────────────────
# Zone de Brillouin BaNi₂As₂ (surface (001), I4/mmm)
# Axes en π/a (comme le notebook) : BZ_HALF = 1.0, X=±1, M=±√2
# ─────────────────────────────────────────────────────────────────────────────

def _draw_bz_pia(ax: plt.Axes, bz_half: float = 1.0,
                 color: str = "white", lw: float = 1.2, alpha: float = 0.8,
                 ls: str = "--") -> None:
    """BZ carrée en unités π/a (±bz_half sur chaque axe)."""
    b = bz_half
    corners = np.array([[-b, -b], [b, -b], [b, b], [-b, b], [-b, -b]])
    ax.plot(corners[:, 0], corners[:, 1],
            color=color, lw=lw, ls=ls, alpha=alpha, zorder=5)


def overlay_bz_hsym(ax: plt.Axes, bz_half: float = 1.0,
                    klim: float = 1.3) -> None:
    """
    Superpose BZ + points haute symétrie (comme le notebook).
    Axes en π/a : Γ=(0,0) blanc, X=(±1,0)/(0,±1) cyan, M=(±1,±1) vert.
    """
    b = bz_half

    # BZ boundary
    _draw_bz_pia(ax, b)

    def _dot(x, y, name, color, off=(4, 4)):
        ax.scatter(x, y, c=color, s=40, zorder=7, linewidths=0)
        ax.annotate(name, (x, y), xytext=off, textcoords="offset points",
                    color=color, fontsize=9, fontweight="bold")

    _dot(0, 0, "Γ", "white")
    for px, py in [(b, 0), (-b, 0), (0, b), (0, -b)]:
        _dot(px, py, "X", "cyan")
    for px, py in [(b, b), (b, -b), (-b, b), (-b, -b)]:
        _dot(px, py, "M", "lime")

    ax.axhline(0, color="white", lw=0.5, ls="--", alpha=0.3)
    ax.axvline(0, color="white", lw=0.5, ls="--", alpha=0.3)
    ax.set_xlim(-klim, klim)
    ax.set_ylim(-klim, klim)
    ax.set_aspect("equal")


# ─────────────────────────────────────────────────────────────────────────────
# Lecture logbook
# ─────────────────────────────────────────────────────────────────────────────

def read_logbook(path: Path) -> pd.DataFrame:
    raw = path.read_bytes()
    txt = raw.decode("utf-8-sig", errors="replace")
    sep = "\t" if txt.count("\t") > txt.count(";") else ";"
    return pd.read_csv(io.StringIO(txt), sep=sep)


def load_kz_params(json_path: Optional[Path]) -> Dict:
    defaults = {"phi": WORK_FUNC, "v0": V0_DEFAULT,
                "c_lattice": C_LATTICE, "a_lattice": A_LATTICE}
    if json_path is None or not json_path.exists():
        print(f"  ⚠ Pas de fichier kz_params.json → V₀ par défaut = {V0_DEFAULT} eV")
        return defaults
    params = json.loads(json_path.read_text())
    defaults.update(params)
    print(f"  Paramètres kz chargés : φ={defaults['phi']} eV  V₀={defaults['v0']} eV")
    return defaults


# ─────────────────────────────────────────────────────────────────────────────
# Mode 1 : FS 2D kx×ky (fast maps)
# ─────────────────────────────────────────────────────────────────────────────

def _extract_fs_map(filepath: Path, phi: float, a: float,
                    ef_window: float = 0.030,
                    smooth_sigma: float = 1.2) -> Optional[Dict]:
    """
    Charge un fast map (.zip DA30) et extrait la carte FS intégrée ±ef_window.
    kx/ky retournés en π/a (comme le notebook).
    """
    try:
        da = erlab.io.load(str(filepath))
        hv = float(da.attrs.get("hv", np.nan))
        if not np.isfinite(hv):
            return None

        ef_kin = hv - phi
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            da_be = da.assign_coords(eV=da.eV - ef_kin, hv=hv, xi=0.0)
        da_be.attrs["configuration"] = 1
        da_be.kspace.work_function = phi

        da_k = da_be.kspace.convert()
        # Convertir kx, ky de Å⁻¹ → π/a
        da_k = da_k.assign_coords(
            kx=da_k.kx * a / np.pi,
            ky=da_k.ky * a / np.pi,
        )

        from scipy.ndimage import gaussian_filter

        ev_arr = np.asarray(da_k.eV.values, dtype=float)
        ev_min = float(ev_arr.min())
        ev_max = float(ev_arr.max())

        # ── Normalisation profil : corrige les variations de flux pendant le scan ──
        # Pour chaque position ky (angle deflecteur), on divise par l'intensité
        # intégrée dans une fenêtre de référence (loin de EF pour éviter les bandes).
        # Fenêtre de référence : -0.6 à -0.2 eV (bulk, loin des features proches EF)
        ref_lo = max(-0.60, ev_min)
        ref_hi = min(-0.20, ev_max - 0.05)
        if ref_hi > ref_lo:
            profile = da_k.sel(eV=slice(ref_lo, ref_hi)).mean("eV")   # (ky, kx) ou (kx, ky)
            # profil par ky : moyenne sur kx
            if "ky" in profile.dims:
                ky_profile = profile.mean("kx")    # 1D selon ky
                norm_da    = da_k / (ky_profile + 1e-12)
            else:
                norm_da = da_k   # fallback sans normalisation
        else:
            norm_da = da_k

        # ── Intégration ±ef_window autour de EF ──
        lo = max(-ef_window, ev_min)
        hi = min( ef_window, ev_max)
        fs_da = norm_da.sel(eV=slice(lo, hi)).mean("eV")

        kx  = np.asarray(fs_da.kx.values, dtype=float)
        ky  = np.asarray(fs_da.ky.values, dtype=float)
        fsm = np.asarray(fs_da.values,    dtype=float)

        # Mettre en forme (nky, nkx) pour pcolormesh(kx, ky, fsm)
        if fsm.shape == (len(kx), len(ky)):
            fsm = fsm.T   # → (nky, nkx)

        # Lissage gaussien + normalisation affichage
        nan_mask = np.isnan(fsm)
        fsm_fill = np.where(nan_mask, 0.0, fsm)
        fsm_sm   = gaussian_filter(fsm_fill, sigma=smooth_sigma)
        fsm_sm[nan_mask] = np.nan

        lo99, hi99 = np.nanpercentile(fsm_sm, [1, 99])
        fsm_n = np.clip((fsm_sm - lo99) / (hi99 - lo99 + 1e-12), 0, 1)

        return {
            "kx": kx, "ky": ky,
            "fs_n": fsm_n,
            "hv": hv, "filepath": filepath,
        }
    except Exception as exc:
        print(f"  ⚠ {filepath.name} : {exc}")
        return None


def plot_fs_fastmaps(data_dir: Path,
                     kz_params: Dict, ef_window: float,
                     out_dir: Path, dpi: int = 130,
                     smooth_sigma: float = 1.2,
                     klim: float = 1.3) -> None:
    """
    Charge tous les .zip de data_dir et trace la FS kx×ky.
    Normalisation profil (flux) + BZ + Γ/X/M comme le notebook.
    """
    phi = kz_params["phi"]
    v0  = kz_params["v0"]
    a   = kz_params["a_lattice"]
    c   = kz_params["c_lattice"]

    zip_files = sorted(data_dir.glob("*.zip"))
    if not zip_files:
        print(f"Aucun .zip trouvé dans {data_dir}")
        return
    print(f"  {len(zip_files)} fichier(s) .zip trouvés")

    erlab.io.set_loader("da30")
    results = []
    for f in zip_files:
        print(f"  → {f.name}")
        r = _extract_fs_map(f, phi, a, ef_window, smooth_sigma)
        if r is not None:
            r["kz"]       = kz_ang(r["hv"], phi, v0)
            r["kz_label"] = nearest_hs_label(r["kz"], c)
            results.append(r)

    if not results:
        print("Aucun fast map chargé.")
        return

    n     = len(results)
    ncols = min(4, n)
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=(5.0 * ncols, 5.0 * nrows),
        squeeze=False,
    )

    for idx, r in enumerate(results):
        ax = axes[idx // ncols][idx % ncols]
        kx, ky = r["kx"], r["ky"]
        hv     = r["hv"]

        ax.pcolormesh(kx, ky, r["fs_n"], cmap="inferno",
                      vmin=0, vmax=1, shading="auto")
        overlay_bz_hsym(ax, bz_half=1.0, klim=klim)
        ax.set_xlabel("kx (π/a)", fontsize=8)
        ax.set_ylabel("ky (π/a)", fontsize=8)
        ax.set_title(
            f"{r['filepath'].stem}\nhν={hv:.0f} eV  "
            f"kz={r['kz']:.3f} Å⁻¹  {r['kz_label']}\n"
            f"±{ef_window*1000:.0f} meV  [norm. profil flux]",
            fontsize=7,
        )

    for idx in range(n, nrows * ncols):
        axes[idx // ncols][idx % ncols].set_visible(False)

    fig.suptitle(
        f"BaNi₂As₂ — Surfaces de Fermi (fast maps)  |  "
        f"φ={phi} eV  V₀={v0:.1f} eV  a={a} Å\n"
        f"Axe : π/a  |  BZ : Γ=blanc  X=cyan  M=vert  |  "
        f"Intégration ±{ef_window*1000:.0f} meV  σ={smooth_sigma}",
        fontsize=11, fontweight="bold",
    )
    plt.tight_layout()
    out_path = out_dir / "fs_fastmaps.png"
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"\nFS fast maps → {out_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Mode 2 : FS kx×kz (scan kz, fixed cuts)
# ─────────────────────────────────────────────────────────────────────────────

def _hv_from_filename(path: Path) -> Optional[float]:
    m = re.search(r"kz[_\s]*([\d.]+)", path.stem, re.IGNORECASE)
    if m:
        return float(m.group(1))
    return None


def plot_fs_kxkz(kz_scan_dir: Path, kz_params: Dict,
                 ef_window: float, out_dir: Path,
                 dpi: int = 130) -> None:
    """
    Empile les MDC à EF de chaque fixed cut pour former la carte kx×kz.

    Comment lire :
      - Axe X : k// = kx (Å⁻¹), direction ΓX (R2≈2°)
      - Axe Y : kz (Å⁻¹) — chaque ligne = un hν différent
      - Couleur : intensité à EF (intégrée ±ef_window)
      - Lignes verticales bleues/rouges = Γ/Z
      - Lignes horizontales pointillées = positions kz des mesures
      - Le motif se répète avec la période 2π/c → mesure la dispersion en kz
    """
    phi = kz_params["phi"]
    v0  = kz_params["v0"]
    a   = kz_params["a_lattice"]
    c   = kz_params["c_lattice"]
    dkz = bz_period(c)

    erlab.io.set_loader("da30")
    files = sorted(
        [p for p in kz_scan_dir.iterdir() if p.suffix.lower() in (".ibw", ".pxt")],
        key=lambda p: _hv_from_filename(p) or 0.0,
    )

    records = []
    for f in files:
        hv_file = _hv_from_filename(f)
        if hv_file is None:
            continue
        try:
            da = erlab.io.load(str(f))
            hv = float(da.attrs.get("hv", hv_file))
            ef_kin = hv - phi
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                da_be = da.assign_coords(eV=da.eV - ef_kin, hv=hv, xi=0.0)
            da_be.attrs["configuration"] = 1
            da_be.kspace.work_function = phi

            try:
                da_be.kspace.work_function = phi
                da_k = da_be.kspace.convert()
                kx  = np.asarray(da_k.kx.values, dtype=float) * a / np.pi
                ev  = np.asarray(da_k.eV.values,  dtype=float)
                dat = np.asarray(da_k.values,       dtype=float)
            except Exception:
                kx  = np.asarray(da_be.coords.get("kx", da_be.coords["xi"]).values, dtype=float)
                ev  = np.asarray(da_be.eV.values, dtype=float)
                dat = np.asarray(da_be.values,     dtype=float)

            if dat.ndim == 2:
                pass
            elif dat.ndim == 3:
                dat = np.squeeze(dat)

            # MDC à EF — détecter l'ordre des axes (nE, nK) ou (nK, nE)
            mask = np.abs(ev) <= ef_window
            if mask.sum() == 0:
                mask[np.argmin(np.abs(ev))] = True
            if dat.shape[0] == len(ev):        # (nE, nK)
                mdc_ef = np.nanmean(dat[mask, :], axis=0)
            else:                               # (nK, nE)
                mdc_ef = np.nanmean(dat[:, mask], axis=1)

            kz = kz_ang(hv, phi, v0)
            records.append({"hv": hv, "kz": kz, "kx": kx, "mdc": mdc_ef})
            print(f"  {f.name}  hν={hv:.1f}  kz={kz:.3f} Å⁻¹  {nearest_hs_label(kz, c)}")
        except Exception as exc:
            print(f"  ⚠ {f.name} : {exc}")

    if not records:
        print("Aucun fichier chargé.")
        return

    records.sort(key=lambda r: r["kz"])

    # ── Axe kx commun ────────────────────────────────────────────────────
    kx_min = max(r["kx"].min() for r in records)
    kx_max = min(r["kx"].max() for r in records)
    kx_com = np.linspace(kx_min, kx_max, 300)
    kz_arr = np.array([r["kz"] for r in records])

    # Matrice FS kx×kz
    fs_mat = np.zeros((len(records), len(kx_com)))
    for i, r in enumerate(records):
        mdc_interp = np.interp(kx_com, r["kx"], r["mdc"])
        lo, hi = np.nanpercentile(mdc_interp, [2, 98])
        fs_mat[i] = (mdc_interp - lo) / (hi - lo + 1e-12)

    # ── Figure principale : FS kx×kz ─────────────────────────────────────
    fig, ax_fs = plt.subplots(figsize=(9, 7))

    # Carte kx×kz
    extent = [kx_com[0], kx_com[-1], kz_arr[0], kz_arr[-1]]
    im = ax_fs.imshow(
        fs_mat, origin="lower", aspect="auto",
        extent=extent, cmap="inferno", vmin=0, vmax=1,
    )
    plt.colorbar(im, ax=ax_fs, label="Intensité à EF (normalisée)", pad=0.12)

    # Lignes Γ et Z
    for n in range(0, 20):
        kg  = n * dkz
        kz_ = (n + 0.5) * dkz
        if kz_arr[0] <= kg <= kz_arr[-1]:
            ax_fs.axhline(kg,  color="steelblue", lw=1.0, ls="-",  alpha=0.6)
            ax_fs.text(kx_com[-1]*0.95, kg, f"Γ({n})",
                       ha="right", va="bottom", fontsize=8, color="steelblue")
        if kz_arr[0] <= kz_ <= kz_arr[-1]:
            ax_fs.axhline(kz_, color="tomato",    lw=1.0, ls="--", alpha=0.6)
            ax_fs.text(kx_com[-1]*0.95, kz_, f"Z({n})",
                       ha="right", va="bottom", fontsize=8, color="tomato")

    # Points de haute symétrie en kx
    for kx_hs, name in [(-1.0, "X'"), (0.0, "Γ"), (1.0, "X")]:
        ax_fs.axvline(kx_hs, color="white", lw=0.8, ls=":", alpha=0.5)
        ax_fs.text(kx_hs, kz_arr[-1], name, color="white", fontsize=8,
                   ha="center", va="top", fontweight="bold")

    ax_fs.set_xlabel("kx (π/a)", fontsize=11)
    ax_fs.set_ylabel("kz (Å⁻¹)", fontsize=11)
    ax_fs.set_title(
        f"BaNi₂As₂ — FS kx×kz  (direction ΓX)\n"
        f"φ={phi} eV  V₀={v0:.1f} eV  Δkz=2π/c={dkz:.3f} Å⁻¹\n"
        f"Intégration ±{ef_window*1000:.0f} meV autour de EF",
        fontsize=10,
    )

    # Axe hν à droite (twinx sur kz)
    ax_hv = ax_fs.twinx()
    ax_hv.set_ylim(kz_arr[0], kz_arr[-1])
    # Interpolation linéaire kz → hν pour placer les ticks proprement
    hv_arr_sorted = np.array([r["hv"] for r in records])
    hv_ticks = np.arange(np.ceil(hv_arr_sorted.min() / 5) * 5,
                         hv_arr_sorted.max() + 1, 5)
    kz_for_hv = np.interp(hv_ticks, hv_arr_sorted, kz_arr)
    ax_hv.set_yticks(kz_for_hv)
    ax_hv.set_yticklabels([f"{h:.0f}" for h in hv_ticks], fontsize=8)
    ax_hv.set_ylabel("hν (eV)", fontsize=10)

    plt.tight_layout()
    out_path = out_dir / "fs_kxkz.png"
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"\nFS kx×kz → {out_path}")

    # ── Figure secondaire : coupes kx individuelles ────────────────────────
    n = len(records)
    ncols = 6
    nrows = (n + ncols - 1) // ncols
    fig2, axes2 = plt.subplots(nrows, ncols,
                               figsize=(3 * ncols, 2.5 * nrows),
                               squeeze=False)
    for idx, r in enumerate(records):
        ax = axes2[idx // ncols][idx % ncols]
        mdc = np.interp(kx_com, r["kx"], r["mdc"])
        lo, hi = np.nanpercentile(mdc, [2, 98])
        mdc_n = (mdc - lo) / (hi - lo + 1e-12)
        ax.plot(kx_com, mdc_n, lw=1.0, color="steelblue")
        ax.fill_between(kx_com, 0, mdc_n, alpha=0.3, color="steelblue")
        ax.axvline(0, color="gray", lw=0.5, ls="--")
        ax.set_title(f"hν={r['hv']:.0f}\n{nearest_hs_label(r['kz'], c)}", fontsize=6)
        ax.set_xticks([-1, 0, 1])
        ax.set_xticklabels(["-1", "0", "1"], fontsize=6)
        ax.set_yticks([])
    for idx in range(n, nrows * ncols):
        axes2[idx // ncols][idx % ncols].set_visible(False)
    fig2.suptitle("MDC à EF par hν  (direction ΓX, kx en π/a)", fontsize=11)
    plt.tight_layout()
    out_path2 = out_dir / "fs_kxkz_slices.png"
    fig2.savefig(out_path2, dpi=dpi, bbox_inches="tight")
    plt.close(fig2)
    print(f"Coupes MDC à EF → {out_path2}")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Surfaces de Fermi ARPES BaNi₂As₂",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--mode", choices=["fastmap", "kxkz"], required=True,
                   help="fastmap = FS kx×ky (deflector scans) | kxkz = FS kx×kz (scan kz)")
    p.add_argument("--data-dir",    type=Path, default=None,
                   help="Dossier des données BM (mode fastmap)")
    p.add_argument("--logbook",     type=Path, default=None,
                   help="CSV logbook (mode fastmap)")
    p.add_argument("--kz-scan-dir", type=Path, default=None,
                   help="Dossier scan kz (mode kxkz)")
    p.add_argument("--kz-params",   type=Path, default=None,
                   help="JSON généré par kz_calc.py (phi, v0, c, a)")
    p.add_argument("--ef-window",   type=float, default=0.030,
                   help="Intégration autour EF [eV] (défaut=0.030)")
    p.add_argument("--out-dir",     type=Path, default=Path("fs_out"),
                   help="Dossier de sortie (défaut=fs_out)")
    p.add_argument("--dpi",         type=int,   default=130)
    return p.parse_args()


def main():
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    erlab.io.set_loader("da30")

    kz_params = load_kz_params(args.kz_params)

    if args.mode == "fastmap":
        if args.data_dir is None:
            raise SystemExit("--mode fastmap requiert --data-dir")
        plot_fs_fastmaps(
            args.data_dir,
            kz_params, args.ef_window,
            args.out_dir, args.dpi,
        )

    elif args.mode == "kxkz":
        if args.kz_scan_dir is None:
            raise SystemExit("--mode kxkz requiert --kz-scan-dir")
        plot_fs_kxkz(
            args.kz_scan_dir, kz_params,
            args.ef_window, args.out_dir, args.dpi,
        )


if __name__ == "__main__":
    main()
