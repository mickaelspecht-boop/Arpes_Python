"""
Loader pour les données ARPES du CLS (Canadian Light Source) / LNLS.

Deux formats sont supportés :

  Format FS (Fermi Surface) — fpath = répertoire :
      <fpath>/<prefix>_param.txt          — paramètres (avec ligne phi)
      <fpath>/<prefix>_Cycle_C_Step_S.txt — données 2D par step
      - Cycles  : répétitions → moyennées automatiquement
      - Steps   : positions phi → dimension 'tilt' du DataArray
      → DataArray 3D (tilt, eV, theta_par)

  Format BM (Band Map) — fpath = fichier de données :
      <dossier>/<prefix>_param.txt        — paramètres (sans ligne phi)
      <dossier>/<prefix>                  — fichier de données unique
      → DataArray 2D (eV, theta_par)

Paramètres manuels (depuis le logbook) :
    hv          : énergie photon en eV
    temperature : température en K
    azi         : angle azimutal en degrés
    pol         : polarisation (LH, LV, CL, CR)

Usage :
    from CLS import load_cls

    # FS (dossier contenant Cycle/Step)
    da = load_cls("./FS5", hv=90.0, temperature=17.0, pol="LH")
    # da.dims = ('tilt', 'eV', 'theta_par')

    # BM (fichier unique)
    da = load_cls("./BM1", hv=90.0, temperature=17.0, pol="LH")
    # da.dims = ('eV', 'theta_par')
"""

from __future__ import annotations

import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
import xarray as xr

from peaks.core.fileIO.base_data_classes.base_data_class import (
    BaseDataLoader,
    LOC_REGISTRY,
)
from peaks.core.fileIO.base_arpes_data_classes.base_arpes_data_class import (
    ARPESAnalyserAnglesMetadataModel,
    ARPESAnalyserMetadataModel,
    ARPESCalibrationModel,
    ARPESDeflectorMetadataModel,
    ARPESMetadataModel,
    ARPESScanMetadataModel,
    ARPESSlitMetadataModel,
    BaseARPESDataLoader,
    BaseManipulatorDataLoader,
    NamedAxisMetadataModel,
    ureg,
)


# ── Manipulateur ───────────────────────────────────────────────────────────────
class CLSManipulator(BaseManipulatorDataLoader):
    _manipulator_axes = {
        "polar": None, "tilt": None, "azi": None,
        "x1": None, "x2": None, "x3": None,
    }
    _manipulator_name_conventions = {
        "polar": None, "tilt": None, "azi": None,
        "x1": None, "x2": None, "x3": None,
    }
    _manipulator_sign_conventions = {}


# ── Loader ─────────────────────────────────────────────────────────────────────
class CLSDataLoader(CLSManipulator, BaseARPESDataLoader):
    """
    Loader ARPES CLS/LNLS.

    Cycles  → moyennés (répétitions)
    Steps   → dimension 'tilt' (scan phi)
    param.txt → axes eV, theta_par, phi par step, positions manipulateur
    """

    _loc_name        = "CLS"
    _loc_description = "Loader ARPES CLS/LNLS (format txt custom)"
    _loc_url         = "https://www.lightsource.ca"
    _metadata_cache: dict = {}
    _analyser_slit_angle       = 0.0
    _analyser_sign_conventions = {}
    _analyser_name_conventions = {"deflector_parallel": None, "deflector_perp": None}

    # ── Parser param.txt ──────────────────────────────────────────────────────
    @staticmethod
    def _parse_param(folder: Path, prefix: str) -> dict:
        """
        Lit le param.txt et retourne un dict avec tous les paramètres.

        Structure du param.txt :
            Ligne 1 : Pass energy
            Ligne 2 : Lens mode
            Ligne 3 : Acquisition mode
            Ligne 4 : Central Energy
            Ligne 5 : Dwell Time
            Ligne 6 : Energy min + delta
            Ligne 7 : Angle min + delta
            Ligne 8 : positions phi pour chaque Step
            Ligne 9 : JSON positions manipulateur
        """
        param_file = folder / f"{prefix}_param.txt"
        txt = param_file.read_text()
        lines = txt.strip().splitlines()

        em      = re.search(r"Energy min:\s*([-\d.]+);\s*Energy delta:\s*([-\d.]+)", txt)
        am      = re.search(r"Angle min:\s*([-\d.]+);\s*Angle delta:\s*([-\d.]+)", txt)
        pe      = re.search(r"Pass energy:\s*([\d.]+)", txt)
        lm      = re.search(r"Lens mode:\s*(\S+)", txt)
        dt      = re.search(r"Dwell Time:\s*([\d.]+)", txt)
        am_mode = re.search(r"Acquisition mode:\s*(\d+)", txt)

        # Phi : première ligne contenant uniquement des nombres (absent pour BM)
        phi_line = next(
            (l for l in lines if re.match(r"^\s*-?[\d.]+(\s+-?[\d.]+)+\s*$", l)),
            None
        )
        phi_values = (
            np.array([float(x) for x in phi_line.split()]) if phi_line is not None else None
        )

        # JSON manipulateur : ligne commençant par {
        json_line = next((l for l in lines if l.strip().startswith("{")), None)
        try:
            motors = json.loads(json_line).get("d", {}) if json_line else {}
        except (json.JSONDecodeError, TypeError, AttributeError):
            motors = {}

        return {
            "energy_min":       float(em.group(1)),
            "energy_delta":     float(em.group(2)),
            "angle_min":        float(am.group(1)),
            "angle_delta":      float(am.group(2)),
            "pass_energy":      float(pe.group(1)) if pe else None,
            "lens_mode":        lm.group(1)        if lm else None,
            "dwell_ms":         float(dt.group(1)) if dt else None,
            "acquisition_mode": int(am_mode.group(1)) if am_mode else None,
            "phi_values":       phi_values,
            "polar":            motors.get("P", {}).get("position", 0.0),
            "tilt_ref":         motors.get("T", {}).get("position", 0.0),
            "x":                motors.get("X", {}).get("position", 0.0),
            "y":                motors.get("Y", {}).get("position", 0.0),
            "z":                motors.get("Z", {}).get("position", 0.0),
        }

    # ── _load_data ─────────────────────────────────────────────────────────────
    @classmethod
    def _load_data(cls, fpath: str, lazy: bool, **kwargs):
        """
        Deux formats supportés :

        Format FS (dossier) — fpath est un répertoire :
            <fpath>/<prefix>_param.txt          (avec ligne phi)
            <fpath>/<prefix>_Cycle_C_Step_S.txt (multiples)
            → DataArray 3D (tilt, eV, theta_par) ou 2D si 1 seul step

        Format BM (fichier) — fpath est un fichier de données :
            <dossier>/<prefix>_param.txt        (sans ligne phi)
            <dossier>/<prefix>                  (fichier de données)
            → DataArray 2D (eV, theta_par)
        """
        hv          = float(kwargs.pop("hv",          21.2))
        temperature = float(kwargs.pop("temperature", float("nan")))
        azi         = float(kwargs.pop("azi",         0.0))
        pol         = str(kwargs.pop("pol",           "LH"))

        path = Path(fpath)

        # ── Détection du format ───────────────────────────────────────────────
        if path.is_file():
            return cls._load_bm(path, fpath, hv, temperature, azi, pol)
        else:
            return cls._load_fs(path, fpath, hv, temperature, azi, pol)

    @classmethod
    def _load_bm(cls, path: Path, fpath: str,
                 hv: float, temperature: float, azi: float, pol: str) -> dict:
        """Charge un fichier BM unique (2D : eV × theta_par)."""
        folder = path.parent
        prefix = path.name  # pas d'extension (ex : BM1)

        param_file = folder / f"{prefix}_param.txt"
        if not param_file.exists():
            raise FileNotFoundError(f"Paramètres introuvables : {param_file}")

        p = cls._parse_param(folder, prefix)
        data = np.loadtxt(path)
        n_eV, n_theta = data.shape

        eV        = p["energy_min"] + np.arange(n_eV)    * p["energy_delta"]
        theta_par = p["angle_min"]  + np.arange(n_theta) * p["angle_delta"]

        cls._metadata_cache[fpath] = {
            "hv":               hv,
            "temperature":      temperature,
            "pol":              pol,
            "polar":            float(p["polar"]),
            "tilt_ref":         float(p["tilt_ref"]),
            "azi":              azi,
            "x":                float(p["x"]),
            "y":                float(p["y"]),
            "z":                float(p["z"]),
            "pass_energy":      p["pass_energy"],
            "lens_mode":        p["lens_mode"],
            "dwell_ms":         p["dwell_ms"],
            "acquisition_mode": p["acquisition_mode"],
            "eV_min":           float(eV[0]),
            "eV_max":           float(eV[-1]),
            "n_steps":          1,
            "n_cycles":         1,
        }

        print(
            f"✓ {prefix} (BM)\n"
            f"  eV    [{eV[0]:.4f} → {eV[-1]:.4f} eV]\n"
            f"  theta [{theta_par[0]:.3f} → {theta_par[-1]:.3f}°]\n"
            f"  hv={hv} eV  T={temperature} K  pol={pol}"
            f"  polar={p['polar']:.3f}°  tilt={p['tilt_ref']:.3f}°"
        )

        return {
            "spectrum": data.astype(np.float32),
            "dims":     ["eV", "theta_par"],
            "coords":   {"eV": eV, "theta_par": theta_par},
            "units":    {"eV": "eV", "theta_par": "deg", "spectrum": "counts"},
        }

    @classmethod
    def _load_fs(cls, folder: Path, fpath: str,
                 hv: float, temperature: float, azi: float, pol: str) -> dict:
        """
        Charge un dataset FS multi-steps (3D : tilt × eV × theta_par).
        Les cycles sont moyennés automatiquement.
        """
        param_files = sorted(folder.glob("*_param.txt"))
        if not param_files:
            raise FileNotFoundError(f"Aucun *_param.txt dans {folder}")
        prefix = param_files[0].name.removesuffix("_param.txt")

        p = cls._parse_param(folder, prefix)

        # Inventaire {step_idx: [fichier_cycle0, fichier_cycle1, ...]}
        all_files = sorted(
            folder.glob(f"{prefix}_Cycle_*_Step_*.txt"),
            key=lambda f: (
                int(re.search(r"Cycle_(\d+)", f.name).group(1)),
                int(re.search(r"Step_(\d+)",  f.name).group(1)),
            ),
        )
        if not all_files:
            raise FileNotFoundError(f"Aucun fichier Cycle/Step dans {folder}")

        steps_dict = defaultdict(list)
        for f in all_files:
            step_idx = int(re.search(r"Step_(\d+)", f.name).group(1))
            steps_dict[step_idx].append(f)

        step_ids = sorted(steps_dict.keys())
        n_steps  = len(step_ids)
        n_cycles = max(len(v) for v in steps_dict.values())

        # Dimensions depuis le premier fichier
        sample = np.loadtxt(steps_dict[step_ids[0]][0])
        n_eV, n_theta = sample.shape

        eV        = p["energy_min"] + np.arange(n_eV)    * p["energy_delta"]
        theta_par = p["angle_min"]  + np.arange(n_theta) * p["angle_delta"]

        # Coordonnées phi depuis param
        phi_vals = p["phi_values"]
        if phi_vals is not None and len(phi_vals) >= n_steps:
            phi_coords = np.array([phi_vals[s] for s in step_ids])
        else:
            if phi_vals is not None:
                print(f"⚠ phi_values ({len(phi_vals)}) < n_steps ({n_steps}) — indices utilisés")
            phi_coords = np.array(step_ids, dtype=float)

        # Moyenne cycles → (n_steps, n_eV, n_theta)
        data3d = np.zeros((n_steps, n_eV, n_theta), dtype=np.float64)
        for i, step_idx in enumerate(step_ids):
            cycle_arrays = np.stack([np.loadtxt(f) for f in steps_dict[step_idx]], axis=0)
            data3d[i] = cycle_arrays.mean(axis=0)

        cls._metadata_cache[fpath] = {
            "hv":               hv,
            "temperature":      temperature,
            "pol":              pol,
            "polar":            float(p["polar"]),
            "tilt_ref":         float(p["tilt_ref"]),
            "azi":              azi,
            "x":                float(p["x"]),
            "y":                float(p["y"]),
            "z":                float(p["z"]),
            "pass_energy":      p["pass_energy"],
            "lens_mode":        p["lens_mode"],
            "dwell_ms":         p["dwell_ms"],
            "acquisition_mode": p["acquisition_mode"],
            "eV_min":           float(eV[0]),
            "eV_max":           float(eV[-1]),
            "n_steps":          n_steps,
            "n_cycles":         n_cycles,
        }

        print(
            f"✓ {prefix} : {n_cycles} cycle(s) moyenné(s) × {n_steps} steps\n"
            f"  phi   [{phi_coords[0]:.3f} → {phi_coords[-1]:.3f}°]\n"
            f"  eV    [{eV[0]:.4f} → {eV[-1]:.4f} eV]\n"
            f"  theta [{theta_par[0]:.3f} → {theta_par[-1]:.3f}°]\n"
            f"  hv={hv} eV  T={temperature} K  pol={pol}  polar={p['polar']:.3f}°  tilt_ref={p['tilt_ref']:.3f}°"
        )

        if n_steps == 1:
            return {
                "spectrum": data3d[0].astype(np.float32),
                "dims":     ["eV", "theta_par"],
                "coords":   {"eV": eV, "theta_par": theta_par},
                "units":    {"eV": "eV", "theta_par": "deg", "spectrum": "counts"},
            }
        else:
            return {
                "spectrum": data3d.astype(np.float32),
                "dims":     ["tilt", "eV", "theta_par"],
                "coords":   {"tilt": phi_coords, "eV": eV, "theta_par": theta_par},
                "units":    {"tilt": "deg", "eV": "eV", "theta_par": "deg", "spectrum": "counts"},
            }

    # ── _load_metadata ─────────────────────────────────────────────────────────
    @classmethod
    def _load_metadata(cls, fpath: str):
        c = cls._metadata_cache.get(fpath, {})

        def Q(v, u):
            if v is None or (isinstance(v, float) and np.isnan(v)):
                return None
            try:
                return float(v) * ureg(u)
            except Exception:
                return v

        return {
            # Photon
            "photon_hv":                   Q(c.get("hv"),          "eV"),
            "photon_polarisation":         c.get("pol"),
            # Température
            "temperature_sample":          Q(c.get("temperature"),  "K"),
            # Manipulateur — positions actuelles
            "manipulator_polar":           Q(c.get("polar"),        "deg"),
            "manipulator_tilt":            Q(c.get("tilt_ref"),     "deg"),
            "manipulator_azi":             Q(c.get("azi"),          "deg"),
            "manipulator_x1":              Q(c.get("x"),            "mm"),
            "manipulator_x2":              Q(c.get("y"),            "mm"),
            "manipulator_x3":              Q(c.get("z"),            "mm"),
            # Références émission normale — identiques aux positions courantes
            # (le framework peaks exige ces clés ; on répète les mêmes valeurs
            # car le CLS ne fournit pas de référence séparée)
            "manipulator_norm_polar":      Q(c.get("polar"),        "deg"),
            "manipulator_norm_tilt":       Q(c.get("tilt_ref"),     "deg"),
            "manipulator_norm_azi":        Q(c.get("azi"),          "deg"),
            "manipulator_polar_reference": Q(c.get("polar"),        "deg"),
            "manipulator_tilt_reference":  Q(c.get("tilt_ref"),     "deg"),
            "manipulator_azi_reference":   Q(c.get("azi"),          "deg"),
            # Analyseur
            "analyser_PE":                 Q(c.get("pass_energy"),  "eV"),
            "analyser_lens_mode":          c.get("lens_mode"),
            "analyser_dwell":              Q(c.get("dwell_ms"),     "ms"),
            "analyser_acquisition_mode":   str(c.get("acquisition_mode"))
                                           if c.get("acquisition_mode") is not None else None,
            "analyser_eV":                 np.asarray([
                                               c.get("eV_min", 0),
                                               c.get("eV_max", 0),
                                           ]) * ureg("eV"),
            "analyser_eV_type":            "Kinetic",
            "analyser_deflector_parallel": Q(0.0, "deg"),
            "analyser_deflector_perp":     Q(0.0, "deg"),
            "analyser_polar":              Q(0.0, "deg"),
            "analyser_tilt":               Q(0.0, "deg"),
            "analyser_azi":                Q(0.0, "deg"),
        }

    # ── _parse_analyser_metadata ───────────────────────────────────────────────
    @classmethod
    def _parse_analyser_metadata(cls, metadata_dict):
        m = metadata_dict
        arpes = ARPESMetadataModel(
            analyser=ARPESAnalyserMetadataModel(
                model="Scienta",
                slit=ARPESSlitMetadataModel(width=None, identifier=None),
            ),
            scan=ARPESScanMetadataModel(
                eV=m.get("analyser_eV"),
                step_size=None,
                PE=m.get("analyser_PE"),
                sweeps=None,
                dwell=m.get("analyser_dwell"),
                lens_mode=m.get("analyser_lens_mode"),
                acquisition_mode=m.get("analyser_acquisition_mode"),
                eV_type=m.get("analyser_eV_type", "Kinetic"),
            ),
            angles=ARPESAnalyserAnglesMetadataModel(
                polar=m.get("analyser_polar"),
                tilt=m.get("analyser_tilt"),
                azi=m.get("analyser_azi"),
            ),
            deflector=ARPESDeflectorMetadataModel(
                parallel=NamedAxisMetadataModel(
                    value=m.get("analyser_deflector_parallel"), local_name=None),
                perp=NamedAxisMetadataModel(
                    value=m.get("analyser_deflector_perp"), local_name=None),
            ),
        )
        return {
            "_calibration": ARPESCalibrationModel(),
            "_analyser":    arpes,
        }, ["analyser_PE", "analyser_dwell", "analyser_eV"]


# ── Enregistrement ─────────────────────────────────────────────────────────────
LOC_REGISTRY["CLS"] = CLSDataLoader


# ── Fonction utilitaire ────────────────────────────────────────────────────────
def load_cls(folder: str,
             hv: float = float("nan"),
             temperature: float = float("nan"),
             azi: float = 0.0,
             pol: str = "LH") -> xr.DataArray:
    """
    Charge des données ARPES CLS/LNLS.

    Les cycles sont moyennés automatiquement.
    Les steps deviennent la dimension 'tilt' (scan phi).
    polar, tilt_ref, x, y, z sont lus depuis le param.txt.

    Paramètres
    ----------
    folder      : dossier contenant *_param.txt et *_Cycle_*_Step_*.txt
    hv          : énergie photon en eV          (depuis le logbook)
    temperature : température en K              (depuis le logbook)
    azi         : angle azimutal en degrés      (défaut 0)
    pol         : polarisation photon           (LH, LV, CL, CR)

    Exemple
    -------
    >>> import CLS
    >>> da = CLS.load_cls("./FS", hv=59.0, temperature=17.0, pol="LH")
    >>> print(da.dims)         # ('tilt', 'eV', 'theta_par')
    >>> print(da.tilt.values)  # positions phi de chaque step
    >>> da.mean("tilt").plot() # carte 2D moyennée
    """
    import peaks as pks
    return pks.load(str(folder), loc="CLS",
                    hv=hv, temperature=temperature, azi=azi, pol=pol)