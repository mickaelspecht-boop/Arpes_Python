"""Tests d'intégration sur fichiers ARPES réels.

But : verrouiller le contrat de sortie de `load_arpes` pour chaque format
supporté, sur de vrais fichiers présents dans `~/Documents/Stage_M2/Code/...`.
Chaque test est `skip`é proprement si le fichier de référence est absent
(autre machine, autre arborescence) — pas de faux échec.

Pour étendre : ajouter une entrée dans `FIXTURES` ci-dessous. Les invariants
testés sont volontairement minimaux et stables (axes, dims, hv, format) ;
les détails de calibration (EF, kx) sont laissés aux tests unitaires.
"""

from __future__ import annotations

import unittest
from pathlib import Path

import numpy as np

from arpes.io.loaders import load_arpes, detect_format


# Racine des données (résolue dynamiquement, jamais en dur dans une assertion)
DATA_ROOT = Path.home() / "Documents" / "Stage_M2" / "Code"


# Chaque entrée : (chemin relatif depuis DATA_ROOT, format attendu, dims, kwargs)
# Les kwargs reproduisent ce que l'app passe au loader pour ce type de fichier.
FIXTURES = [
    {
        "label": "Solaris BM .pxt (BaNi2As2_0015)",
        "path": "BaNi2As2_/BaNi2As2_0015.pxt",
        "format": "solaris_da30",
        "ndim": 2,
        "kwargs": {"work_func": 4.5, "ef_offset": 0.0},
    },
    {
        "label": "Solaris BM .ibw (BaNi2As2_0015 fixed cut)",
        "path": "BaNi2As2_/BaNi2As2_0015fixed cut.ibw",
        "format": "solaris_da30",
        "ndim": 2,
        "kwargs": {"work_func": 4.5, "ef_offset": 0.0},
    },
    {
        "label": "Solaris FS .zip (BaNi2As2_0001)",
        "path": "BaNi2As2_/BaNi2As2_0001.zip",
        "format": "solaris_da30",
        "ndim": 2,  # data 2D (FS volume vit dans metadata['fs_data'])
        "kwargs": {"work_func": 4.5, "ef_offset": 0.0},
        "expect_fs": True,
    },
    {
        "label": "BESSY/SES .ibw (Ba1220009w)",
        "path": "Ba122/Ba1220009w_Band Map B122_009.ibw",
        "format": "bessy_ses_ibw",
        "ndim": 2,
        "kwargs": {"work_func": 4.031, "ef_offset": 0.0, "hv": 100.0},
    },
    {
        "label": "CLS texte (BM1)",
        "path": "Ba122_C05_2/BM1",
        "format": None,  # le format CLS n'est pas dans le registry actuel
        "ndim": 2,
        "kwargs": {"work_func": 4.5, "ef_offset": 0.0, "hv": 100.0},
        "skip_load": True,  # on ne teste que detect_format
    },
]


def _resolve(fix: dict) -> Path | None:
    p = DATA_ROOT / fix["path"]
    return p if p.exists() else None


class TestLoadersIntegration(unittest.TestCase):
    """Charge chaque fichier de référence et vérifie les invariants minimums."""

    def _check_common_invariants(self, ds, fix: dict) -> None:
        # axes énergie
        self.assertEqual(ds.energy.ndim, 1, f"{fix['label']}: energy must be 1D")
        self.assertGreater(len(ds.energy), 1, f"{fix['label']}: energy axis trop court")
        self.assertTrue(np.all(np.isfinite(ds.energy)), f"{fix['label']}: energy contient NaN/Inf")
        # data
        self.assertEqual(
            ds.data.ndim, fix["ndim"],
            f"{fix['label']}: data.ndim={ds.data.ndim} attendu {fix['ndim']}",
        )
        self.assertTrue(np.any(np.isfinite(ds.data)), f"{fix['label']}: data tout NaN")
        # hv strictement positif (anti-régression du fix Solaris hv≤0)
        if ds.hv is not None:
            self.assertGreater(
                float(ds.hv), 0,
                f"{fix['label']}: hv={ds.hv} doit être > 0 (le loader doit refuser 0)",
            )
        # axe k
        if ds.kx is not None:
            self.assertEqual(ds.kx.ndim, 1, f"{fix['label']}: kx must be 1D")
            self.assertEqual(
                ds.data.shape[0], len(ds.kx),
                f"{fix['label']}: data.shape[0]={ds.data.shape[0]} != len(kx)={len(ds.kx)}",
            )
        # FS volume si attendu
        if fix.get("expect_fs"):
            fs = ds.metadata.get("fs_data")
            self.assertIsNotNone(fs, f"{fix['label']}: fs_data absent du metadata")
            self.assertEqual(np.asarray(fs).ndim, 3, f"{fix['label']}: fs_data doit être 3D")
        # contrat metadata
        self.assertIn("loader_label", ds.metadata)
        self.assertIn("lab", ds.metadata)
        self.assertIn("energy_axis", ds.metadata)
        self.assertIn("pass_energy_eV", ds.metadata)
        self.assertGreater(
            float(ds.metadata["pass_energy_eV"]), 0,
            f"{fix['label']}: pass_energy_eV doit être présent et > 0",
        )


def _make_test(fix: dict):
    def test(self):
        path = _resolve(fix)
        if path is None:
            self.skipTest(f"Fixture absente : {fix['path']} (machine/arbo différente)")
        # 1) detect_format si attendu
        if fix.get("format") is not None:
            fmt = detect_format(path)
            self.assertEqual(
                fmt, fix["format"],
                f"{fix['label']}: detect_format={fmt!r} attendu {fix['format']!r}",
            )
        # 2) chargement (skip si loader CLS pas dans registry)
        if fix.get("skip_load"):
            self.skipTest(f"{fix['label']}: chargement non couvert (loader hors registry)")
        try:
            ds = load_arpes(str(path), **fix["kwargs"])
        except ImportError as e:
            self.skipTest(f"{fix['label']}: dépendance manquante ({e})")
        self._check_common_invariants(ds, fix)
    test.__name__ = "test_" + fix["label"].lower().replace(" ", "_").replace("/", "_").replace(".", "_")
    test.__doc__ = f"Charge et vérifie : {fix['label']}"
    return test


# Greffe dynamique des tests sur la classe
for _fix in FIXTURES:
    setattr(TestLoadersIntegration, _make_test(_fix).__name__, _make_test(_fix))


if __name__ == "__main__":
    unittest.main()
