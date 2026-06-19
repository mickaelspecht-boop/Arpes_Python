from __future__ import annotations

from arpes.core.experience_log import build_experience_log
from arpes.core.session import FileEntry, FileMeta, FitParams


def test_bm_experience_log_reports_corrections_and_fit():
    entry = FileEntry(
        view_mode="EDCnorm",
        edcnorm=True,
        fit_params=FitParams(n_pairs=1),
        fit_result={
            "e_fitted": [-0.1, 0.0],
            "kF_plus": [[0.2, 0.21]],
            "kF_minus": [[-0.2, -0.21]],
            "sigma_kF_plus": [[0.003, 0.003]],
            "gamma_corrige": [[0.04, 0.05]],
            "grid_active": True,
            "distorted": False,
        },
        meta=FileMeta(scan_kind="bm", hv=60.0, direction="G-M", loader_label="CLS"),
    )
    entry.grid_correction = {"enabled": True, "strength": 0.5}

    text = build_experience_log(entry, name="BM5")

    assert "Processing log - BM5" in text
    assert "Signal kind: BM" in text
    assert "EDC normalization active" in text
    assert "Detector-grid correction active" in text
    assert "MDC fit result stored" in text
    assert "Lorentzian" in text


def test_fs_experience_log_reports_fs_transforms_and_pockets():
    entry = FileEntry(
        meta=FileMeta(scan_kind="fs", hv=80.0, loader_label="Solaris"),
        fs_center_kx=0.1,
        fs_center_ky=-0.2,
        fs_rotation_deg=12.0,
        fs_v0=13.5,
        fs_kz_plane="Z",
        fs_phi_c_deg=3.0,
        propagate_distortion_to_fs=True,
    )
    entry.fs_lattice = {"a": 4.1, "space_group": "I4/mmm"}
    entry.fs_pockets = [{"kF_mean": 0.42, "area_pct_bz": 5.5, "topology": "electron"}]

    text = build_experience_log(entry, name="FS1")

    assert "Signal kind: FS" in text
    assert "FS center" in text
    assert "k_centered = k_raw - k_gamma" in text
    assert "FS display rotation" in text
    assert "BM distortion propagation to FS volume: enabled" in text
    assert "FS pockets extracted: 1" in text
