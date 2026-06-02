"""Result dialog for FS pocket characterization."""
from __future__ import annotations

from PyQt6.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QFormLayout,
    QLabel,
    QVBoxLayout,
)


class PocketResultDialog(QDialog):
    def __init__(self, parent, pocket: dict, *, allow_delete: bool = False):
        super().__init__(parent)
        self.setWindowTitle("Poche FS")
        self.resize(380, 320)
        self.delete_requested = False
        lay = QVBoxLayout(self)

        title = QLabel(str(pocket.get("hs_label_nearest") or "Poche"))
        title.setStyleSheet("font-size:18px; font-weight:bold;")
        lay.addWidget(title)

        unc = pocket.get("uncertainty") or {}
        if pocket.get("n_bootstrap_valid"):
            sub = QLabel(
                f"Bootstrap {int(pocket['n_bootstrap_valid'])}/"
                f"{int(pocket.get('n_bootstrap_total', 0))} valides"
            )
            sub.setStyleSheet("color:#9cf; font-size:11px;")
            lay.addWidget(sub)

        form = QFormLayout()
        form.addRow("Topologie :", QLabel(_topology_text(pocket)))
        form.addRow("Level :", QLabel(_fmt(pocket.get("level"), "{:.4f}")))
        form.addRow("Aire BZ :", QLabel(_fmt_unc(pocket.get("area_pct_bz"), unc.get("area_pct_bz"), "{:.2f}", " %")))
        form.addRow("Aire :", QLabel(_fmt_unc(pocket.get("area_inv_a2"), unc.get("area_inv_a2"), "{:.4f}", " (π/a)^2")))
        form.addRow("kF moyen :", QLabel(_fmt_unc(pocket.get("kF_mean"), unc.get("kF_mean"), "{:.4f}", " π/a")))
        form.addRow("Ellipse a :", QLabel(_fmt_unc(pocket.get("kF_a"), unc.get("kF_a"), "{:.4f}", " π/a")))
        form.addRow("Ellipse b :", QLabel(_fmt_unc(pocket.get("kF_b"), unc.get("kF_b"), "{:.4f}", " π/a")))
        form.addRow("Angle :", QLabel(_fmt_unc(pocket.get("ellipse_angle_deg"), unc.get("ellipse_angle_deg"), "{:.1f}", " °")))
        form.addRow("kF Γ-X :", QLabel(_fmt_unc(pocket.get("kF_gamma_x"), unc.get("kF_gamma_x"), "{:.4f}", " π/a")))
        form.addRow("kF Γ-M :", QLabel(_fmt_unc(pocket.get("kF_gamma_m"), unc.get("kF_gamma_m"), "{:.4f}", " π/a")))
        form.addRow("Aspect ratio :", QLabel(_fmt_unc(pocket.get("aspect_ratio"), unc.get("aspect_ratio"), "{:.3f}", "")))
        form.addRow("Excentricité :", QLabel(_fmt_unc(pocket.get("eccentricity"), unc.get("eccentricity"), "{:.3f}", "")))
        form.addRow("⟨1/R⟩ :", QLabel(_fmt_unc(pocket.get("curvature_mean"), unc.get("curvature_mean"), "{:.3f}", " (π/a)⁻¹")))
        form.addRow("Var(1/R) :", QLabel(_fmt_unc(pocket.get("curvature_var"), unc.get("curvature_var"), "{:.3f}", "")))
        form.addRow("n porteurs (2D) :", QLabel(_fmt_unc(pocket.get("n_carriers_2D"), unc.get("n_carriers_2D"), "{:.4f}", " /cell")))
        form.addRow("Centre :", QLabel(
            f"{float(pocket.get('centroid_kx', 0.0)):+.4f}, "
            f"{float(pocket.get('centroid_ky', 0.0)):+.4f}"
        ))
        form.addRow("HS proche :", QLabel(
            f"{pocket.get('hs_label_nearest') or '-'} "
            f"({_fmt(pocket.get('hs_distance'), '{:.4f} π/a')})"
        ))
        if pocket.get("mp_label"):
            form.addRow("MP :", QLabel(str(pocket.get("mp_label"))))
        lay.addLayout(form)
        dft = pocket.get("dft_compare") or {}
        if dft:
            sep = QLabel("— DFT 3D vs ARPES —")
            sep.setStyleSheet("color:#9cf; font-size:11px; font-weight:bold;")
            lay.addWidget(sep)
            f2 = QFormLayout()
            f2.addRow("ΔkF moyen :", QLabel(_fmt(dft.get("delta_kF_mean_pct"), "{:+.2f} %")))
            f2.addRow("Δ aire :", QLabel(_fmt(dft.get("delta_area_pct"), "{:+.2f} %")))
            f2.addRow("Hausdorff :", QLabel(_fmt(dft.get("hausdorff"), "{:.4f} π/a")))
            f2.addRow("Δ centre :", QLabel(_fmt(dft.get("centroid_shift"), "{:.4f} π/a")))
            f2.addRow("kz utilisé :", QLabel(_fmt(dft.get("kz_used_1_per_ang"), "{:.4f} 1/Å")))
            lay.addLayout(f2)
        elif pocket.get("dft_compare_error"):
            err = QLabel(str(pocket.get("dft_compare_error")))
            err.setStyleSheet("color:#fa6; font-size:10px;")
            lay.addWidget(err)

        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok)
        if allow_delete:
            btn_delete = buttons.addButton("Supprimer", QDialogButtonBox.ButtonRole.DestructiveRole)
            btn_delete.clicked.connect(self._delete)
        buttons.accepted.connect(self.accept)
        lay.addWidget(buttons)

    def _delete(self) -> None:
        self.delete_requested = True
        self.accept()


def _fmt(value, pattern: str) -> str:
    try:
        return pattern.format(float(value))
    except Exception:
        return "-"


def _fmt_unc(value, std, pattern: str, unit: str) -> str:
    try:
        v = float(value)
    except Exception:
        return "-"
    base = pattern.format(v)
    try:
        s = float(std)
    except Exception:
        s = None
    if s is None or not (s == s) or s <= 0.0:
        return f"{base}{unit}"
    return f"{base} ± {pattern.format(s)}{unit}"


def _topology_text(pocket: dict) -> str:
    topo = str(pocket.get("topology") or "unclear")
    conf = _fmt(pocket.get("topology_confidence"), "{:.2f}")
    label = {"electron": "electron", "hole": "hole", "unclear": "incertain"}.get(topo, topo)
    return f"{label}  conf={conf}"
