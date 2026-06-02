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
        self.resize(360, 260)
        self.delete_requested = False
        lay = QVBoxLayout(self)

        title = QLabel(str(pocket.get("hs_label_nearest") or "Poche"))
        title.setStyleSheet("font-size:18px; font-weight:bold;")
        lay.addWidget(title)

        form = QFormLayout()
        form.addRow("Topologie :", QLabel(_topology_text(pocket)))
        form.addRow("Level :", QLabel(_fmt(pocket.get("level"), "{:.4f}")))
        form.addRow("Aire BZ :", QLabel(_fmt(pocket.get("area_pct_bz"), "{:.2f} %")))
        form.addRow("Aire :", QLabel(_fmt(pocket.get("area_inv_a2"), "{:.4f} (π/a)^2")))
        form.addRow("kF moyen :", QLabel(_fmt(pocket.get("kF_mean"), "{:.4f} π/a")))
        form.addRow("Ellipse a :", QLabel(_fmt(pocket.get("kF_a"), "{:.4f} π/a")))
        form.addRow("Ellipse b :", QLabel(_fmt(pocket.get("kF_b"), "{:.4f} π/a")))
        form.addRow("Angle :", QLabel(_fmt(pocket.get("ellipse_angle_deg"), "{:.1f} °")))
        form.addRow("kF Γ-X :", QLabel(_fmt(pocket.get("kF_gamma_x"), "{:.4f} π/a")))
        form.addRow("kF Γ-M :", QLabel(_fmt(pocket.get("kF_gamma_m"), "{:.4f} π/a")))
        form.addRow("Aspect ratio :", QLabel(_fmt(pocket.get("aspect_ratio"), "{:.3f}")))
        form.addRow("Excentricité :", QLabel(_fmt(pocket.get("eccentricity"), "{:.3f}")))
        form.addRow("⟨1/R⟩ :", QLabel(_fmt(pocket.get("curvature_mean"), "{:.3f} (π/a)⁻¹")))
        form.addRow("Var(1/R) :", QLabel(_fmt(pocket.get("curvature_var"), "{:.3f}")))
        form.addRow("n porteurs (2D) :", QLabel(_fmt(pocket.get("n_carriers_2D"), "{:.4f} /cell")))
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


def _topology_text(pocket: dict) -> str:
    topo = str(pocket.get("topology") or "unclear")
    conf = _fmt(pocket.get("topology_confidence"), "{:.2f}")
    label = {"electron": "electron", "hole": "hole", "unclear": "incertain"}.get(topo, topo)
    return f"{label}  conf={conf}"
