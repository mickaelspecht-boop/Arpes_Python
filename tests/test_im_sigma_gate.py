"""Im Σ(E) button is gated on a fit existing (Qt offscreen)."""
import pytest

pytest.importorskip("PyQt6")
from PyQt6.QtWidgets import QApplication

from arpes.ui.widgets.params import FitParamsPanel

app = QApplication.instance() or QApplication([])


def test_im_sigma_renamed_and_gated_by_fit():
    p = FitParamsPanel()
    assert p.btn_im_sigma.text() == "Im Σ(E) — scattering rate"
    # Disabled until a fit exists (the action needs fit_result + vF).
    assert p.btn_im_sigma.isEnabled() is False
    p.update_fit_quality({"e_fitted": [0.0, 0.1]}, 5.0)
    assert p.btn_im_sigma.isEnabled() is True
    # Clearing the fit (None) re-disables it.
    p.update_fit_quality(None, 5.0)
    assert p.btn_im_sigma.isEnabled() is False
