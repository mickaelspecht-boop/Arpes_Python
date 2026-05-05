import importlib
import unittest


class PlotsSplitTests(unittest.TestCase):
    def test_split_modules_import_and_export_public_api(self):
        try:
            package = importlib.import_module("arpes.ui.widgets.plots")
            modules = [
                importlib.import_module("arpes.ui.widgets.plots.common"),
                importlib.import_module("arpes.ui.widgets.plots.band_map"),
                importlib.import_module("arpes.ui.widgets.plots.fermi_surface"),
                importlib.import_module("arpes.ui.widgets.plots.mdc_edc"),
                importlib.import_module("arpes.ui.widgets.plots.fit_overlay"),
            ]
            shim = importlib.import_module("arpes_plots")
        except ModuleNotFoundError as exc:
            self.skipTest(f"optional plotting dependency unavailable: {exc}")

        for module in modules:
            public = [name for name in dir(module) if not name.startswith("_")]
            self.assertTrue(public, module.__name__)

        self.assertIs(shim.fit_mdc_peak_pairs, package.fit_mdc_peak_pairs)
        self.assertIs(shim._resolution_correct_gamma, package._resolution_correct_gamma)
