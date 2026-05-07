from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from arpes.theory.local_loaders import (
    load_local_band_data,
    load_qe_bands,
    load_yaml_bands,
)


class TestLocalDftLoaders(unittest.TestCase):
    def test_yaml_schema_loads_band_axis_labels_and_efermi(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "bands.yaml"
            path.write_text(
                """
material_id: local:test
formula: AB2
k_distance: [0.0, 0.5, 1.0]
labels:
  Gamma: 0.0
  X: 1.0
bands:
  - [0.0, 0.5, 1.0]
  - [2.0, 2.5, 3.0]
efermi: 0.5
""",
                encoding="utf-8",
            )

            data = load_yaml_bands(path)

        self.assertEqual(data.source, "local_yaml")
        self.assertEqual(data.material_id, "local:test")
        self.assertEqual([item["label"] for item in data.labels], ["Γ", "X"])
        np.testing.assert_allclose(data.k_distance, [0.0, 0.5, 1.0])
        np.testing.assert_allclose(data.bands[0], [-0.5, 0.0, 0.5])

    def test_json_dispatch_computes_k_distance_from_kpoints(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "bands.json"
            path.write_text(
                json.dumps({
                    "kpoints": [[0.0, 0.0, 0.0], [0.5, 0.0, 0.0], [0.5, 0.5, 0.0]],
                    "labels": [{"label": "G", "index": 0}, {"label": "M", "index": 2}],
                    "bands": [[-1.0, 0.0, 1.0]],
                    "efermi": 0.0,
                }),
                encoding="utf-8",
            )

            data = load_local_band_data(path)

        self.assertEqual(data.source, "local_yaml")
        np.testing.assert_allclose(data.k_distance, [0.0, 0.5, 1.0])
        self.assertEqual(data.labels, [{"label": "Γ", "k": 0.0}, {"label": "M", "k": 1.0}])

    def test_qe_table_loads_k_rows_as_band_matrix(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "qe.dat"
            path.write_text(
                "# k e1 e2\n"
                "0.0 -0.1 1.0\n"
                "0.5 0.0 1.1\n"
                "1.0 0.1 1.2\n",
                encoding="utf-8",
            )

            data = load_qe_bands(path)

        self.assertEqual(data.source, "local_qe")
        np.testing.assert_allclose(data.k_distance, [0.0, 0.5, 1.0])
        np.testing.assert_allclose(data.bands, [[-0.1, 0.0, 0.1], [1.0, 1.1, 1.2]])


if __name__ == "__main__":
    unittest.main()
