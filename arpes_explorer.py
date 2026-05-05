"""Entry-point shim — ArpesExplorer a été déplacé dans `arpes.app`.

Conservé à la racine pour :
- compat tests (`from arpes_explorer import ...`)
- compat lazy imports dans controllers/builders (en attente de κ.2)
- exécution directe `python3 arpes_explorer.py`

Re-exporte tous les symboles publics + privés requis par les tests via
`from arpes.app import ...`. Mutations du global `AP` doivent passer par
`arpes.app` directement (`from arpes import app as _ae; _ae.AP = ...`).
"""
from __future__ import annotations

from arpes.app import *  # noqa: F401,F403
from arpes.app import (  # noqa: F401  symboles privés requis par tests/lazy imports
    AP,
    ArpesExplorer,
    FileBrowserPanel,
    FileEntry,
    FitParamsPanel,
    MplCanvas,
    QApplication,
    ResultsPanel,
    Session,
)
from arpes.core.session import FileMeta  # noqa: F401  test import direct
from arpes.io.logbook import _format_direction_label, _infer_logbook_mapping  # noqa: F401
from arpes.app import (  # noqa: F401
    _load_ap,
    _loader_label,
    apply_ef_correction_to_dict,
    detect_format,
    load_arpes_file,
    main,
)


if __name__ == "__main__":
    main()
