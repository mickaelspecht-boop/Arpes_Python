"""Entry-point shim — ArpesExplorer a été déplacé dans `arpes.app`."""
from __future__ import annotations

from arpes.app import *  # noqa: F401,F403
from arpes.app import (  # noqa: F401  symboles requis par tests
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
from arpes.app import main  # noqa: F401


if __name__ == "__main__":
    main()
