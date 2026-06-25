"""Band visibility, naming and plotting style for the Results panel."""
from __future__ import annotations

import colorsys

import numpy as np
from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtWidgets import QAbstractItemView, QTreeWidget, QTreeWidgetItem

_ROLE_FILE = Qt.ItemDataRole.UserRole
_ROLE_PAIR = Qt.ItemDataRole.UserRole + 1
_ROLE_LABEL = Qt.ItemDataRole.UserRole + 2
_PAIR_MARKERS = ("o", "s", "D", "P", "X", "v", "<", ">")
_PAIR_LINESTYLES = ("-", "--", "-.", ":")


def band_key(filename: str, pair_index: int) -> tuple[str, int]:
    return str(filename), int(pair_index)


def band_name(entry, pair_index: int) -> str:
    pairs = list(getattr(getattr(entry, "fit_params", None), "pairs", None) or [])
    if 0 <= pair_index < len(pairs):
        label = str(pairs[pair_index].get("label") or "").strip()
        if label:
            return label
    return f"P{pair_index + 1}"


def band_label(filename: str, entry, pair_index: int) -> str:
    return f"{filename} · {band_name(entry, pair_index)}"


def band_visible(panel, filename: str, pair_index: int) -> bool:
    return band_key(filename, pair_index) not in getattr(panel, "_hidden_bands", set())


def band_style(base_color, pair_index: int) -> dict:
    """Distinct pair style while retaining the file's base-colour family."""
    rgb = tuple(float(v) for v in base_color[:3])
    h, s, v = colorsys.rgb_to_hsv(*rgb)
    if pair_index:
        h = (h + 0.16 * pair_index) % 1.0
        s = min(1.0, max(0.58, s * 1.08))
        v = min(1.0, max(0.72, v * (1.0 - 0.04 * (pair_index % 3))))
    return {
        "color": colorsys.hsv_to_rgb(h, s, v),
        "marker_minus": _PAIR_MARKERS[(2 * pair_index) % len(_PAIR_MARKERS)],
        "marker_plus": _PAIR_MARKERS[(2 * pair_index + 1) % len(_PAIR_MARKERS)],
        "linestyle": _PAIR_LINESTYLES[pair_index % len(_PAIR_LINESTYLES)],
    }


def high_uncertainty_mask(values, sigma) -> np.ndarray:
    """Flag imprecise points for display without removing them from results."""
    values = np.asarray(values, dtype=float)
    sigma = np.asarray(sigma, dtype=float)
    n = min(values.size, sigma.size)
    out = np.zeros(values.size, dtype=bool)
    if n == 0:
        return out
    finite_sigma = sigma[:n][np.isfinite(sigma[:n]) & (sigma[:n] > 0)]
    if finite_sigma.size == 0:
        return out
    threshold = max(0.03, 3.0 * float(np.nanmedian(finite_sigma)))
    out[:n] = np.isfinite(values[:n]) & (
        ~np.isfinite(sigma[:n]) | (sigma[:n] > threshold)
    )
    return out


def plot_branch_segments(
    ax, k_values, e_values, *, color, alpha=0.6, linestyle="-"
) -> None:
    """Connect finite points without drawing across gaps or branch jumps."""
    k = np.asarray(k_values, dtype=float)
    e = np.asarray(e_values, dtype=float)
    n = min(k.size, e.size)
    if n < 2:
        return
    k, e = k[:n], e[:n]
    finite = np.isfinite(k) & np.isfinite(e)
    start = prev = None

    def draw_segment(first, last):
        if first is not None and last is not None and last - first + 1 >= 2:
            ax.plot(k[first:last + 1], e[first:last + 1], linestyle,
                    lw=0.9, color=color, alpha=alpha, zorder=2)

    for idx, ok in enumerate(finite):
        if not ok:
            draw_segment(start, prev)
            start = prev = None
            continue
        if start is None:
            start = idx
        elif prev is not None and (
            abs(k[idx] - k[prev]) > 0.10 or abs(e[idx] - e[prev]) > 0.08
        ):
            draw_segment(start, prev)
            start = idx
        prev = idx
    draw_segment(start, prev)


def build_band_registry(panel, layout) -> None:
    panel._hidden_bands = set()
    panel._bands_syncing = False
    panel._bands_signature = None
    tree = QTreeWidget()
    tree.setHeaderLabels(["Fichier / bande", "État"])
    tree.setMaximumHeight(175)
    tree.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
    tree.setAlternatingRowColors(True)
    tree.setStyleSheet(
        "QTreeWidget{background:#222;color:#ddd;font-size:10px;"
        "alternate-background-color:#292929;}"
        "QHeaderView::section{background:#333;color:#eee;}"
    )
    tree.setToolTip(
        "Coche/décoche une bande pour l'affichage et l'export.\n"
        "Double-clique son nom pour le renommer; le nom est sauvegardé dans la session."
    )
    tree.itemChanged.connect(lambda item, col: _on_band_item_changed(panel, item, col))
    panel._tree_bands = tree
    layout.addWidget(tree)


def sync_band_registry(panel) -> None:
    fitted = [
        (name, entry, int((entry.fit_result or {}).get("n_pairs")
                          or entry.fit_params.n_pairs or 1))
        for name, entry in panel._session.files.items()
        if entry.fit_result is not None
    ]
    signature = tuple(
        (
            name, n_pairs,
            tuple(
                (
                    band_name(entry, i),
                    (entry.fit_params.pairs[i].get("results_visible", True)
                     if i < len(entry.fit_params.pairs) else True),
                )
                for i in range(n_pairs)
            ),
        )
        for name, entry, n_pairs in fitted
    )
    if signature == getattr(panel, "_bands_signature", None):
        return
    panel._bands_syncing = True
    try:
        tree = panel._tree_bands
        tree.clear()
        valid_keys = set()
        for name, entry, n_pairs in fitted:
            parent = QTreeWidgetItem([name, f"{n_pairs} bande(s)"])
            parent.setFlags(parent.flags() & ~Qt.ItemFlag.ItemIsEditable)
            tree.addTopLevelItem(parent)
            for pair_index in range(n_pairs):
                key = band_key(name, pair_index)
                valid_keys.add(key)
                pairs = list(getattr(entry.fit_params, "pairs", None) or [])
                if (
                    pair_index < len(pairs)
                    and pairs[pair_index].get("results_visible") is False
                ):
                    panel._hidden_bands.add(key)
                child = QTreeWidgetItem([band_name(entry, pair_index), f"P{pair_index + 1}"])
                child.setData(0, _ROLE_FILE, name)
                child.setData(0, _ROLE_PAIR, pair_index)
                child.setData(0, _ROLE_LABEL, band_name(entry, pair_index))
                child.setFlags(
                    child.flags()
                    | Qt.ItemFlag.ItemIsUserCheckable
                    | Qt.ItemFlag.ItemIsEditable
                )
                child.setCheckState(
                    0, Qt.CheckState.Unchecked if key in panel._hidden_bands
                    else Qt.CheckState.Checked,
                )
                parent.addChild(child)
            parent.setExpanded(n_pairs > 1)
        panel._hidden_bands.intersection_update(valid_keys)
        panel._bands_signature = signature
        tree.resizeColumnToContents(0)
    finally:
        panel._bands_syncing = False


def _on_band_item_changed(panel, item, column: int) -> None:
    if getattr(panel, "_bands_syncing", False):
        return
    filename = item.data(0, _ROLE_FILE)
    pair_index = item.data(0, _ROLE_PAIR)
    if filename is None or pair_index is None:
        return
    key = band_key(str(filename), int(pair_index))
    if item.checkState(0) == Qt.CheckState.Checked:
        panel._hidden_bands.discard(key)
        visible = True
    else:
        panel._hidden_bands.add(key)
        visible = False
    entry = panel._session.files.get(str(filename))
    changed_visibility = False
    if entry is not None:
        pairs = entry.fit_params.pairs
        while len(pairs) <= int(pair_index):
            pairs.append({"kF_init": 0.30, "gamma_init": 0.08, "gamma_max": 0.30})
        if pairs[int(pair_index)].get("results_visible", True) != visible:
            pairs[int(pair_index)]["results_visible"] = visible
            changed_visibility = True
    previous_label = str(item.data(0, _ROLE_LABEL) or "")
    current_label = item.text(0).strip() or f"P{int(pair_index) + 1}"
    changed_label = False
    if column == 0 and current_label != previous_label:
        if entry is not None:
            pairs[int(pair_index)]["label"] = current_label
            panel._bands_signature = None
            changed_label = True
            if panel._host is not None and hasattr(panel._host, "_status"):
                panel._host._status(
                    f"✓ Bande renommée: {filename} P{int(pair_index) + 1} → "
                    f"{current_label} (session)"
                )
    if changed_visibility or changed_label:
        panel._session.save()
    QTimer.singleShot(0, panel.refresh)
