"""Selection description helper for FileBrowserPanel."""
from __future__ import annotations

from pathlib import Path

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QListWidgetItem


def describe_item(panel, item: QListWidgetItem | None) -> str:
    if item is None:
        return "Sélectionne un fichier à charger"
    group = item.data(Qt.ItemDataRole.UserRole + 2)
    if group is not None:
        return "Dossier de groupe : double-clic ou Charger pour ouvrir/réduire"
    path_txt = item.data(Qt.ItemDataRole.UserRole)
    if not path_txt:
        return "Sélectionne un fichier à charger"
    p = Path(path_txt)
    key = item.data(Qt.ItemDataRole.UserRole + 1) or panel._session.key_for_path(p)
    status = panel._file_status(key)
    entry = panel._session.files.get(key)
    loader = panel._loader_label_for_path(p, key) or "inconnu"
    kind = "FS" if panel._fs_suffix_for_path(p) else "BM"
    try:
        rel = str(p.relative_to(panel._folder)) if panel._folder else str(p)
    except Exception:
        rel = str(p)
    bits = [f"{p.name}", rel, f"{kind} {loader}", f"état: {status}"]
    hv, hv_src = panel._meta_value_for_path(p, "hv")
    temp, temp_src = panel._meta_value_for_path(p, "temperature")
    pol, pol_src = panel._meta_value_for_path(p, "polarization")
    direction, dir_src = panel._meta_value_for_path(p, "direction")
    azi, azi_src = panel._meta_value_for_path(p, "azi")
    polar, p_src = panel._meta_value_for_path(p, "polar")
    tilt, t_src = panel._meta_value_for_path(p, "tilt")
    if hv is not None:
        bits.append(f"hν={float(hv):.1f} eV ({hv_src})")
    if temp is not None:
        bits.append(f"T={float(temp):.1f} K ({temp_src})")
    if pol:
        bits.append(f"pol={pol} ({pol_src})")
    if direction:
        bits.append(f"direction={direction} ({dir_src})")
    geom = []
    if azi is not None and abs(float(azi)) > 1e-9:
        geom.append(f"azi={float(azi):.1f}°")
    if polar is not None and abs(float(polar)) > 1e-9:
        geom.append(f"polar={float(polar):.1f}°")
    if tilt is not None and abs(float(tilt)) > 1e-9:
        geom.append(f"tilt={float(tilt):.1f}°")
    if geom:
        sources = sorted({s for s in (azi_src, p_src, t_src) if s})
        src_txt = f" ({'+'.join(sources)})" if sources else ""
        bits.append("géom: " + ", ".join(geom) + src_txt)
    if entry is not None:
        if entry.fit_result:
            bits.append("fit enregistré")
        tags = panel._tags_for_path(p, key)
        if tags:
            bits.append("tags: " + ", ".join(tags))
    return "\n".join(bits)
