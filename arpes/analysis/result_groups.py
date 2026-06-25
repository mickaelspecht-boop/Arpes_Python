"""Results-panel grouping logic (pure, no PyQt).

Two independent ways to organise the fitted files of a session:

* **Auto-grouping** — derived on the fly from file metadata (compound /
  polarisation / direction). Never stored; recomputed at draw time.
* **Manual groups** — user-named, colour-tagged groups stored on
  ``Session.groups`` (see ``core/session.py``). Membership is *exclusive*
  (one group per file) so "colour by group" is unambiguous and a file never
  appears twice.

All functions here are headless and unit-tested; the Qt tree/menus live in
``ui/widgets/results_groups.py`` and only call into this layer.
"""
from __future__ import annotations

from arpes.analysis.aggregation import compound_label

# "Group by" selector values (also the combo item order).
GROUP_BY_NONE = "None"
GROUP_BY_COMPOUND = "Compound"
GROUP_BY_POLARISATION = "Polarisation"
GROUP_BY_DIRECTION = "Direction"
GROUP_BY_MANUAL = "Manual groups"
GROUP_BY_OPTIONS = (
    GROUP_BY_NONE,
    GROUP_BY_COMPOUND,
    GROUP_BY_POLARISATION,
    GROUP_BY_DIRECTION,
    GROUP_BY_MANUAL,
)
UNGROUPED = "Ungrouped"


# -- manual group storage helpers (operate on session.groups) ----------------

def _groups(session) -> list[dict]:
    g = getattr(session, "groups", None)
    if not isinstance(g, list):
        g = []
        session.groups = g
    return g


def group_names(session) -> list[str]:
    return [str(g.get("name", "")) for g in _groups(session)]


def find_group(session, name: str) -> dict | None:
    key = str(name).casefold()
    for g in _groups(session):
        if str(g.get("name", "")).casefold() == key:
            return g
    return None


def add_group(session, name: str | None = None, color_idx: int | None = None) -> dict:
    """Create a group (auto-named ``Group N`` and auto-coloured if unspecified).

    Returns the existing group if ``name`` already exists (case-insensitive)."""
    groups = _groups(session)
    if name:
        existing = find_group(session, name)
        if existing is not None:
            return existing
    else:
        n = len(groups) + 1
        while find_group(session, f"Group {n}") is not None:
            n += 1
        name = f"Group {n}"
    if color_idx is None:
        color_idx = len(groups)
    g = {"name": str(name), "color_idx": int(color_idx), "members": []}
    groups.append(g)
    return g


def remove_group(session, name: str) -> None:
    key = str(name).casefold()
    session.groups = [g for g in _groups(session)
                      if str(g.get("name", "")).casefold() != key]


def rename_group(session, old: str, new: str) -> bool:
    """Rename a group. False on empty name or a clash with another group."""
    new = str(new).strip()
    if not new:
        return False
    g = find_group(session, old)
    if g is None:
        return False
    clash = find_group(session, new)
    if clash is not None and clash is not g:
        return False
    g["name"] = new
    return True


def set_group_color(session, name: str, color_idx: int) -> None:
    g = find_group(session, name)
    if g is not None:
        g["color_idx"] = int(color_idx)


def assign_to_group(session, name: str, filenames) -> None:
    """Add ``filenames`` to group ``name``, removing them from every other
    group first (exclusive membership)."""
    g = find_group(session, name)
    if g is None:
        return
    fset = [str(f) for f in filenames]
    fkey = set(fset)
    for other in _groups(session):
        if other is g:
            continue
        other["members"] = [m for m in other.get("members", []) if m not in fkey]
    members = g.setdefault("members", [])
    for f in fset:
        if f not in members:
            members.append(f)


def unassign(session, filenames) -> None:
    fkey = {str(f) for f in filenames}
    for g in _groups(session):
        g["members"] = [m for m in g.get("members", []) if m not in fkey]


def group_of_file(session, filename: str) -> dict | None:
    name = str(filename)
    for g in _groups(session):
        if name in g.get("members", []):
            return g
    return None


def prune_groups(session, valid_files) -> None:
    """Drop members that no longer exist in the session (e.g. removed files)."""
    valid = {str(f) for f in valid_files}
    for g in _groups(session):
        g["members"] = [m for m in g.get("members", []) if m in valid]


# -- auto-grouping (metadata-derived, not stored) ----------------------------

def auto_group_label(session, filename: str, by: str) -> str:
    """Group label for one file under an auto-grouping dimension."""
    entry = getattr(session, "files", {}).get(filename)
    meta = getattr(entry, "meta", None)
    if by == GROUP_BY_COMPOUND:
        return compound_label(filename)
    if by == GROUP_BY_POLARISATION:
        return (str(getattr(meta, "polarization", "") or "").strip()) or "?"
    if by == GROUP_BY_DIRECTION:
        return (str(getattr(meta, "direction", "") or "").strip()) or "?"
    return ""


def grouped_files(session, fitted_names, by: str) -> list[tuple[str, list[str]]]:
    """Ordered ``[(group_label, [filenames])]`` for the chosen dimension.

    * ``None``           -> single unnamed bucket ``("", names)`` (flat list).
    * compound/pol/dir   -> buckets by metadata label, first-seen order.
    * manual             -> one bucket per ``session.groups`` (in order) holding
      its still-present members, then an ``UNGROUPED`` bucket for the rest.
    Only files present in ``fitted_names`` are placed.
    """
    names = [str(n) for n in fitted_names]
    if by == GROUP_BY_NONE:
        return [("", names)]
    if by == GROUP_BY_MANUAL:
        present = set(names)
        out: list[tuple[str, list[str]]] = []
        claimed: set[str] = set()
        for g in _groups(session):
            members = [m for m in g.get("members", []) if m in present]
            claimed.update(members)
            out.append((str(g.get("name", "")), members))
        rest = [n for n in names if n not in claimed]
        if rest or not out:
            out.append((UNGROUPED, rest))
        return out
    # metadata auto-grouping, preserving first-seen label order
    order: list[str] = []
    buckets: dict[str, list[str]] = {}
    for n in names:
        label = auto_group_label(session, n, by)
        if label not in buckets:
            buckets[label] = []
            order.append(label)
        buckets[label].append(n)
    return [(label, buckets[label]) for label in order]


def file_color_index(session, fitted_names, by: str) -> dict[str, int]:
    """Map each fitted file to a palette colour index for "colour by group".

    Auto modes index groups by appearance order; manual mode uses each group's
    stored ``color_idx`` and tags ungrouped files with ``-1`` (caller renders
    them grey). ``None`` mode returns an empty map (per-file colours apply).
    """
    if by == GROUP_BY_NONE:
        return {}
    out: dict[str, int] = {}
    if by == GROUP_BY_MANUAL:
        for g in _groups(session):
            for m in g.get("members", []):
                out[str(m)] = int(g.get("color_idx", 0))
        for n in fitted_names:
            out.setdefault(str(n), -1)
        return out
    for idx, (_label, names) in enumerate(grouped_files(session, fitted_names, by)):
        for n in names:
            out[n] = idx
    return out
