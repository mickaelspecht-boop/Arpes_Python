"""The active zone rectangle must track the live panel ROI, not its snapshot.

This is the "zones don't update with parameter changes" guard: while a zone is
selected, editing the k/E window moves its rectangle immediately because the
overlay draws the active zone from ctrl._fit_roi_bounds() (live spinboxes).
Other zones keep their stored snapshot.
"""
from __future__ import annotations

from types import SimpleNamespace

from arpes.ui.controllers.fit_overlay_drawer import draw_zone_overlays


class _FakeAx:
    def __init__(self):
        self.patches = []
        self.texts = []

    def add_patch(self, patch):
        self.patches.append(patch)

    def text(self, *a, **k):
        self.texts.append((a, k))

    def scatter(self, *a, **k):
        pass


def _zone(zid, color_idx, win):
    return {
        "id": zid,
        "label": zid,
        "color_idx": color_idx,
        "active": True,
        "fit_params": {"k_min": win[0], "k_max": win[1],
                       "ev_start": win[2], "ev_end": win[3]},
        "fit_result": None,
    }


def _ctrl(zones, active_id, live_bounds):
    entry = SimpleNamespace(fit_zones=zones, active_zone_id=active_id, annotations={})
    session = SimpleNamespace(
        key_for_path=lambda p: p,
        get_or_create=lambda k: entry,
    )
    parent = SimpleNamespace(_current_path="f", _session=session)
    return SimpleNamespace(_parent=parent, _fit_roi_bounds=lambda: live_bounds)


def test_active_rect_uses_live_bounds_not_snapshot():
    # Active zone snapshot says [-0.8,0.8]; live spinboxes say [-0.2,0.4].
    active = _zone("a", 0, (-0.8, 0.8, -0.9, 0.0))
    other = _zone("b", 1, (0.1, 0.5, -0.5, 0.0))
    ctrl = _ctrl([active, other], "a", live_bounds=(-0.2, 0.4, -0.7, -0.05))
    ax = _FakeAx()
    draw_zone_overlays(ctrl, ax)

    rects = {round(p.get_x(), 4): (round(p.get_width(), 4), round(p.get_height(), 4))
             for p in ax.patches}
    # Active rect from LIVE bounds: x=-0.2, width=0.6, height=0.65.
    assert -0.2 in rects
    assert rects[-0.2] == (0.6, 0.65)
    # Other rect from its stored snapshot: x=0.1, width=0.4.
    assert 0.1 in rects
    assert rects[0.1][0] == 0.4


def test_active_rect_falls_back_to_snapshot_when_no_live_bounds():
    active = _zone("a", 0, (-0.8, 0.8, -0.9, 0.0))
    ctrl = _ctrl([active], "a", live_bounds=None)
    ax = _FakeAx()
    draw_zone_overlays(ctrl, ax)
    xs = [round(p.get_x(), 4) for p in ax.patches]
    assert xs == [-0.8]
