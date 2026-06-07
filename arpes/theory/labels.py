"""Label and path helpers for DFT band structures."""
from __future__ import annotations

from typing import Any
import re


def normalize_direction_label(value: Any) -> str:
    """Normalize common logbook direction spellings without changing meaning."""
    text = "" if value is None else str(value).strip()
    if not text:
        return ""
    text = re.sub(r"(?i)\bgamma\b", "Γ", text)
    text = re.sub(r"(?i)\bg(?=\s*(?:$|[-_/→> ]))", "Γ", text)
    text = text.replace("->", "-").replace("→", "-").replace("_", "-").replace("/", "-")
    text = re.sub(r"\s+", "", text)
    text = re.sub(r"-+", "-", text)
    if re.fullmatch(r"(?i)g[mkxy]", text):
        text = "Γ" + text[1:]
    if len(text) == 2 and text.startswith("Γ"):
        text = f"Γ-{text[1]}"
    return text.upper().replace("Γ", "Γ")


def branch_display_names(branches: list[dict[str, Any]] | None) -> list[str]:
    """Display names for MP branches, in real path order."""
    out: list[str] = []
    counts: dict[str, int] = {}
    for br in branches or []:
        name = _clean_segment_name(str(br.get("name") or ""))
        if not name:
            name = "?"
        counts[name] = counts.get(name, 0) + 1
        out.append(name if counts[name] == 1 else f"{name} ({counts[name]})")
    return out


def _clean_segment_name(name: str) -> str:
    """`\\Gamma-X` / `GAMMA-X` -> `Γ-X`."""
    if "-" not in name:
        return _clean_label(name)
    a, _, b = name.partition("-")
    return f"{_clean_label(a)}-{_clean_label(b)}"


def _branch_index_for_segment(
    branches: list[dict[str, Any]], segment: str
) -> dict[str, Any] | None:
    """Find the branch whose display name matches the selected segment."""
    names = branch_display_names(branches)
    for disp, br in zip(names, branches):
        if disp == segment:
            return br
    for disp, br in zip(names, branches):
        if disp.split(" (")[0] == segment:
            return br
    return None


def segment_from_direction(
    direction: str,
    labels: list[dict[str, Any]],
    branches: list[dict[str, Any]] | None = None,
) -> str:
    """Return a matching segment name, else ``""``."""
    norm = normalize_direction_label(direction)
    if not norm or "-" not in norm:
        return ""
    a, b = [part.strip() for part in norm.split("-", 1)]
    if not a or not b:
        return ""
    if branches:
        disp_names = branch_display_names(branches)
        for disp in disp_names:
            base = disp.split(" (")[0]
            if "-" not in base:
                continue
            la, _, lb = base.partition("-")
            up = lambda s: s.upper().replace("GAMMA", "Γ")
            if (up(la), up(lb)) == (a, b) or (up(lb), up(la)) == (a, b):
                return disp
        return ""
    names = [str(item.get("label") or "") for item in labels]
    pairs = set()
    for left, right in zip(names, names[1:]):
        if left and right:
            pairs.add((left.upper().replace("GAMMA", "Γ"), right.upper().replace("GAMMA", "Γ")))
    if (a, b) in pairs:
        return f"{a}-{b}"
    if (b, a) in pairs:
        return f"{b}-{a}"
    return ""


def available_segments(
    labels: list[dict[str, Any]],
    branches: list[dict[str, Any]] | None = None,
) -> list[str]:
    """Segments DFT proposables dans le menu."""
    if branches:
        return branch_display_names(branches)
    names = [str(item.get("label") or "") for item in labels if item.get("label")]
    out: list[str] = []
    seen: set[str] = set()

    def push(seg: str) -> None:
        if seg and seg not in seen:
            seen.add(seg)
            out.append(seg)

    if "Γ" in names:
        for label in names:
            if label and label != "Γ":
                push(f"Γ-{label}")
    last = ""
    for label in names:
        if last and label:
            push(f"{last}-{label}")
        last = label
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            if a and b and a != b:
                push(f"{a}-{b}")
    return out


_GREEK_MAP = {
    "GAMMA": "Γ", "SIGMA": "Σ", "DELTA": "Δ", "LAMBDA": "Λ",
    "PI": "Π", "OMEGA": "Ω", "PHI": "Φ", "THETA": "Θ", "EPSILON": "Ε",
}

_SUBSCRIPT_MAP = str.maketrans("0123456789+-", "₀₁₂₃₄₅₆₇₈₉₊₋")


def _clean_label(label: Any) -> str:
    """Convertit labels pymatgen (`\\Sigma_1`, `\\Gamma`, etc.) en Unicode."""
    text = str(label).strip()
    if not text:
        return ""
    if text.upper() in {"G", "GAMMA", "\\GAMMA"}:
        return "Γ"
    raw = text.lstrip("\\")
    base, _, sub = raw.partition("_")
    sub = sub.strip("{}")
    base_upper = base.upper()
    base_clean = _GREEK_MAP.get(base_upper, base)
    if base_clean == base and len(base_upper) == 1:
        base_clean = base_upper
    if sub:
        try:
            sub_clean = sub.translate(_SUBSCRIPT_MAP)
        except Exception:
            sub_clean = sub
        return f"{base_clean}{sub_clean}"
    return base_clean
