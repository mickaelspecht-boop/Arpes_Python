"""Beamline angle-sign conventions, with data-driven fallback."""
from __future__ import annotations

from dataclasses import dataclass, asdict, field
from typing import Callable

# Sentinel: no frozen convention for this beamline → data-driven mode.
UNCALIBRATED = "UNCALIBRATED"

# Sign sources, in increasing reliability.
_SIGN_SOURCES = ("data_driven", "au_calibrated", "manual")

# Confidence thresholds (cf. physicist advice). In π/a except gap (unitless).
CONFIDENCE_AMBIGUOUS = 0.05   # relative score gap < 5% → undetermined sign
CONFIDENCE_REFUSE = 0.02      # gap < 2% AND large |gamma| → hard refusal
TIE_REL = 0.01                # |S1−S2|/S1 < 1% → tie (order bias)
GAMMA_NATIVE = 0.05           # |gamma_best| < threshold → native Γ / low SNR
GAMMA_SUSPECT = 0.5           # |gamma_best| > threshold → physically suspicious offset
MAD_RATIO_MAX = 3.0           # mad/|gamma| > threshold → flat MDC, unstable fit


@dataclass
class BeamlineAngleConvention:
    """Frozen sign convention for a beamline (or measurement point)."""
    beamline: str
    theta_sign: int = 1               # +1 or −1
    azi_sign: int = 1                 # +1 or −1
    source: str = "data_driven"       # data_driven | au_calibrated | manual
    calibrated_date: str = ""         # ISO8601, "" if uncalibrated
    notes: str = ""

    def is_frozen(self) -> bool:
        """True if the convention is frozen (Au or manual), not inferred."""
        return self.source in ("au_calibrated", "manual")

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, raw: dict) -> "BeamlineAngleConvention":
        raw = raw or {}
        src = str(raw.get("source", "data_driven"))
        if src not in _SIGN_SOURCES:
            src = "data_driven"
        def _sign(v) -> int:
            try:
                return -1 if float(v) < 0 else 1
            except (TypeError, ValueError):
                return 1
        return cls(
            beamline=str(raw.get("beamline", "")),
            theta_sign=_sign(raw.get("theta_sign", 1)),
            azi_sign=_sign(raw.get("azi_sign", 1)),
            source=src,
            calibrated_date=str(raw.get("calibrated_date", "") or ""),
            notes=str(raw.get("notes", "") or ""),
        )


# Runtime registry: geometry key -> serializable convention dict.
ConventionRegistry = dict


def convention_key(
    beamline: str,
    hv: float | None,
    azi: float | None,
    polar: float | None,
) -> str:
    """Granular registry key (cf. redteam CASE4: avoid cross-contamination).

    Rounds hv in 5 eV steps, azi in 5° steps, polar in 2° steps. Two
    geometrically close measurements share the key; distinct geometries do not
    contaminate each other.
    """
    def _bucket(v, step) -> str:
        try:
            f = float(v)
        except (TypeError, ValueError):
            return "na"
        return str(int(round(f / step) * step))
    bl = (beamline or "").strip() or "unknown"
    return f"{bl}|hv={_bucket(hv, 5)}|azi={_bucket(azi, 5)}|polar={_bucket(polar, 2)}"


def get_convention(
    registry: ConventionRegistry,
    key: str,
) -> BeamlineAngleConvention | None:
    """Frozen convention for this key, or ``None`` (UNCALIBRATED)."""
    raw = (registry or {}).get(key)
    if not raw or raw == UNCALIBRATED:
        return None
    conv = BeamlineAngleConvention.from_dict(raw)
    return conv if conv.is_frozen() else None


def freeze_convention(
    registry: ConventionRegistry,
    key: str,
    conv: BeamlineAngleConvention,
) -> None:
    """Freeze a convention in the registry (source au_calibrated/manual only)."""
    if not conv.is_frozen():
        raise ValueError(
            "freeze_convention requires source 'au_calibrated' or 'manual', "
            f"got '{conv.source}'."
        )
    registry[key] = conv.to_dict()


def _candidate_sign(label: str) -> int:
    """Implicit sign of a candidate from its label (−1 if negative)."""
    lab = (label or "").lower()
    return -1 if ("neg" in lab or "minus" in lab or lab.startswith("-")) else 1


def filter_candidates(
    candidates: list[dict],
    registry: ConventionRegistry,
    *,
    beamline: str,
    hv: float | None,
    azi: float | None,
    polar: float | None,
) -> list[dict]:
    """Restrict candidates to the frozen sign if a convention exists.

    UNCALIBRATED (no frozen convention) → returns the intact list
    (data-driven mode). Otherwise only keeps candidates with the frozen theta
    sign.
    """
    if not candidates:
        return []
    conv = get_convention(registry, convention_key(beamline, hv, azi, polar))
    if conv is None:
        return list(candidates)
    kept = [c for c in candidates if _candidate_sign(c.get("candidate", "")) == conv.theta_sign]
    return kept or list(candidates)


def select_best_candidate(
    candidates: list[dict],
    score_fn: Callable[[dict], float],
) -> dict:
    """Choose the minimum-score candidate and measure confidence.

    ``score_fn`` is injected (loads+scores a candidate) to remain headless
    testable. Returns a dict:
      best, best_score, second_score, opposite_score, confidence, tie,
      scores (by label).
    """
    if not candidates:
        return {"best": None, "best_score": float("inf"), "second_score": float("inf"),
                "confidence": 0.0, "tie": False, "scores": {}}
    scored: list[tuple[float, dict]] = []
    for cfg in candidates:
        s = float(score_fn(cfg))
        scored.append((s, cfg))
    scored.sort(key=lambda x: x[0])
    best_score, best = scored[0]
    second_score = scored[1][0] if len(scored) > 1 else float("inf")
    # S2 with opposite sign (cf. physicist: do not compare two azi variants of
    # the same sign). Fallback: global second-best.
    best_sign = _candidate_sign(best.get("candidate", ""))
    opposite_score = float("inf")
    for s, cfg in scored[1:]:
        if _candidate_sign(cfg.get("candidate", "")) != best_sign:
            opposite_score = s
            break
    ref_second = opposite_score if opposite_score < float("inf") else second_score
    eps = 1e-9
    if best_score < float("inf") and ref_second < float("inf"):
        confidence = float((ref_second - best_score) / (abs(best_score) + eps))
        tie = bool(abs(ref_second - best_score) / (abs(best_score) + eps) < TIE_REL)
    else:
        confidence = float("inf")  # only one candidate → no sign ambiguity
        tie = False
    return {
        "best": best, "best_score": best_score, "second_score": second_score,
        "opposite_score": opposite_score, "confidence": confidence, "tie": tie,
        "scores": {c.get("candidate", f"#{i}"): s for i, (s, c) in enumerate(scored)},
    }


def evaluate_confidence(
    *,
    confidence: float,
    gamma_best: float,
    mad_best: float,
    gamma_residual_after: float,
    tie: bool,
) -> dict:
    """Ambiguity/refusal verdict from selection metrics.

    Returns ``{ambiguous, refuse, reasons:list[str]}``. Physicist thresholds.
    """
    reasons: list[str] = []
    g = abs(float(gamma_best))
    if tie:
        reasons.append("tie scores (order bias)")
    if confidence < CONFIDENCE_AMBIGUOUS:
        reasons.append(f"confidence {confidence:.3f} < {CONFIDENCE_AMBIGUOUS}")
    if g < GAMMA_NATIVE:
        reasons.append(f"|Γ| {g:.3f} < {GAMMA_NATIVE} π/a (native Γ / low SNR)")
    if g > GAMMA_SUSPECT:
        reasons.append(f"|Γ| {g:.3f} > {GAMMA_SUSPECT} π/a (suspicious offset)")
    if g > 0 and float(mad_best) / g > MAD_RATIO_MAX:
        reasons.append("high mad/|Γ| (flat MDC)")
    if abs(float(gamma_residual_after)) > GAMMA_NATIVE:
        reasons.append(
            f"residual Γ {gamma_residual_after:+.3f} π/a after offset "
            "(off-Γ pocket? kF may be biased)"
        )
    refuse = bool(confidence < CONFIDENCE_REFUSE and g > 0.10)
    return {"ambiguous": bool(reasons), "refuse": refuse, "reasons": reasons}
