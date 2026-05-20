"""Regime-aware allocation (legacy regime→exposure table).

Maps regime + confidence to a target gross exposure and leverage cap. Mid-vol
regimes (bear / neutral by default) require trend confirmation; without it the
exposure is scaled toward zero. Below `cfg.confidence_floor`, exposure fades
linearly toward zero.

The final-prompt vol-tier strategies size positions directly; this table still
backs `allocate()` and the legacy orchestrator path.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class AllocationDecision:
    target_exposure: float
    leverage_cap: float
    reason: str


def allocate(regime: str, confidence: float, trend_ok: bool, cfg) -> AllocationDecision:  # noqa: ANN001
    """cfg: AllocationCfg."""
    entry = cfg.by_regime.get(regime)
    if entry is None:
        return AllocationDecision(0.0, 1.0, f"unknown regime '{regime}' → flat")

    exposure = entry.target_exposure
    leverage_cap = entry.leverage_cap
    reasons: list[str] = [f"regime={regime}"]

    if entry.requires_trend_confirmation and not trend_ok:
        exposure = 0.0
        reasons.append("trend gate failed")

    if confidence < cfg.confidence_floor:
        scale = max(0.0, confidence) / cfg.confidence_floor if cfg.confidence_floor > 0 else 1.0
        exposure *= scale
        reasons.append(f"low confidence {confidence:.2f}, scale={scale:.2f}")

    return AllocationDecision(
        target_exposure=exposure,
        leverage_cap=leverage_cap,
        reason="; ".join(reasons),
    )
