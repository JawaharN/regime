"""Regime stability filter.

Rules:
- A new regime must persist >= min_persistence_bars consecutive bars before it
  becomes the *actionable* regime. Until then, the previous actionable regime is
  kept and a transition size cut applies (callers reduce position size by
  `transition_size_cut`, default 25%, while a change is unconfirmed).
- Trailing window: if the inferred regime changes more than `flicker_threshold`
  times within the last `flicker_window` updates, the state is UNSTABLE.
- When unstable, confidence is multiplied by `unstable_confidence_decay`.

The filter is *stateful*: feed every new HMM inference into `.update()`.
"""

from __future__ import annotations

import logging
from collections import deque
from dataclasses import dataclass

logger = logging.getLogger("regime_trader.regime_stability")


@dataclass
class StabilityResult:
    actionable_regime: str | None
    confidence: float
    unstable: bool
    inferred_regime: str
    persistence_count: int
    flicker_count: int
    in_transition: bool = False
    size_multiplier: float = 1.0


class RegimeStabilityFilter:
    def __init__(self, min_persistence_bars: int, flicker_window: int,
                 flicker_threshold: int, unstable_confidence_decay: float,
                 transition_size_cut: float = 0.25) -> None:
        if min_persistence_bars < 1:
            raise ValueError("min_persistence_bars must be >= 1")
        self.min_persistence_bars = min_persistence_bars
        self.flicker_window = flicker_window
        self.flicker_threshold = flicker_threshold
        self.unstable_confidence_decay = unstable_confidence_decay
        self.transition_size_cut = transition_size_cut
        self._history: deque[str] = deque(maxlen=flicker_window)
        self._current_actionable: str | None = None
        self._candidate: str | None = None
        self._candidate_streak: int = 0
        self._in_transition: bool = False

    def update(self, regime: str, confidence: float) -> StabilityResult:
        self._history.append(regime)

        # Persistence gate.
        if regime == self._candidate:
            self._candidate_streak += 1
        else:
            self._candidate = regime
            self._candidate_streak = 1

        if self._current_actionable is None:
            if self._candidate_streak >= self.min_persistence_bars:
                self._current_actionable = self._candidate
        elif regime != self._current_actionable and self._candidate_streak >= self.min_persistence_bars:
            self._current_actionable = self._candidate

        # A change is "in transition" while a *different* regime is building
        # persistence but has not yet been confirmed.
        self._in_transition = (
            self._current_actionable is not None
            and self._candidate != self._current_actionable
            and 0 < self._candidate_streak < self.min_persistence_bars
        )

        flicker_count = self._flicker_count()
        unstable = flicker_count > self.flicker_threshold
        effective_conf = confidence * (self.unstable_confidence_decay if unstable else 1.0)
        size_mult = (1.0 - self.transition_size_cut) if self._in_transition else 1.0

        if unstable:
            logger.warning(
                "regime unstable: window=%s flicker_count=%s inferred=%s actionable=%s",
                len(self._history), flicker_count, regime, self._current_actionable,
            )

        return StabilityResult(
            actionable_regime=self._current_actionable,
            confidence=effective_conf,
            unstable=unstable,
            inferred_regime=regime,
            persistence_count=self._candidate_streak,
            flicker_count=flicker_count,
            in_transition=self._in_transition,
            size_multiplier=size_mult,
        )

    def is_in_transition(self) -> bool:
        """True when an unconfirmed regime change is currently building."""
        return self._in_transition

    def _flicker_count(self) -> int:
        if len(self._history) < 2:
            return 0
        count = 0
        prev = self._history[0]
        for r in list(self._history)[1:]:
            if r != prev:
                count += 1
                prev = r
        return count
