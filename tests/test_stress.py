"""Final-prompt stress-test suite: Monte-Carlo crash, gap risk, misclassification."""

from __future__ import annotations

from core.config import load_config
from backtest import stress_tests


def test_monte_carlo_crash_reports_distribution(synthetic_ohlcv):
    out = stress_tests.monte_carlo_crash(synthetic_ohlcv, n_runs=30, n_shocks=8)
    assert set(out) >= {"n_runs", "mean_max_loss", "worst_max_loss", "kill_fired_rate"}
    assert out["n_runs"] == 30
    assert out["worst_max_loss"] <= out["mean_max_loss"]      # worst is the deepest
    assert 0.0 <= out["kill_fired_rate"] <= 1.0


def test_gap_risk_test_reports_expected_vs_realised(synthetic_ohlcv):
    out = stress_tests.gap_risk_test(synthetic_ohlcv)
    assert set(out) >= {"expected_loss_pct", "realised_loss_pct", "exceeds_expected"}
    assert out["realised_loss_pct"] >= 0.0


def test_regime_misclassification_risk_contained(synthetic_ohlcv):
    """Shuffling the regime→strategy map must not blow past the drawdown cap."""
    cfg = load_config()

    def fake_backtest(ohlcv, cfg, shuffle_seed):  # noqa: ANN001
        # shuffled mapping degrades returns but the risk layer caps the damage.
        dd = -0.04 if shuffle_seed is None else -0.09
        return {"summary": {"max_drawdown": dd}}

    out = stress_tests.regime_misclassification(fake_backtest, cfg, synthetic_ohlcv, seed=1)
    assert "risk_contained" in out
    assert out["risk_contained"] is True
