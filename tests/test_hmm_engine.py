"""HMM engine: fit, inference, regime count search, label stability."""

from __future__ import annotations

from pathlib import Path

from core.config import load_config
from data.feature_engineering import build_features, feature_spec_from_cfg
from core.hmm_engine import HMMEngine
from core.regime_labeling import label_states


def _fit_engine(ohlcv):
    cfg = load_config()
    features = build_features(ohlcv, cfg.features).dropna()
    spec = feature_spec_from_cfg(cfg.features)
    engine = HMMEngine(cfg.hmm, cfg.regime_labels.names, spec)
    engine.fit(features)
    return engine, features, cfg


def test_hmm_fits_and_picks_components_in_range(synthetic_ohlcv):
    engine, _, cfg = _fit_engine(synthetic_ohlcv)
    assert engine.n_components is not None
    assert cfg.hmm.n_components_min <= engine.n_components <= cfg.hmm.n_components_max
    assert engine.model is not None


def test_infer_returns_known_label(synthetic_ohlcv):
    engine, features, _ = _fit_engine(synthetic_ohlcv)
    out = engine.infer_forward(features)
    valid_labels = {"crash", "bear", "neutral", "bull", "euphoria"}
    assert out.label in valid_labels
    assert 0.0 <= out.confidence <= 1.0


def test_labels_ordered_by_mean_return():
    """Lowest mean-return state must get the worst label; highest the best."""
    state_returns = {0: 0.001, 1: -0.005, 2: 0.0}  # states with various returns
    labels = ["crash", "bear", "neutral", "bull", "euphoria"]
    out = label_states(state_returns, labels)
    # state 1 (most negative) → crash, state 2 (middle) → neutral, state 0 (highest) → euphoria
    assert out[1] == "crash"
    assert out[0] == "euphoria"
    assert out[2] == "neutral"


def test_save_and_load_roundtrip(tmp_path: Path, synthetic_ohlcv):
    engine, features, cfg = _fit_engine(synthetic_ohlcv)
    path = tmp_path / "model.pkl"
    engine.save(path)
    spec = feature_spec_from_cfg(cfg.features)
    reloaded = HMMEngine.load(path, cfg.hmm, spec)
    assert reloaded.n_components == engine.n_components
    assert reloaded.state_to_label == engine.state_to_label

    # Same inference on same features
    a = engine.infer_forward(features)
    b = reloaded.infer_forward(features)
    assert a.state == b.state and a.label == b.label


def test_load_rejects_feature_hash_mismatch(tmp_path: Path, synthetic_ohlcv):
    engine, _, cfg = _fit_engine(synthetic_ohlcv)
    path = tmp_path / "model.pkl"
    engine.save(path)

    spec = feature_spec_from_cfg(cfg.features)
    # Tweak the spec — different window → different hash → must raise
    from data.feature_engineering import FeatureSpec
    altered = FeatureSpec(
        return_window=spec.return_window,
        vol_window=spec.vol_window + 1,
        volume_zscore_window=spec.volume_zscore_window,
        atr_window=spec.atr_window,
        range_expansion_window=spec.range_expansion_window,
    )
    import pytest
    with pytest.raises(RuntimeError, match="Feature spec hash mismatch"):
        HMMEngine.load(path, cfg.hmm, altered)
