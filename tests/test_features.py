"""Final-prompt feature set: all features present, causal, z-scored."""

from __future__ import annotations

from core.config import load_config
from data.feature_engineering import FEATURE_COLUMNS, build_features


def test_full_feature_set_present(synthetic_ohlcv):
    cfg = load_config()
    feats = build_features(synthetic_ohlcv, cfg.features)
    assert tuple(feats.columns) == FEATURE_COLUMNS
    assert len(FEATURE_COLUMNS) >= 11  # final prompt asks for 11+ features


def test_no_nan_past_warmup(synthetic_ohlcv):
    cfg = load_config()
    feats = build_features(synthetic_ohlcv, cfg.features).dropna()
    assert not feats.empty
    assert not feats.isna().any().any()


def test_features_are_causal(synthetic_ohlcv):
    """Computing features on a prefix must match the full-series rows for that prefix."""
    cfg = load_config()
    full = build_features(synthetic_ohlcv, cfg.features)
    cut = int(len(synthetic_ohlcv) * 0.8)
    prefix = build_features(synthetic_ohlcv.iloc[:cut], cfg.features)
    overlap = full.iloc[:cut].dropna()
    pre = prefix.reindex(overlap.index)
    # rolling z-scores are causal → identical on the overlapping window
    assert (overlap - pre).abs().max().max() < 1e-9


def test_features_are_zscored(synthetic_ohlcv):
    """Z-scored columns sit in a sane standardised range."""
    cfg = load_config()
    feats = build_features(synthetic_ohlcv, cfg.features).dropna()
    # the bulk of every column should fall within a few standard deviations
    assert feats.abs().mean().max() < 5.0
