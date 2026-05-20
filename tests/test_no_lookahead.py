"""Asserts HMM forward-inference at bar t depends only on bars [0..t].

The test is: infer the regime at bar t two ways —
  (a) using only features[0..t]
  (b) using features[0..T] (the full series, T > t), then taking row t
If implementation is causal, (a) and (b) agree at row t.
"""

from __future__ import annotations

from core.config import load_config
from data.feature_engineering import build_features, feature_spec_from_cfg
from core.hmm_engine import HMMEngine


def test_forward_inference_is_causal(synthetic_ohlcv):
    cfg = load_config()
    features = build_features(synthetic_ohlcv, cfg.features).dropna()
    spec = feature_spec_from_cfg(cfg.features)
    engine = HMMEngine(cfg.hmm, cfg.regime_labels.names, spec)
    engine.fit(features)

    # Pick a mid-series time t (after warmup + enough history).
    t_index = int(len(features) * 0.7)
    # (a) infer using only the prefix
    prefix = features.iloc[: t_index + 1]
    last_at_t = engine.infer_forward(prefix)

    # (b) full-series forward path, take element at the same time
    full_path = engine.infer_forward_path(features)
    same_t = full_path[t_index]

    assert last_at_t.state == same_t.state, (
        f"causality violation: prefix-only state {last_at_t.state} != "
        f"full-series state at same t {same_t.state}"
    )
    # Confidence should match closely (forward filter is deterministic in features prefix)
    assert abs(last_at_t.confidence - same_t.confidence) < 1e-9


def test_forward_path_independent_of_future_extension(synthetic_ohlcv):
    """Extending the series with more future bars must not change historical regimes."""
    cfg = load_config()
    features = build_features(synthetic_ohlcv, cfg.features).dropna()
    spec = feature_spec_from_cfg(cfg.features)
    engine = HMMEngine(cfg.hmm, cfg.regime_labels.names, spec)
    engine.fit(features)

    half = len(features) // 2
    short = features.iloc[: half + 50]
    long = features.iloc[: half + 200]

    short_path = engine.infer_forward_path(short)
    long_path = engine.infer_forward_path(long)

    # Compare overlapping prefix
    for i in range(len(short_path)):
        assert short_path[i].state == long_path[i].state, f"future bars changed past state at i={i}"
