"""HMM regime detection engine.

The HMM is a **volatility classifier**, not a price predictor. We fit a Gaussian
HMM on the causal feature matrix and, at live time, infer the current regime
with the **forward (filtered) algorithm only** — the regime at bar t depends
solely on features[0..t]. We never call ``model.predict()`` (Viterbi) on live
data: Viterbi smooths with future observations and leaks look-ahead.

Model selection is **pure BIC**:

    BIC = -2 * log_likelihood + n_params * log(n_samples)

For each candidate state count we run ``n_init`` random restarts, keep the best
log-likelihood, and pick the candidate with the lowest BIC. Covariance is
``full`` per the prompt; if a full-covariance fit is singular on a short
history we fall back to ``diag`` for that candidate so selection still works.
"""

from __future__ import annotations

import logging
import pickle
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from hmmlearn.hmm import GaussianHMM

from core.regime_labeling import label_states, label_states_by_count
from data.feature_engineering import FEATURE_COLUMNS, FeatureSpec

logger = logging.getLogger("regime_trader.hmm")


@dataclass
class RegimeInference:
    """Lightweight inference result (backward-compatible)."""

    state: int
    label: str
    confidence: float


@dataclass
class RegimeInfo:
    """Per-regime profile derived at training time. Drives strategy routing."""

    regime_id: int
    regime_name: str
    expected_return: float
    expected_volatility: float
    recommended_strategy_type: str          # low_vol | mid_vol | high_vol
    max_leverage_allowed: float
    max_position_size_pct: float
    min_confidence_to_act: float


@dataclass
class RegimeState:
    """Rich live-inference result combined with stability information."""

    label: str
    state_id: int
    probability: float
    state_probabilities: list[float] = field(default_factory=list)
    timestamp: datetime | None = None
    is_confirmed: bool = True
    consecutive_bars: int = 1


def _n_params(model: GaussianHMM) -> int:
    """Free-parameter count for a fitted GaussianHMM (for BIC)."""
    n = model.n_components
    d = model.n_features
    start = n - 1
    trans = n * (n - 1)
    means = n * d
    if model.covariance_type == "full":
        cov = n * d * (d + 1) // 2
    elif model.covariance_type == "tied":
        cov = d * (d + 1) // 2
    elif model.covariance_type == "spherical":
        cov = n
    else:  # diag
        cov = n * d
    return start + trans + means + cov


class HMMEngine:
    def __init__(self, hmm_cfg, regime_labels: list[str], feature_spec: FeatureSpec) -> None:  # noqa: ANN001
        self.hmm_cfg = hmm_cfg
        self.regime_labels = regime_labels
        self.feature_spec = feature_spec
        self.model: GaussianHMM | None = None
        self.state_to_label: dict[int, str] = {}
        self.n_components: int | None = None
        self.training_returns_: dict[int, float] = {}
        self.bic_scores_: dict[int, float] = {}
        self.regime_infos: list[RegimeInfo] = []
        self.trained_at: datetime | None = None

    # ------------------------------------------------------------------ fit

    def _candidates(self) -> list[int]:
        cands = list(getattr(self.hmm_cfg, "n_candidates", []) or [])
        if not cands:
            cands = list(range(self.hmm_cfg.n_components_min, self.hmm_cfg.n_components_max + 1))
        return sorted(set(cands))

    def _fit_one(self, x: np.ndarray, n_components: int) -> tuple[GaussianHMM | None, float]:
        """Fit `n_init` restarts; return (best model, best log-likelihood)."""
        best_model: GaussianHMM | None = None
        best_ll = -np.inf
        n_init = max(1, getattr(self.hmm_cfg, "n_init", 1))
        for i in range(n_init):
            for cov_type in (self.hmm_cfg.covariance_type, "diag"):
                try:
                    m = GaussianHMM(
                        n_components=n_components,
                        covariance_type=cov_type,
                        n_iter=self.hmm_cfg.n_iter,
                        tol=self.hmm_cfg.tol,
                        random_state=self.hmm_cfg.random_state + i,
                    )
                    m.fit(x)
                    ll = float(m.score(x))
                    if np.isfinite(ll) and ll > best_ll:
                        best_ll, best_model = ll, m
                    break  # this cov_type worked — don't also try diag
                except Exception:  # noqa: BLE001 — singular cov / non-convergence
                    continue
        return best_model, best_ll

    def fit(self, features: pd.DataFrame, returns_for_labeling: pd.Series | None = None) -> None:
        """Train the HMM, selecting the state count by pure BIC."""
        x = self._validate_features(features)
        n_samples = len(x)

        if returns_for_labeling is None:
            ret = features.get("ret_1")
            if ret is None:
                ret = pd.Series(0.0, index=features.index)
            returns_for_labeling = ret
        clean_idx = features[list(FEATURE_COLUMNS)].dropna().index
        ret_aligned = returns_for_labeling.reindex(clean_idx).to_numpy()

        best_bic = np.inf
        best_model: GaussianHMM | None = None
        best_n = None
        self.bic_scores_ = {}
        for n_components in self._candidates():
            model, ll = self._fit_one(x, n_components)
            if model is None:
                logger.warning("HMM candidate n=%d failed to fit", n_components)
                continue
            bic = -2.0 * ll + _n_params(model) * np.log(n_samples)
            self.bic_scores_[n_components] = float(bic)
            logger.info("HMM candidate n=%d: ll=%.1f BIC=%.1f cov=%s",
                        n_components, ll, bic, model.covariance_type)
            if bic < best_bic:
                best_bic, best_model, best_n = bic, model, n_components

        if best_model is None:
            raise RuntimeError("No HMM converged across the candidate range")

        self.model = best_model
        self.n_components = best_n
        logger.info("HMM selected n=%d (BIC=%.1f)", best_n, best_bic)

        # Per-state training statistics (training-time predict is fine here).
        states = self.model.predict(x)
        state_returns: dict[int, float] = {}
        state_vols: dict[int, float] = {}
        for s in range(best_n):
            mask = states == s
            if mask.any():
                state_returns[int(s)] = float(np.nanmean(ret_aligned[mask]))
                state_vols[int(s)] = float(np.nanstd(ret_aligned[mask]))
        self.training_returns_ = state_returns
        self.state_to_label = label_states(state_returns, self.regime_labels)
        self.regime_infos = self._build_regime_infos(state_returns, state_vols)
        self.trained_at = datetime.now(timezone.utc)

    def _build_regime_infos(self, state_returns: dict[int, float],
                            state_vols: dict[int, float]) -> list[RegimeInfo]:
        if not state_vols:
            return []
        # Rank by expected volatility ascending → strategy tier via rank/(n-1).
        ordered = sorted(state_vols.items(), key=lambda kv: kv[1])
        n = len(ordered)
        names = label_states_by_count(state_returns)
        infos: list[RegimeInfo] = []
        for rank, (sid, vol) in enumerate(ordered):
            pos = rank / (n - 1) if n > 1 else 0.0
            if pos <= 0.33:
                stype, lev, max_pos = "low_vol", 1.25, 0.95
            elif pos >= 0.67:
                stype, lev, max_pos = "high_vol", 1.0, 0.60
            else:
                stype, lev, max_pos = "mid_vol", 1.0, 0.95
            infos.append(RegimeInfo(
                regime_id=sid,
                regime_name=names.get(sid, self.state_to_label.get(sid, str(sid))),
                expected_return=state_returns.get(sid, 0.0),
                expected_volatility=vol,
                recommended_strategy_type=stype,
                max_leverage_allowed=lev,
                max_position_size_pct=max_pos,
                min_confidence_to_act=getattr(self.hmm_cfg, "min_confidence", 0.55),
            ))
        return sorted(infos, key=lambda r: r.regime_id)

    # ----------------------------------------------------------- inference

    def infer_forward(self, features: pd.DataFrame) -> RegimeInference:
        """Forward-only inference: regime at the last bar of `features`."""
        if self.model is None:
            raise RuntimeError("HMMEngine.fit() must be called before infer_forward()")
        x = self._validate_features(features)
        log_alpha = self._forward_log_alpha(x)
        last = log_alpha[-1] - np.max(log_alpha[-1])
        posterior = np.exp(last) / np.exp(last).sum()
        state = int(np.argmax(posterior))
        return RegimeInference(
            state=state,
            label=self.state_to_label.get(state, "unknown"),
            confidence=float(posterior[state]),
        )

    def infer_regime_state(self, features: pd.DataFrame,
                           timestamp: datetime | None = None) -> RegimeState:
        """Rich forward-only inference returning the full posterior."""
        if self.model is None:
            raise RuntimeError("HMMEngine.fit() must be called before inference")
        x = self._validate_features(features)
        log_alpha = self._forward_log_alpha(x)
        last = log_alpha[-1] - np.max(log_alpha[-1])
        posterior = np.exp(last) / np.exp(last).sum()
        state = int(np.argmax(posterior))
        return RegimeState(
            label=self.state_to_label.get(state, "unknown"),
            state_id=state,
            probability=float(posterior[state]),
            state_probabilities=[float(p) for p in posterior],
            timestamp=timestamp,
        )

    def infer_forward_path(self, features: pd.DataFrame) -> list[RegimeInference]:
        """Forward-filter every bar; each bar's regime uses only its own past."""
        if self.model is None:
            raise RuntimeError("HMMEngine.fit() must be called before infer_forward_path()")
        x = self._validate_features(features)
        log_alpha = self._forward_log_alpha(x)
        log_alpha = log_alpha - log_alpha.max(axis=1, keepdims=True)
        posterior = np.exp(log_alpha)
        posterior = posterior / posterior.sum(axis=1, keepdims=True)
        states = posterior.argmax(axis=1)
        confs = posterior[np.arange(len(states)), states]
        return [
            RegimeInference(state=int(s), label=self.state_to_label.get(int(s), "unknown"),
                            confidence=float(c))
            for s, c in zip(states, confs, strict=True)
        ]

    # ------------------------------------------------------ persistence

    def save(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "model": self.model,
            "state_to_label": self.state_to_label,
            "n_components": self.n_components,
            "training_returns": self.training_returns_,
            "bic_scores": self.bic_scores_,
            "regime_infos": self.regime_infos,
            "trained_at": self.trained_at,
            "feature_spec_hash": self.feature_spec.hash(),
            "regime_labels": self.regime_labels,
        }
        with open(path, "wb") as f:
            pickle.dump(payload, f)

    @classmethod
    def load(cls, path: str | Path, hmm_cfg, feature_spec: FeatureSpec) -> HMMEngine:  # noqa: ANN001
        with open(path, "rb") as f:
            payload = pickle.load(f)
        if payload["feature_spec_hash"] != feature_spec.hash():
            raise RuntimeError(
                f"Feature spec hash mismatch: model was trained with "
                f"{payload['feature_spec_hash']}, current is {feature_spec.hash()}. "
                "Retrain the model."
            )
        eng = cls(hmm_cfg, payload["regime_labels"], feature_spec)
        eng.model = payload["model"]
        eng.state_to_label = payload["state_to_label"]
        eng.n_components = payload["n_components"]
        eng.training_returns_ = payload.get("training_returns", {})
        eng.bic_scores_ = payload.get("bic_scores", {})
        eng.regime_infos = payload.get("regime_infos", [])
        eng.trained_at = payload.get("trained_at")
        return eng

    # --------------------------------------------------- internals

    def _validate_features(self, features: pd.DataFrame) -> np.ndarray:
        missing = [c for c in FEATURE_COLUMNS if c not in features.columns]
        if missing:
            raise ValueError(f"Features missing columns: {missing}")
        df = features[list(FEATURE_COLUMNS)].dropna()
        if df.empty:
            raise ValueError("No non-NaN rows in features — increase history or shrink windows")
        return df.to_numpy()

    def _forward_log_alpha(self, x: np.ndarray) -> np.ndarray:
        """Strictly causal forward log-alpha recursion.

        log_alpha[t, j] = log P(x[0..t], z_t = j)
        """
        assert self.model is not None
        n_obs = len(x)
        n_states = self.model.n_components
        log_emiss = self.model._compute_log_likelihood(x)
        log_startprob = np.log(self.model.startprob_ + 1e-300)
        log_transmat = np.log(self.model.transmat_ + 1e-300)

        log_alpha = np.full((n_obs, n_states), -np.inf)
        log_alpha[0] = log_startprob + log_emiss[0]
        for t in range(1, n_obs):
            prev = log_alpha[t - 1][:, None] + log_transmat
            log_alpha[t] = log_emiss[t] + _logsumexp_rows(prev)
        return log_alpha


def _logsumexp_rows(a: np.ndarray) -> np.ndarray:
    """logsumexp along axis 0 of a (K, K) matrix → (K,)."""
    m = a.max(axis=0)
    return m + np.log(np.exp(a - m).sum(axis=0))
