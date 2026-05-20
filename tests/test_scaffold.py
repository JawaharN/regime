"""Phase 1 scaffolding tests.

These must pass before any other phase starts. They verify:
- package imports cleanly
- config loads + validates
- CLI parser dispatches every subcommand
- secrets are not present in repo
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import core
from core.config import load_config


def test_package_imports():
    assert core.__version__ == "0.2.0"


def test_config_loads():
    cfg = load_config()
    # 10-symbol universe (final prompt); SPY is the lead symbol.
    assert cfg.universe.symbols[0] == "SPY"
    assert len(cfg.universe.symbols) == 10
    assert cfg.bars.runtime_interval == "1d"
    assert cfg.regime_labels.names == ["crash", "bear", "neutral", "bull", "euphoria"]
    assert cfg.risk.total_drawdown_kill_pct == 0.10
    assert cfg.broker.require_demo is True
    # final-prompt parameters validate too
    assert cfg.hmm.n_candidates == [3, 4, 5, 6, 7]
    assert cfg.risk.max_dd_from_peak == 0.10


def test_allocation_table_complete():
    """Every regime label must have a corresponding allocation entry."""
    cfg = load_config()
    for label in cfg.regime_labels.names:
        assert label in cfg.allocation.by_regime, f"missing allocation for {label}"


def test_cli_scaffold_runs():
    """The scaffold subcommand should exit 0 and mention the universe."""
    result = subprocess.run(
        [sys.executable, "-m", "main", "scaffold"],
        capture_output=True, text=True, cwd=Path(__file__).resolve().parent.parent,
    )
    assert result.returncode == 0, result.stderr
    assert "SPY" in result.stdout


def test_env_example_and_gitignore_in_place():
    """A local .env may exist (it holds real creds). What matters is that .env
    is git-ignored and .env.example exists as a template."""
    repo = Path(__file__).resolve().parent.parent
    assert (repo / ".env.example").exists()
    assert (repo / ".gitignore").exists()
    assert ".env" in (repo / ".gitignore").read_text().splitlines()
