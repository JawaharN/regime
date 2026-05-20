"""Rich terminal dashboard.

Reads ``state/dashboard.json`` (written by the main loop) and renders three
panels — REGIME, PORTFOLIO, POSITIONS — refreshed on a timer. It never calls the
broker: the main loop is the only writer, the dashboard is a pure reader.

    python main.py run --dashboard
"""

from __future__ import annotations

import json
import time
from pathlib import Path


def _load_state(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text())
    except (json.JSONDecodeError, OSError):
        return {}


def render_text(state: dict) -> str:
    """Plain-text fallback render (used when `rich` is unavailable / in tests)."""
    if not state:
        return "regime_trader — waiting for the main loop to write state/dashboard.json"
    lines = ["=== REGIME ==="]
    regime = state.get("regime", {})
    lines.append(
        f"  {regime.get('label', '?')}  p={regime.get('probability', 0):.2f}  "
        f"stable={regime.get('consecutive_bars', 0)} bars  "
        f"flicker={regime.get('flicker_count', 0)}  vol_rank={regime.get('vol_rank', '?')}"
    )
    lines.append("=== PORTFOLIO ===")
    pf = state.get("portfolio", {})
    lines.append(
        f"  equity={pf.get('equity', 0):,.2f}  daily_pnl={pf.get('daily_pnl', 0):+.2%}  "
        f"alloc={pf.get('allocation', 0):.0%}  leverage={pf.get('leverage', 1):.2f}x  "
        f"cash={pf.get('cash', 0):,.2f}"
    )
    lines.append("=== POSITIONS ===")
    for p in state.get("positions", []):
        lines.append(
            f"  {p.get('symbol', '?'):6s} entry={p.get('entry', 0):.2f} "
            f"now={p.get('current', 0):.2f} uPnL={p.get('unrealized_pnl', 0):+.2f} "
            f"stop={p.get('stop', 0):.2f} bars={p.get('holding_bars', 0)}"
        )
    if state.get("kill_switch"):
        lines.append("!! KILL SWITCH ARMED — main loop will not start")
    return "\n".join(lines)


def _build_layout(state: dict):
    from rich.panel import Panel
    from rich.table import Table

    regime = state.get("regime", {})
    regime_tbl = Table.grid(padding=(0, 2))
    regime_tbl.add_row("Label", str(regime.get("label", "?")))
    regime_tbl.add_row("Probability", f"{regime.get('probability', 0):.2f}")
    regime_tbl.add_row("Stable for", f"{regime.get('consecutive_bars', 0)} bars")
    regime_tbl.add_row("Flicker count", str(regime.get("flicker_count", 0)))
    regime_tbl.add_row("Vol rank", str(regime.get("vol_rank", "?")))

    pf = state.get("portfolio", {})
    pf_tbl = Table.grid(padding=(0, 2))
    pf_tbl.add_row("Equity", f"{pf.get('equity', 0):,.2f}")
    pf_tbl.add_row("Daily P&L", f"{pf.get('daily_pnl', 0):+.2%}")
    pf_tbl.add_row("Allocation", f"{pf.get('allocation', 0):.0%}")
    pf_tbl.add_row("Leverage", f"{pf.get('leverage', 1):.2f}x")
    pf_tbl.add_row("Cash", f"{pf.get('cash', 0):,.2f}")
    pf_tbl.add_row("Buying power", f"{pf.get('buying_power', 0):,.2f}")

    pos_tbl = Table(expand=True)
    for col in ("Symbol", "Entry", "Current", "uP&L", "Stop", "Bars", "Regime@entry"):
        pos_tbl.add_column(col)
    for p in state.get("positions", []):
        pos_tbl.add_row(
            str(p.get("symbol", "?")), f"{p.get('entry', 0):.2f}",
            f"{p.get('current', 0):.2f}", f"{p.get('unrealized_pnl', 0):+.2f}",
            f"{p.get('stop', 0):.2f}", str(p.get("holding_bars", 0)),
            str(p.get("regime_at_entry", "?")),
        )

    from rich.console import Group
    return Group(
        Panel(regime_tbl, title="REGIME", border_style="cyan"),
        Panel(pf_tbl, title="PORTFOLIO", border_style="green"),
        Panel(pos_tbl, title="POSITIONS", border_style="magenta"),
    )


def run_dashboard(state_path: str = "state/dashboard.json",
                  refresh_seconds: int = 5, iterations: int = 0) -> int:
    """Launch the Rich live dashboard. `iterations`>0 caps refreshes (for tests)."""
    path = Path(state_path)
    try:
        from rich.live import Live
    except ImportError:
        # Plain-text fallback.
        count = 0
        while iterations <= 0 or count < iterations:
            print("\033[2J\033[H" + render_text(_load_state(path)))
            count += 1
            if iterations > 0 and count >= iterations:
                break
            time.sleep(refresh_seconds)
        return 0

    with Live(_build_layout(_load_state(path)), refresh_per_second=2, screen=False) as live:
        count = 0
        while iterations <= 0 or count < iterations:
            live.update(_build_layout(_load_state(path)))
            count += 1
            if iterations > 0 and count >= iterations:
                break
            time.sleep(refresh_seconds)
    return 0
