"""Dashboard renderers and LAN-accessible HTML server.

Reads ``state/dashboard.json`` (written by the main loop) and serves a small HTML
monitoring page plus a JSON endpoint. The dashboard never calls the broker: the
main loop is the only writer, the dashboard is a pure reader.

    python main.py dashboard
"""

from __future__ import annotations

import html
import json
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any


WAITING_MESSAGE = "regime_trader — waiting for the main loop to write state/dashboard.json"


def _load_state(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text())
    except (json.JSONDecodeError, OSError):
        return {}


def render_text(state: dict) -> str:
    """Plain-text fallback render kept for tests and quick terminal checks."""
    if not state:
        return WAITING_MESSAGE
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


def _fmt_money(value: Any) -> str:
    try:
        return f"${float(value):,.2f}"
    except (TypeError, ValueError):
        return "$0.00"


def _fmt_pct(value: Any, digits: int = 2) -> str:
    try:
        return f"{float(value):+.{digits}%}"
    except (TypeError, ValueError):
        return f"{0:+.{digits}%}"


def _fmt_ratio(value: Any) -> str:
    try:
        return f"{float(value):.2f}x"
    except (TypeError, ValueError):
        return "1.00x"


def _risk_badge(ok: bool) -> tuple[str, str]:
    return ("ok", "✅") if ok else ("bad", "⚠️")


def render_html(state: dict, refresh_seconds: int = 5) -> str:
    regime = state.get("regime", {})
    portfolio = state.get("portfolio", {})
    positions = state.get("positions", [])
    recent_signals = state.get("recent_signals", [])
    risk_status = state.get("risk_status", {})
    system = state.get("system", {})

    if not state:
        body = f"<div class='empty'>{html.escape(WAITING_MESSAGE)}</div>"
    else:
        daily_dd_ok = risk_status.get("daily_drawdown", {}).get("ok", True)
        peak_dd_ok = risk_status.get("from_peak", {}).get("ok", True)
        daily_cls, daily_icon = _risk_badge(daily_dd_ok)
        peak_cls, peak_icon = _risk_badge(peak_dd_ok)

        position_rows = "".join(
            "<tr>"
            f"<td>{html.escape(str(p.get('symbol', '?')))}</td>"
            f"<td>{html.escape(str(p.get('side', 'LONG')))}</td>"
            f"<td>{_fmt_money(p.get('current', p.get('entry', 0)))}</td>"
            f"<td>{_fmt_pct(p.get('unrealized_pnl_pct', p.get('unrealized_pnl', 0)), 2)}</td>"
            f"<td>{_fmt_money(p.get('stop', 0))}</td>"
            f"<td>{html.escape(str(p.get('holding_bars', 0)))}</td>"
            "</tr>"
            for p in positions
        ) or "<tr><td colspan='6' class='muted'>No open positions</td></tr>"

        signal_rows = "".join(
            "<tr>"
            f"<td>{html.escape(str(sig.get('time', '--:--')))}</td>"
            f"<td>{html.escape(str(sig.get('symbol', '-')))}</td>"
            f"<td>{html.escape(str(sig.get('action', '-')))}</td>"
            f"<td>{html.escape(str(sig.get('reason', '-')))}</td>"
            "</tr>"
            for sig in recent_signals
        ) or "<tr><td colspan='4' class='muted'>No recent signals</td></tr>"

        body = f"""
<div class="grid">
  <section class="panel">
    <h2>REGIME</h2>
    <div class="metric-line">
      <strong id="regime-label">{html.escape(str(regime.get('label', '?')))}</strong>
      <span id="regime-probability">({_fmt_pct(regime.get('probability', 0), 0)})</span>
    </div>
    <div class="subtle">Stability: <span id="regime-stability">{html.escape(str(regime.get('consecutive_bars', 0)))}</span> bars</div>
    <div class="subtle">Flicker: <span id="regime-flicker">{html.escape(str(regime.get('flicker_count', 0)))}</span> | Vol rank: <span id="regime-vol-rank">{html.escape(str(regime.get('vol_rank', '?')))}</span></div>
  </section>

  <section class="panel">
    <h2>PORTFOLIO</h2>
    <div class="metric-line">Equity: <strong id="portfolio-equity">{_fmt_money(portfolio.get('equity', 0))}</strong></div>
    <div class="subtle">Daily: <span id="portfolio-daily-pnl">{_fmt_pct(portfolio.get('daily_pnl', 0), 2)}</span></div>
    <div class="subtle">Allocation: <span id="portfolio-allocation">{_fmt_pct(portfolio.get('allocation', 0), 0)}</span> | Leverage: <span id="portfolio-leverage">{_fmt_ratio(portfolio.get('leverage', 1))}</span></div>
    <div class="subtle">Cash: <span id="portfolio-cash">{_fmt_money(portfolio.get('cash', 0))}</span> | Buying power: <span id="portfolio-buying-power">{_fmt_money(portfolio.get('buying_power', 0))}</span></div>
  </section>

  <section class="panel wide">
    <h2>POSITIONS</h2>
    <table>
      <thead><tr><th>Symbol</th><th>Side</th><th>Price</th><th>P&amp;L</th><th>Stop</th><th>Bars</th></tr></thead>
      <tbody id="positions-body">{position_rows}</tbody>
    </table>
  </section>

  <section class="panel wide">
    <h2>RECENT SIGNALS</h2>
    <table>
      <thead><tr><th>Time</th><th>Symbol</th><th>Action</th><th>Reason</th></tr></thead>
      <tbody id="signals-body">{signal_rows}</tbody>
    </table>
  </section>

  <section class="panel">
    <h2>RISK STATUS</h2>
    <div class="risk {daily_cls}">Daily DD: {html.escape(str(risk_status.get('daily_drawdown', {}).get('value', '0%')))} / {html.escape(str(risk_status.get('daily_drawdown', {}).get('limit', '3%')))} {daily_icon}</div>
    <div class="risk {peak_cls}">From Peak: {html.escape(str(risk_status.get('from_peak', {}).get('value', '0%')))} / {html.escape(str(risk_status.get('from_peak', {}).get('limit', '10%')))} {peak_icon}</div>
    <div class="risk {'bad' if state.get('kill_switch') else 'ok'}">Kill switch: {'ARMED' if state.get('kill_switch') else 'clear'}</div>
  </section>

  <section class="panel">
    <h2>SYSTEM</h2>
    <div class="subtle">Data: {html.escape(str(system.get('data', 'unknown')))}</div>
    <div class="subtle">API: {html.escape(str(system.get('api', 'unknown')))}</div>
    <div class="subtle">HMM: {html.escape(str(system.get('hmm', 'unknown')))}</div>
    <div class="subtle">Mode: {html.escape(str(system.get('mode', 'PAPER')))}</div>
  </section>
</div>
"""

    payload = json.dumps(state)
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>regime_trader dashboard</title>
  <style>
    :root {{ color-scheme: dark; }}
    body {{ font-family: Arial, sans-serif; background: #0b1220; color: #e5edf7; margin: 0; padding: 24px; }}
    h1 {{ margin: 0 0 16px; font-size: 28px; }}
    h2 {{ margin: 0 0 12px; font-size: 16px; letter-spacing: 0.08em; color: #8ab4ff; }}
    .grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(320px, 1fr)); gap: 16px; }}
    .panel {{ background: #121a2b; border: 1px solid #24324d; border-radius: 12px; padding: 16px; box-shadow: 0 8px 24px rgba(0,0,0,0.25); }}
    .wide {{ grid-column: 1 / -1; }}
    .metric-line {{ font-size: 20px; margin-bottom: 8px; }}
    .subtle {{ color: #b7c4d6; margin: 6px 0; }}
    .empty {{ background: #121a2b; border: 1px dashed #40506f; border-radius: 12px; padding: 24px; color: #b7c4d6; }}
    .muted {{ color: #8ea0b7; text-align: center; }}
    table {{ width: 100%; border-collapse: collapse; }}
    th, td {{ text-align: left; padding: 10px 8px; border-bottom: 1px solid #22304c; }}
    th {{ color: #8ea0b7; font-weight: 600; font-size: 13px; }}
    .risk {{ margin: 8px 0; padding: 10px 12px; border-radius: 10px; font-weight: 600; }}
    .risk.ok {{ background: rgba(32, 201, 151, 0.16); color: #74f0c0; }}
    .risk.bad {{ background: rgba(255, 107, 107, 0.16); color: #ff9b9b; }}
    .footer {{ margin-top: 14px; color: #8ea0b7; font-size: 13px; }}
  </style>
</head>
<body>
  <h1>regime_trader dashboard</h1>
  {body}
  <div class="footer">Auto-refresh every {refresh_seconds} seconds.</div>
  <script>
    const initialState = {payload};
    async function refreshDashboard() {{
      try {{
        const response = await fetch('/api/state', {{ cache: 'no-store' }});
        if (!response.ok) return;
        const nextState = await response.json();
        if (JSON.stringify(nextState) !== JSON.stringify(initialState)) {{
          window.location.reload();
        }}
      }} catch (_err) {{
        // keep the last rendered view visible
      }}
    }}
    setInterval(refreshDashboard, {max(refresh_seconds, 1) * 1000});
  </script>
</body>
</html>
"""


def _make_handler(state_path: Path, refresh_seconds: int):
    class DashboardHandler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:  # noqa: N802
            if self.path in ("/", "/index.html"):
                page = render_html(_load_state(state_path), refresh_seconds=refresh_seconds).encode()
                self.send_response(HTTPStatus.OK)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(page)))
                self.end_headers()
                self.wfile.write(page)
                return
            if self.path == "/api/state":
                payload = json.dumps(_load_state(state_path)).encode()
                self.send_response(HTTPStatus.OK)
                self.send_header("Content-Type", "application/json; charset=utf-8")
                self.send_header("Cache-Control", "no-store")
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)
                return
            self.send_error(HTTPStatus.NOT_FOUND, "Not found")

        def log_message(self, format: str, *args: Any) -> None:
            return

    return DashboardHandler


def run_dashboard_server(state_path: str = "state/dashboard.json",
                         host: str = "0.0.0.0",
                         port: int = 8000,
                         refresh_seconds: int = 5) -> int:
    path = Path(state_path)
    server = ThreadingHTTPServer((host, port), _make_handler(path, refresh_seconds))
    print(f"[dashboard] serving HTTP on http://{host}:{port}")
    if host == "0.0.0.0":
        print(f"[dashboard] open from another PC: http://192.168.2.40:{port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n[dashboard] stopping")
    finally:
        server.server_close()
    return 0
