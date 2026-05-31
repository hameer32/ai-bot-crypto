"""
HTML Dashboard Generator
=========================
Generates a self-refreshing HTML dashboard showing both models' live performance.
Saved to results/paper_trading/dashboard.html — open in any browser.
"""
from __future__ import annotations
import os
from datetime import datetime, timezone
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from live.paper_engine import PaperAccount


def generate_dashboard(accounts: dict[str, "PaperAccount"], output_path: str):
    """Generate the full HTML dashboard and write to output_path."""

    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")

    model_sections = ""
    for model_key, acc in accounts.items():
        s = acc.stats()
        cfg_color = "#2196F3" if "global" in model_key else "#4CAF50"
        model_sections += _model_section(acc, s, cfg_color)

    # Comparison table
    comparison = _comparison_table({k: v.stats() for k, v in accounts.items()})

    # Closed trades (all models combined, newest first)
    all_trades = []
    for acc in accounts.values():
        all_trades.extend(acc.closed_trades)
    all_trades.sort(key=lambda t: t.exit_time, reverse=True)
    trades_html = _trades_table(all_trades[:50])  # last 50

    # Open positions
    all_open = []
    for acc in accounts.values():
        all_open.extend([(acc.model_name, p) for p in acc.open_positions])
    open_html = _open_positions_table(all_open)

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <meta http-equiv="refresh" content="60">
  <title>Paper Trading Dashboard</title>
  <style>
    * {{ box-sizing: border-box; margin: 0; padding: 0; }}
    body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
            background: #0d1117; color: #e6edf3; padding: 20px; }}
    h1 {{ font-size: 1.6em; margin-bottom: 4px; color: #58a6ff; }}
    .updated {{ font-size: 0.8em; color: #8b949e; margin-bottom: 20px; }}
    .grid {{ display: grid; grid-template-columns: 1fr 1fr; gap: 16px; margin-bottom: 20px; }}
    .card {{ background: #161b22; border: 1px solid #30363d; border-radius: 8px; padding: 16px; }}
    .card h2 {{ font-size: 1.1em; margin-bottom: 12px; padding-bottom: 8px;
                border-bottom: 1px solid #30363d; }}
    .metric-grid {{ display: grid; grid-template-columns: repeat(3,1fr); gap: 8px; }}
    .metric {{ background: #0d1117; border-radius: 6px; padding: 10px; text-align: center; }}
    .metric .val {{ font-size: 1.4em; font-weight: 700; }}
    .metric .lbl {{ font-size: 0.7em; color: #8b949e; margin-top: 2px; }}
    .pos {{ color: #3fb950; }}
    .neg {{ color: #f85149; }}
    .neu {{ color: #e6edf3; }}
    .warn {{ color: #d29922; }}
    table {{ width: 100%; border-collapse: collapse; font-size: 0.82em; }}
    th {{ background: #21262d; padding: 8px 6px; text-align: left;
          color: #8b949e; border-bottom: 1px solid #30363d; }}
    td {{ padding: 6px; border-bottom: 1px solid #21262d; }}
    tr:hover {{ background: #1f2937; }}
    .badge {{ display: inline-block; padding: 2px 8px; border-radius: 12px;
              font-size: 0.75em; font-weight: 600; }}
    .badge-long {{ background: #1a4d2e; color: #3fb950; }}
    .badge-short {{ background: #4d1a1a; color: #f85149; }}
    .badge-win {{ background: #1a4d2e; color: #3fb950; }}
    .badge-loss {{ background: #4d1a1a; color: #f85149; }}
    .badge-open {{ background: #2d3748; color: #90cdf4; }}
    .section-title {{ font-size: 1.1em; font-weight: 600; margin: 20px 0 10px;
                      color: #58a6ff; border-left: 3px solid #58a6ff; padding-left: 10px; }}
    .kill-switch {{ background: #4d1a1a; color: #f85149; padding: 8px 12px;
                    border-radius: 6px; font-weight: 600; margin-bottom: 10px; }}
    @media (max-width: 768px) {{ .grid {{ grid-template-columns: 1fr; }} }}
  </style>
</head>
<body>
  <h1>📊 Paper Trading — Live Dashboard</h1>
  <p class="updated">Last updated: {now} &nbsp;|&nbsp; Auto-refreshes every 60 seconds</p>

  <div class="section-title">Model Performance</div>
  <div class="grid">
    {model_sections}
  </div>

  <div class="section-title">Model Comparison</div>
  {comparison}

  <div class="section-title">Open Positions ({len(all_open)})</div>
  {open_html}

  <div class="section-title">Recent Closed Trades (last 50)</div>
  {trades_html}

</body>
</html>"""

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        f.write(html)


def _model_section(acc, s: dict, color: str) -> str:
    kill = f'<div class="kill-switch">⚠️ KILL SWITCH ACTIVE — Max drawdown reached</div>' if s["kill_switch"] else ""

    def _cls(v): return "pos" if v > 0 else ("neg" if v < 0 else "neu")

    return f"""
<div class="card" style="border-top: 3px solid {color};">
  <h2 style="color:{color}">{acc.model_name}</h2>
  {kill}
  <div class="metric-grid">
    <div class="metric">
      <div class="val {_cls(s['return_pct'])}">{s['return_pct']:+.1f}%</div>
      <div class="lbl">Total Return</div>
    </div>
    <div class="metric">
      <div class="val" style="color:{color}">${s['capital']:,.2f}</div>
      <div class="lbl">Capital</div>
    </div>
    <div class="metric">
      <div class="val {_cls(s['total_pnl'])}">${s['total_pnl']:+,.2f}</div>
      <div class="lbl">Total P&L</div>
    </div>
    <div class="metric">
      <div class="val {'pos' if s['win_rate']>=50 else 'neg'}">{s['win_rate']:.1f}%</div>
      <div class="lbl">Win Rate</div>
    </div>
    <div class="metric">
      <div class="val {'pos' if s['avg_rr']>0 else 'neg'}">{s['avg_rr']:+.2f}R</div>
      <div class="lbl">Avg R:R</div>
    </div>
    <div class="metric">
      <div class="val {'neg' if s['max_dd']>5 else 'warn'}">{s['max_dd']:.1f}%</div>
      <div class="lbl">Max Drawdown</div>
    </div>
    <div class="metric">
      <div class="val neu">{s['total_trades']}</div>
      <div class="lbl">Total Trades</div>
    </div>
    <div class="metric">
      <div class="val" style="color:#58a6ff">{s['open_trades']}</div>
      <div class="lbl">Open Now</div>
    </div>
    <div class="metric">
      <div class="val {_cls(s['today_pnl'])}">${s['today_pnl']:+,.2f}</div>
      <div class="lbl">Today P&L</div>
    </div>
  </div>
  <div style="margin-top:10px; font-size:0.8em; color:#8b949e;">
    Wins: {s['wins']} &nbsp;|&nbsp; Losses: {s['losses']} &nbsp;|&nbsp; Today trades: {s['today_trades']}
  </div>
</div>"""


def _comparison_table(stats: dict) -> str:
    rows = ""
    metrics = [
        ("Return",        "return_pct",    lambda v: f"{v:+.1f}%",  True),
        ("Capital",       "capital",       lambda v: f"${v:,.2f}",  True),
        ("Win Rate",      "win_rate",      lambda v: f"{v:.1f}%",   True),
        ("Avg R:R",       "avg_rr",        lambda v: f"{v:+.2f}R",  True),
        ("Max Drawdown",  "max_dd",        lambda v: f"{v:.1f}%",   False),
        ("Total Trades",  "total_trades",  lambda v: str(v),        True),
        ("Open Now",      "open_trades",   lambda v: str(v),        None),
        ("Today P&L",     "today_pnl",     lambda v: f"${v:+,.2f}", True),
    ]

    header_keys = list(stats.keys())
    header_html = "".join(f"<th>{k}</th>" for k in header_keys)

    for label, key, fmt, higher_better in metrics:
        vals = {k: v.get(key, 0) for k, v in stats.items()}
        if higher_better is not None:
            best = max(vals.values()) if higher_better else min(vals.values())
        else:
            best = None

        cells = ""
        for k in header_keys:
            v = vals[k]
            is_best = (best is not None) and (v == best) and len(set(vals.values())) > 1
            style = ' style="color:#ffd700;font-weight:700"' if is_best else ""
            cells += f"<td{style}>{fmt(v)}</td>"
        rows += f"<tr><td><b>{label}</b></td>{cells}</tr>"

    return f"""<table>
  <thead><tr><th>Metric</th>{header_html}</tr></thead>
  <tbody>{rows}</tbody>
</table>"""


def _open_positions_table(open_positions: list) -> str:
    if not open_positions:
        return '<p style="color:#8b949e; padding:10px">No open positions</p>'

    rows = ""
    for model_name, pos in open_positions:
        dir_badge = f'<span class="badge badge-long">LONG</span>' if pos.direction == "long" else f'<span class="badge badge-short">SHORT</span>'
        entry_time = pos.entry_time[:16].replace("T", " ")
        conf = f"{pos.confidence:.2f}"
        rows += f"""<tr>
          <td>{pos.symbol}</td>
          <td>{dir_badge}</td>
          <td>{pos.entry_price:.6g}</td>
          <td class="neg">{pos.sl:.6g}</td>
          <td class="pos">{pos.tp1:.6g}</td>
          <td>${pos.dollar_risk:.2f}</td>
          <td>{conf}</td>
          <td style="color:#8b949e">{entry_time}</td>
          <td><span class="badge badge-open">{model_name}</span></td>
        </tr>"""

    return f"""<table>
  <thead><tr>
    <th>Symbol</th><th>Dir</th><th>Entry</th><th>SL</th><th>TP1</th>
    <th>$ Risk</th><th>Conf</th><th>Entry Time</th><th>Model</th>
  </tr></thead>
  <tbody>{rows}</tbody>
</table>"""


def _trades_table(trades: list) -> str:
    if not trades:
        return '<p style="color:#8b949e; padding:10px">No closed trades yet</p>'

    rows = ""
    for t in trades:
        dir_badge = f'<span class="badge badge-long">L</span>' if t.direction == "long" else f'<span class="badge badge-short">S</span>'
        outcome_badge = f'<span class="badge badge-{"win" if t.pnl > 0 else "loss"}">{t.outcome}</span>'
        pnl_cls = "pos" if t.pnl > 0 else "neg"
        rr_cls  = "pos" if t.rr > 0 else "neg"
        exit_time = t.exit_time[:16].replace("T", " ")
        rows += f"""<tr>
          <td>{t.symbol}</td>
          <td>{dir_badge}</td>
          <td>{t.entry_price:.6g}</td>
          <td>{t.exit_price:.6g}</td>
          <td class="{pnl_cls}">${t.pnl:+.2f}</td>
          <td class="{rr_cls}">{t.rr:+.2f}R</td>
          <td>{outcome_badge}</td>
          <td style="color:#8b949e; font-size:0.85em">{exit_time}</td>
          <td style="color:#8b949e; font-size:0.85em">{t.model_name[:10]}</td>
        </tr>"""

    return f"""<table>
  <thead><tr>
    <th>Symbol</th><th>Dir</th><th>Entry</th><th>Exit</th>
    <th>P&L</th><th>R:R</th><th>Outcome</th><th>Closed</th><th>Model</th>
  </tr></thead>
  <tbody>{rows}</tbody>
</table>"""
