"""Reproducible tabular exports and standalone interactive research figures."""

import html
import json
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import plotly.graph_objects as go

from aycapital.cross_asset.backtest import BacktestResult
from aycapital.cross_asset.config import ResearchConfig
from aycapital.cross_asset.macro import MacroData, macro_display_frame
from aycapital.cross_asset.market import MarketPanel
from aycapital.cross_asset.signals import SignalSet, latest_table


def _line_plot(data: pd.DataFrame, title: str, y_title: str) -> go.Figure:
    figure = go.Figure()
    for column in data:
        figure.add_trace(go.Scatter(x=data.index, y=data[column], name=str(column), mode="lines"))
    figure.update_layout(
        title=title,
        xaxis_title="Session/observation date",
        yaxis_title=y_title,
        template="plotly_white",
    )
    return figure


def export_report(
    config: ResearchConfig,
    panel: MarketPanel,
    signals: SignalSet,
    backtest: BacktestResult,
    output: str | Path,
    macro: dict[str, MacroData] | None = None,
) -> dict[str, Any]:
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    classes = {ticker: asset["asset_class"] for ticker, asset in config.assets.items()}
    latest = latest_table(signals, classes, panel.closes_utc.iloc[-1])
    latest["source"] = panel.metadata["source"]
    latest["currency"] = config.report_currency
    latest.to_csv(output / "latest_signals.csv", index_label="ticker")
    (output / "latest_signals.json").write_text(
        latest.reset_index(names="ticker").to_json(orient="records", indent=2)
    )
    history = pd.concat(signals.metrics | {"score": signals.score}, axis=1)
    history.to_csv(output / "signal_history.csv", index_label="session")
    panel.prices.to_csv(output / "adjusted_prices.csv", index_label="session")
    panel.closes_utc.to_csv(output / "session_closes.csv", index_label="session")
    for filename, frame in [
        ("backtest_returns", backtest.returns),
        ("equity", backtest.equity),
        ("turnover", backtest.turnover),
    ]:
        frame.join(backtest.periods).to_csv(output / f"{filename}.csv", index_label="session")
    backtest.statistics.to_csv(output / "performance.csv")
    backtest.trades.to_csv(output / "paper_trades.csv", index=False)
    correlation = panel.prices.pct_change(fill_method=None).rolling(60).corr()
    correlation.to_csv(output / "rolling_correlations.csv", index_label=["session", "ticker"])
    latest_correlation = correlation.loc[panel.prices.index[-1]]
    latest_correlation.to_csv(output / "latest_correlation.csv", index_label="ticker")
    figures = [_line_plot(signals.score, "Historical cross-asset evidence scores", "Score [0,1]")]
    figures.extend(
        _line_plot(data, name.replace("_", " "), "Historical metric")
        for name, data in signals.metrics.items()
    )
    heatmap = go.Figure(
        go.Heatmap(
            z=latest_correlation.to_numpy(),
            x=list(latest_correlation.columns),
            y=list(latest_correlation.index),
            zmin=-1,
            zmax=1,
            colorscale="RdBu",
            reversescale=True,
        )
    )
    heatmap.update_layout(title="Latest trailing 60-session return correlation")
    figures.append(heatmap)
    for label in ("in_sample", "out_of_sample"):
        figures.append(
            _line_plot(
                backtest.equity.loc[backtest.periods == label],
                f"{label}: independent account (costs included)",
                "NAV",
            )
        )
    macro_metadata = {}
    if macro:
        display = macro_display_frame(macro)
        display.to_csv(output / "macro_display.csv", index_label="observation_date")
        for series in display:
            title = config.macro.get("series", {}).get(series, series)
            figures.append(
                _line_plot(
                    display[[series]].dropna(), f"Macro descriptive snapshot: {title}", series
                )
            )
        for series, data in macro.items():
            data.observations.to_csv(output / f"macro_{series}_observations.csv", index=False)
            macro_metadata[series] = data.metadata
    metadata = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "signal_session": str(panel.prices.index[-1].date()),
        "signal_available_after_utc": panel.closes_utc.iloc[-1].isoformat(),
        "market": panel.metadata,
        "backtest": backtest.metadata,
        "configuration": asdict(config),
        "macro": macro_metadata,
        "disclaimer": "Research proxies; competition eligibility unknown. "
        "Historical/synthetic scores do not forecast 4–8 week returns.",
    }
    (output / "metadata.json").write_text(json.dumps(metadata, indent=2, default=str) + "\n")
    banner = (
        f"{panel.metadata['source'].upper()} research • {config.report_currency} • "
        "No predictive accuracy claim • Competition eligibility unknown"
    )
    fragments = [
        "<!doctype html><html><head><meta charset='utf-8'>",
        "<title>A-Y Cross-Asset Research</title></head><body>",
        f"<h1>A-Y Cross-Asset Research</h1><p>{html.escape(banner)}</p>",
        "<p>Signals use completed session closes. Paper execution occurs next session "
        "close. Macro charts are descriptive snapshots and do not enter the strategy.</p>",
        latest.to_html(float_format=lambda v: f"{v:.5f}"),
        "<h2>Historical/synthetic backtest statistics</h2>",
        backtest.statistics.to_html(float_format=lambda v: f"{v:.5f}"),
    ]
    fragments.extend(
        fig.to_html(full_html=False, include_plotlyjs=(i == 0)) for i, fig in enumerate(figures)
    )
    fragments.append("</body></html>")
    (output / "report.html").write_text("\n".join(fragments))
    return metadata
