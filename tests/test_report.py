import asyncio
import json

import numpy as np
import pandas as pd
from fakes import RoutingAnthropic, tool_response

from alpha_lab.config import GatesConfig, load_settings
from alpha_lab.events.bus import EventBus
from alpha_lab.ledger.db import Ledger
from alpha_lab.ledger.report import build_report_data, render_markdown, write_report
from alpha_lab.pipeline.run import execute_run

GOOD = """
import pandas as pd


def signal(prices: pd.DataFrame) -> pd.DataFrame:
    score = prices.shift(21) / prices.shift(252) - 1.0
    ranks = score.rank(axis=1, pct=True)
    long = (ranks > 0.8).astype(float)
    short = (ranks <= 0.2).astype(float)
    n_long = long.sum(axis=1).replace(0.0, float("nan"))
    n_short = short.sum(axis=1).replace(0.0, float("nan"))
    return (long.div(n_long, axis=0) - short.div(n_short, axis=0)).fillna(0.0)
"""
LOW = {"concerns": [], "suspicion_level": "low", "suggested_change": "none"}
IDEA = {
    "name": "Twelve month winners",
    "hypothesis": "Stocks that rose over the past year keep rising for a while.",
    "description": "Rank by return from 252 to 21 days ago. Long the top fifth, short the bottom fifth.",
    "expected_holding_period_days": 21,
}


def test_report_is_written_with_funnel_dsr_and_figures(tmp_path):
    settings = load_settings()
    settings = settings.model_copy(
        update={
            "gates": GatesConfig(
                min_train_sharpe=-100.0, min_validation_sharpe=-100.0, max_turnover_daily=10.0
            )
        }
    )
    index = pd.bdate_range("2010-01-01", "2021-12-31")
    rng = np.random.default_rng(23)
    prices = pd.DataFrame(
        100.0 * np.exp(np.cumsum(rng.normal(0.0003, 0.012, size=(len(index), 10)), axis=0)),
        index=index,
        columns=[f"T{i}" for i in range(10)],
    )
    ledger = Ledger(tmp_path / "report.db")
    client = RoutingAnthropic(
        {
            "submit_idea": [tool_response("submit_idea", IDEA)],
            "submit_code": [tool_response("submit_code", {"code": GOOD})],
            "submit_critique": [tool_response("submit_critique", LOW)],
        }
    )
    asyncio.run(
        execute_run(
            run_id="rep",
            settings=settings,
            prices=prices,
            n_pipelines=1,
            budget_usd=5.0,
            ledger=ledger,
            bus=EventBus(ledger),
            client=client,
        )
    )

    data = build_report_data(ledger, "rep")
    assert data.funnel[0] == ("ideas generated", 1)
    assert data.funnel[-1][0] == "passed validation"
    assert len(data.candidates) == 1
    assert np.isfinite(data.candidates[0].deflated_sharpe)

    markdown = render_markdown(data)
    for heading in [
        "## Funnel",
        "## Spend by agent and model",
        "## Multiple testing",
        "## Known limitations",
    ]:
        assert heading in markdown

    path = write_report(ledger, "rep", out_dir=tmp_path / "out")
    assert path.exists()
    assert (tmp_path / "out" / "report.html").exists()
    for name in ["galaxy.html", "equity.html"]:
        assert (tmp_path / "out" / "figures" / name).exists()
    assert json.loads(json.dumps(data.spend))  # spend rows are plain data
