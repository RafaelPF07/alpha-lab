"""Regression test: two pipelines running at the same time must not both accept the same idea."""

import asyncio
import hashlib

import numpy as np
import pandas as pd
from fakes import RoutingAnthropic, tool_response

from alpha_lab.config import GatesConfig, load_settings
from alpha_lab.events.bus import EventBus
from alpha_lab.ledger.db import Ledger
from alpha_lab.pipeline.run import execute_run


def idea_for(n: int) -> dict:
    """Ideas whose wording is unrelated to each other (hash-derived), so only true copies match."""
    digest = hashlib.sha256(f"idea-{n}".encode()).hexdigest()
    return {
        "name": f"signal {digest[:12]}",
        "hypothesis": f"Hypothesis built on {digest[12:40]} and an untested story.",
        "description": f"Rank stocks by {digest[40:64]} over a window of {n} days.",
        "expected_holding_period_days": 5,
    }


def test_concurrent_pipelines_do_not_accept_the_same_idea(tmp_path):
    settings = load_settings()
    settings = settings.model_copy(
        update={
            "gates": GatesConfig(
                min_train_sharpe=-100.0, min_validation_sharpe=0.3, max_turnover_daily=10.0
            ),
            "run": settings.run.model_copy(update={"max_concurrency": 2}),
        }
    )
    index = pd.bdate_range("2010-01-01", "2012-12-31")
    rng = np.random.default_rng(9)
    prices = pd.DataFrame(
        100.0 * np.exp(np.cumsum(rng.normal(0.0, 0.01, size=(len(index), 4)), axis=0)),
        index=index,
        columns=[f"T{i}" for i in range(4)],
    )
    # Both pipelines get the same idea first. The coder always returns code that fails the
    # static check, so the test stays fast and only exercises the idea stage.
    bad_code = [
        tool_response(
            "submit_code",
            {
                "code": "import os"
                + chr(10)
                + f"x{i} = 1"
                + chr(10)
                + "def signal(prices):"
                + chr(10)
                + "    return prices"
            },
        )
        for i in range(8)
    ]
    client = RoutingAnthropic(
        {
            "submit_idea": [
                tool_response("submit_idea", idea_for(1)),
                tool_response("submit_idea", idea_for(1)),
            ],
            "submit_code": bad_code,
        },
        idea_factory=lambda k: idea_for(k + 10),
    )
    ledger = Ledger(tmp_path / "race.db")
    asyncio.run(
        execute_run(
            run_id="race",
            settings=settings,
            prices=prices,
            n_pipelines=2,
            budget_usd=5.0,
            ledger=ledger,
            bus=EventBus(ledger),
            client=client,
        )
    )
    names = [row["name"] for row in ledger.all_ideas()]
    assert len(names) == len(set(names)), f"duplicate ideas were accepted: {names}"
    assert ledger.count_rejected_ideas() == 1
