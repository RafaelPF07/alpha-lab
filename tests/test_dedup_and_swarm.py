import asyncio

import numpy as np
import pandas as pd
from fakes import RoutingAnthropic, tool_response

from alpha_lab.checks.dedup import idea_similarity, normalised_code_hash
from alpha_lab.config import GatesConfig, load_settings
from alpha_lab.events.bus import EventBus
from alpha_lab.ledger.db import Ledger
from alpha_lab.pipeline.run import execute_run
from alpha_lab.pipeline.swarm import pick_themes

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
GOOD2 = GOOD.replace("prices.shift(21)", "prices.shift(22)")
IMPORTS_OS = "import os\n" + GOOD
LOW = {"concerns": [], "suspicion_level": "low", "suggested_change": "none"}
HIGH = {
    "concerns": ["uses future data"],
    "suspicion_level": "high",
    "suggested_change": "remove it",
}


def idea_for(n: int) -> dict:
    """Ideas that are clearly different from each other, for any n."""
    families = ["momentum", "volume shocks", "sector spreads", "earnings drift"]
    return {
        "name": f"{families[n % 4]} signal {n}",
        "hypothesis": f"Hypothesis {n}: asset {n * 101} shows a distinct, testable pattern.",
        "description": f"Rank stocks by a unique window of {n * 7 + 3} days, number {n}.",
        "expected_holding_period_days": 5 + n % 20,
    }


def relaxed_settings(concurrency: int = 1):
    settings = load_settings()
    return settings.model_copy(
        update={
            "gates": GatesConfig(
                min_train_sharpe=-100.0, min_validation_sharpe=0.3, max_turnover_daily=10.0
            ),
            "run": settings.run.model_copy(update={"max_concurrency": concurrency}),
        }
    )


def synthetic_prices() -> pd.DataFrame:
    rng = np.random.default_rng(3)
    index = pd.bdate_range("2010-01-01", "2018-12-31")
    log_returns = rng.normal(0.0003, 0.012, size=(len(index), 10))
    prices = 100.0 * np.exp(np.cumsum(log_returns, axis=0))
    return pd.DataFrame(prices, index=index, columns=[f"T{i}" for i in range(10)])


async def _run(client, ledger, n, budget, settings, run_id="r"):
    return await execute_run(
        run_id=run_id,
        settings=settings,
        prices=synthetic_prices(),
        n_pipelines=n,
        budget_usd=budget,
        ledger=ledger,
        bus=EventBus(ledger),
        client=client,
    )


def run(client, ledger, *, n=1, budget=5.0, concurrency=1, settings=None):
    settings = settings or relaxed_settings(concurrency)
    return asyncio.run(_run(client, ledger, n, budget, settings))


def test_code_hash_ignores_comments_and_formatting():
    a = "def signal(prices):\n    return prices * 0.0  # zero\n"
    b = "def signal( prices ):\n\n    return prices*0.0\n"
    assert normalised_code_hash(a) == normalised_code_hash(b)


def test_code_hash_changes_when_the_computation_changes():
    assert normalised_code_hash(GOOD) != normalised_code_hash(GOOD2)


def test_idea_similarity_is_high_for_copies_and_low_for_different_ideas():
    assert idea_similarity("A b c", "does a thing", "A b c", "does a thing") == 1.0
    assert idea_similarity("momentum", "buy winners", "volatility carry", "sell options") < 0.6


def test_themes_are_reproducible_from_the_seed():
    themes = ["a", "b", "c"]
    assert pick_themes(themes, 10, 42) == pick_themes(themes, 10, 42)
    assert set(pick_themes(themes, 50, 1)) == {"a", "b", "c"}


def test_duplicate_idea_is_rejected_and_regenerated(tmp_path):
    ledger = Ledger(tmp_path / "l.db")
    client = RoutingAnthropic(
        {
            # Pipeline 1 gets idea 1. Pipeline 2 first gets an exact copy of it.
            "submit_idea": [
                tool_response("submit_idea", idea_for(1)),
                tool_response("submit_idea", idea_for(1)),
            ],
            "submit_code": [
                tool_response("submit_code", {"code": GOOD}),
                tool_response("submit_code", {"code": GOOD}),  # duplicate, so rejected
                tool_response("submit_code", {"code": GOOD2}),  # the revision
            ],
            "submit_critique": [tool_response("submit_critique", LOW)] * 3,
        },
        idea_factory=lambda k: idea_for(k + 10),
    )
    summary = run(client, ledger, n=2, concurrency=1)
    assert ledger.count_rejected_ideas() == 1
    names = [row["name"] for row in ledger.all_ideas()]
    assert len(names) == len(set(names))  # the copy was never accepted
    assert all(r.status != "failed_ideas" for r in summary.results)


def test_identical_code_across_pipelines_is_rejected_and_not_a_trial(tmp_path):
    ledger = Ledger(tmp_path / "l.db")
    client = RoutingAnthropic(
        {
            "submit_code": [
                tool_response("submit_code", {"code": GOOD}),
                tool_response("submit_code", {"code": GOOD}),  # duplicate of the first
                tool_response("submit_code", {"code": GOOD2}),  # the revision
            ],
            "submit_critique": [tool_response("submit_critique", LOW)] * 2,
        },
        idea_factory=idea_for,
    )
    summary = run(client, ledger, n=2, concurrency=1)
    assert all(r.status == "critic_pass" for r in summary.results)
    rejected = ledger.conn.execute(
        "SELECT COUNT(*) FROM attempts WHERE status = 'rejected_duplicate_code'"
    ).fetchone()[0]
    assert rejected == 1
    # Every attempt that was not rejected reached the sandbox, so each one is a trial.
    assert ledger.count_trials() == len(ledger.all_attempts()) - rejected


def test_high_suspicion_from_critic_sends_the_coder_back(tmp_path):
    ledger = Ledger(tmp_path / "l.db")
    client = RoutingAnthropic(
        {
            "submit_code": [
                tool_response("submit_code", {"code": GOOD}),
                tool_response("submit_code", {"code": GOOD2}),
            ],
            "submit_critique": [
                tool_response("submit_critique", HIGH),
                tool_response("submit_critique", LOW),
            ],
        },
        idea_factory=idea_for,
    )
    summary = run(client, ledger, n=1)
    assert summary.results[0].status == "critic_pass"
    assert summary.results[0].attempts == 2
    first = ledger.attempts_for_run("r")[0]
    assert first["status"] == "failed_critic"
    assert "high suspicion" in first["error"]


def test_budget_stops_the_swarm_without_starting_new_pipelines(tmp_path):
    ledger = Ledger(tmp_path / "l.db")
    client = RoutingAnthropic(
        {"submit_code": [tool_response("submit_code", {"code": IMPORTS_OS})] * 50},
        idea_factory=idea_for,
    )
    # Each call reserves a worst-case 4000 output tokens, so only a few calls fit the budget.
    summary = run(client, ledger, n=6, budget=0.0025, concurrency=1)
    assert summary.status == "budget_stopped"
    assert len(summary.results) < 6
    assert ledger.get_run("r")["status"] == "budget_stopped"
