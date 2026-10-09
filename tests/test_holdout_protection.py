"""Holdout protection: needs the flag, needs confirmation, runs once per signal, and is recorded first."""

import numpy as np
import pandas as pd
import pytest
from typer.testing import CliRunner

from alpha_lab.cli import app
from alpha_lab.config import load_settings
from alpha_lab.data.loader import write_cache
from alpha_lab.ledger.db import Ledger
from alpha_lab.pipeline.holdout import HoldoutCandidate, HoldoutRefused, run_holdout

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
RAISES = "def signal(prices):\n    raise ValueError('boom')\n"


@pytest.fixture
def world(tmp_path):
    settings = load_settings()
    index = pd.bdate_range("2010-01-01", settings.dates.holdout_end)
    rng = np.random.default_rng(31)
    prices = pd.DataFrame(
        100.0 * np.exp(np.cumsum(rng.normal(0.0003, 0.012, size=(len(index), 6)), axis=0)),
        index=index,
        columns=[f"T{i}" for i in range(6)],
    )
    data_dir = tmp_path / "data"
    write_cache(prices, settings.dates, data_dir)
    holdout_days = int((index > pd.Timestamp(settings.dates.validation_end)).sum())
    ledger = Ledger(tmp_path / "ledger.db")
    ledger.create_run("run", "{}", 1.0)
    return settings, ledger, data_dir, holdout_days


def add_attempt(ledger, attempt_id, code):
    ledger.add_attempt(
        attempt_id=attempt_id,
        idea_id="idea",
        run_id="run",
        attempt_number=1,
        code=code,
        code_hash=attempt_id,
        status="validation_pass",
    )
    return HoldoutCandidate(attempt_id=attempt_id, name=attempt_id, code=code)


def yes(_):
    return True


def no(_):
    return False


def test_refuses_without_the_flag_and_records_nothing(world):
    settings, ledger, data_dir, _ = world
    candidate = add_attempt(ledger, "a1", GOOD)
    with pytest.raises(HoldoutRefused, match="i-understand"):
        run_holdout(
            ledger=ledger,
            run_id="run",
            settings=settings,
            candidates=[candidate],
            i_understand_this_is_final=False,
            confirm=yes,
            data_dir=data_dir,
        )
    assert ledger.holdouts() == []


def test_declining_the_confirmation_records_nothing(world):
    settings, ledger, data_dir, _ = world
    candidate = add_attempt(ledger, "a1", GOOD)
    with pytest.raises(HoldoutRefused, match="not confirmed"):
        run_holdout(
            ledger=ledger,
            run_id="run",
            settings=settings,
            candidates=[candidate],
            i_understand_this_is_final=True,
            confirm=no,
            data_dir=data_dir,
        )
    assert ledger.holdouts() == []


def test_runs_once_on_holdout_days_only(world):
    settings, ledger, data_dir, holdout_days = world
    candidate = add_attempt(ledger, "a1", GOOD)
    results = run_holdout(
        ledger=ledger,
        run_id="run",
        settings=settings,
        candidates=[candidate],
        i_understand_this_is_final=True,
        confirm=yes,
        data_dir=data_dir,
    )
    assert results[0].status == "complete"
    assert results[0].metrics["n_days"] == holdout_days  # only holdout returns were scored
    assert ledger.get_holdout("a1")["status"] == "complete"


def test_cannot_run_twice_on_the_same_signal(world):
    settings, ledger, data_dir, _ = world
    candidate = add_attempt(ledger, "a1", GOOD)
    run_holdout(
        ledger=ledger,
        run_id="run",
        settings=settings,
        candidates=[candidate],
        i_understand_this_is_final=True,
        confirm=yes,
        data_dir=data_dir,
    )
    with pytest.raises(HoldoutRefused, match="already been run"):
        run_holdout(
            ledger=ledger,
            run_id="run",
            settings=settings,
            candidates=[candidate],
            i_understand_this_is_final=True,
            confirm=yes,
            data_dir=data_dir,
        )


def test_a_failed_holdout_still_counts_as_used(world):
    settings, ledger, data_dir, _ = world
    candidate = add_attempt(ledger, "a2", RAISES)
    results = run_holdout(
        ledger=ledger,
        run_id="run",
        settings=settings,
        candidates=[candidate],
        i_understand_this_is_final=True,
        confirm=yes,
        data_dir=data_dir,
    )
    assert results[0].status == "error"
    assert ledger.get_holdout("a2")["status"] == "error"
    with pytest.raises(HoldoutRefused, match="already been run"):
        run_holdout(
            ledger=ledger,
            run_id="run",
            settings=settings,
            candidates=[candidate],
            i_understand_this_is_final=True,
            confirm=yes,
            data_dir=data_dir,
        )


def test_cli_refuses_without_the_flag():
    result = CliRunner().invoke(app, ["holdout", "--run", "anything"])
    assert result.exit_code == 2
    assert "i-understand-this-is-final" in result.output
