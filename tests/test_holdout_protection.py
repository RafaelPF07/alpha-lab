"""Holdout protection: needs the flag and a confirmation, runs once per idea, and claims before any read.

Every test uses a synthetic cache and a fake price source. No test touches the network or runs/.
"""

import sqlite3
from datetime import timedelta

import numpy as np
import pandas as pd
import pytest

import alpha_lab.pipeline.holdout as holdout_module
from alpha_lab.config import load_settings, load_universe
from alpha_lab.data.loader import CacheMismatch, write_cache, write_fingerprint
from alpha_lab.ledger.db import HoldoutConflict, Ledger
from alpha_lab.pipeline.holdout import (
    HoldoutCandidate,
    HoldoutRefused,
    holdout_sha256,
    run_holdout,
)

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
VARIANT = GOOD.replace("score", "mom").replace("shift(21)", "shift(20)")
RAISES = "def signal(prices):\n    raise ValueError('boom')\n"


class FakeSource:
    """Serves the synthetic full history for any request, and records every request."""

    def __init__(self, frame):
        self.frame = frame
        self.calls = []
        self.before_fetch = None

    def fetch_adjusted_close(self, tickers, start, end_exclusive):
        self.calls.append((list(tickers), start, end_exclusive))
        if self.before_fetch is not None:
            self.before_fetch()
        rows = (self.frame.index >= pd.Timestamp(start)) & (
            self.frame.index < pd.Timestamp(end_exclusive)
        )
        return self.frame.loc[rows].reindex(columns=list(tickers))


class World:
    pass


@pytest.fixture
def world(tmp_path):
    settings = load_settings()
    universe = load_universe()
    index = pd.bdate_range("2010-01-01", settings.dates.holdout_end)
    rng = np.random.default_rng(31)
    frame = pd.DataFrame(
        100.0 * np.exp(np.cumsum(rng.normal(0.0003, 0.012, size=(len(index), 6)), axis=0)),
        index=index,
        columns=[f"T{i}" for i in range(6)],
    )
    data_dir = tmp_path / "data"
    write_cache(frame, settings.dates, data_dir)
    write_fingerprint(settings, universe, data_dir)

    w = World()
    w.settings, w.universe, w.data_dir, w.frame = settings, universe, data_dir, frame
    w.holdout_days = int((index > pd.Timestamp(settings.dates.validation_end)).sum())
    w.ledger = Ledger(tmp_path / "ledger.db")
    w.ledger.create_run("run", "{}", 1.0)
    w.source = FakeSource(frame)
    return w


def add_attempt(w, attempt_id, code, idea_id=None):
    idea_id = idea_id or f"idea-{attempt_id}"
    w.ledger.add_attempt(
        attempt_id=attempt_id,
        idea_id=idea_id,
        run_id="run",
        attempt_number=1,
        code=code,
        code_hash=attempt_id,
        status="validation_pass",
    )
    return HoldoutCandidate(attempt_id=attempt_id, idea_id=idea_id, name=attempt_id, code=code)


def holdout(w, candidates, confirm=lambda _: True, flag=True, source=None):
    return run_holdout(
        ledger=w.ledger,
        run_id="run",
        settings=w.settings,
        universe=w.universe,
        candidates=candidates,
        i_understand_this_is_final=flag,
        confirm=confirm,
        data_dir=w.data_dir,
        source=source or w.source,
    )


def yes(_):
    return True


def no(_):
    return False


def test_refuses_without_the_flag_and_records_nothing(world):
    candidate = add_attempt(world, "a1", GOOD)
    with pytest.raises(HoldoutRefused, match="i-understand"):
        holdout(world, [candidate], flag=False)
    assert world.ledger.holdouts() == []
    assert world.source.calls == []


def test_declining_the_confirmation_records_nothing(world):
    candidate = add_attempt(world, "a1", GOOD)
    with pytest.raises(HoldoutRefused, match="not confirmed"):
        holdout(world, [candidate], confirm=no)
    assert world.ledger.holdouts() == []
    assert world.source.calls == []


def test_runs_once_on_holdout_days_only_and_records_the_data_hash(world):
    candidate = add_attempt(world, "a1", GOOD)
    results = holdout(world, [candidate])
    assert results[0].status == "complete"
    assert results[0].metrics["n_days"] == world.holdout_days  # only holdout returns were scored
    row = world.ledger.get_holdout("a1")
    assert row["status"] == "complete"
    expected = world.frame[
        (world.frame.index > pd.Timestamp(world.settings.dates.validation_end))
        & (world.frame.index <= pd.Timestamp(world.settings.dates.holdout_end))
    ]
    assert row["data_sha256"] == holdout_sha256(expected.astype("float64"))


def test_cannot_run_twice_on_the_same_signal(world):
    candidate = add_attempt(world, "a1", GOOD)
    holdout(world, [candidate])
    with pytest.raises(HoldoutRefused, match="already been run"):
        holdout(world, [candidate])


def test_a_failed_holdout_still_counts_as_used(world):
    candidate = add_attempt(world, "a2", RAISES)
    results = holdout(world, [candidate])
    assert results[0].status == "error"
    assert world.ledger.get_holdout("a2")["status"] == "error"
    with pytest.raises(HoldoutRefused, match="already been run"):
        holdout(world, [candidate])


def test_same_idea_with_different_code_gets_no_second_holdout(world):
    # DECISION-3: the one-run rule keys on idea_id, so a re-parameterised attempt is refused.
    first = add_attempt(world, "a1", GOOD, idea_id="idea-x")
    holdout(world, [first])
    second = add_attempt(world, "a2", VARIANT, idea_id="idea-x")
    with pytest.raises(HoldoutRefused, match="already been run"):
        holdout(world, [second])
    assert len(world.ledger.holdouts()) == 1
    assert world.ledger.get_holdout("a2") is None


def test_claim_is_written_before_the_holdout_is_requested(world):
    # SEC-09: the spy runs at the moment of the request. The rows must already exist then.
    candidate = add_attempt(world, "a1", GOOD)
    seen = {}

    def spy():
        seen["rows"] = [dict(r) for r in world.ledger.holdouts()]

    world.source.before_fetch = spy
    holdout(world, [candidate])
    assert [r["status"] for r in seen["rows"]] == ["running"]


def test_failure_after_the_claim_leaves_rows_as_error_and_blocks_a_second_run(world, monkeypatch):
    # SEC-09: inject a failure after the claim and before the read.
    candidate = add_attempt(world, "a1", GOOD)

    def fail_the_read(*args, **kwargs):
        raise OSError("injected failure before the holdout read")

    monkeypatch.setattr(holdout_module, "download_holdout", fail_the_read)
    results = holdout(world, [candidate])
    assert results[0].status == "error"
    rows = world.ledger.holdouts()
    assert [r["status"] for r in rows] == ["error"]
    monkeypatch.undo()
    with pytest.raises(HoldoutRefused, match="already been run"):
        holdout(world, [candidate])


def test_interrupt_after_the_claim_marks_rows_error_and_propagates(world, monkeypatch):
    candidate = add_attempt(world, "a1", GOOD)

    def interrupted(*args, **kwargs):
        raise KeyboardInterrupt

    monkeypatch.setattr(holdout_module, "download_holdout", interrupted)
    with pytest.raises(KeyboardInterrupt):
        holdout(world, [candidate])
    assert [r["status"] for r in world.ledger.holdouts()] == ["error"]


def test_claim_is_atomic_for_a_batch(world):
    # A batch that clashes part way through must leave no row behind.
    a = add_attempt(world, "a1", GOOD, idea_id="same")
    b = add_attempt(world, "a2", VARIANT, idea_id="same")
    rows = [
        {"holdout_id": "h1", "attempt_id": a.attempt_id, "idea_id": "same", "run_id": "run"},
        {"holdout_id": "h2", "attempt_id": b.attempt_id, "idea_id": "same", "run_id": "run"},
    ]
    with pytest.raises(sqlite3.IntegrityError):
        world.ledger.claim_holdouts(rows)
    assert world.ledger.holdouts() == []
    world.ledger.claim_holdouts(rows[:1])
    with pytest.raises(HoldoutConflict, match="already been run"):
        world.ledger.claim_holdouts(rows[:1])


def test_one_holdout_per_idea_is_enforced_by_a_unique_index(world):
    add_attempt(world, "a1", GOOD)
    world.ledger.ensure_holdout_key()
    indexes = world.ledger.conn.execute("PRAGMA index_list(holdout_runs)").fetchall()
    names = {row["name"]: row["unique"] for row in indexes}
    assert names.get("holdout_runs_one_per_idea") == 1


def test_already_run_candidates_are_skipped_and_the_rest_run(world):
    first = add_attempt(world, "a1", GOOD)
    holdout(world, [first])
    second = add_attempt(world, "b1", VARIANT)
    offered = {}

    def confirm(items):
        offered["names"] = [c.name for c in items]
        return True

    results = holdout(world, [first, second], confirm=confirm)
    assert offered["names"] == ["b1"]
    by_id = {r.attempt_id: r.status for r in results}
    assert by_id == {"a1": "skipped", "b1": "complete"}


def test_two_attempts_of_one_idea_in_one_batch_are_refused(world):
    a = add_attempt(world, "a1", GOOD, idea_id="same")
    b = add_attempt(world, "a2", VARIANT, idea_id="same")
    with pytest.raises(HoldoutRefused, match="same idea"):
        holdout(world, [a, b])
    assert world.ledger.holdouts() == []


def test_stale_cache_is_refused_before_any_claim_or_request(world):
    settings, universe = world.settings, world.universe
    other = universe.model_copy(update={"stocks": [*universe.stocks, "ZZZZ"]})
    write_fingerprint(settings, other, world.data_dir)  # the cache now claims other settings
    candidate = add_attempt(world, "a1", GOOD)
    with pytest.raises(CacheMismatch, match="lab data fetch --refresh"):
        holdout(world, [candidate])
    assert world.ledger.holdouts() == []
    assert world.source.calls == []


def test_the_holdout_request_uses_the_cache_tickers_and_a_context_window(world):
    candidate = add_attempt(world, "a1", GOOD)
    holdout(world, [candidate])
    assert len(world.source.calls) == 1
    tickers, start, end_exclusive = world.source.calls[0]
    assert tickers == [f"T{i}" for i in range(6)]
    assert start == world.settings.dates.validation_end - timedelta(
        days=holdout_module.CONTEXT_CALENDAR_DAYS
    )
    assert end_exclusive == world.settings.dates.holdout_end + timedelta(days=1)


def test_no_holdout_file_is_written_to_the_data_directory(world):
    candidate = add_attempt(world, "a1", GOOD)
    holdout(world, [candidate])
    names = sorted(p.name for p in world.data_dir.iterdir())
    # The pre-holdout cache is expected. Any other file that names the holdout window is not.
    assert not [n for n in names if "holdout" in n.lower() and "pre_holdout" not in n.lower()]


def test_the_claim_records_the_invocation(world):
    candidate = add_attempt(world, "a1", GOOD)
    run_holdout(
        ledger=world.ledger,
        run_id="run",
        settings=world.settings,
        universe=world.universe,
        candidates=[candidate],
        i_understand_this_is_final=True,
        confirm=yes,
        data_dir=world.data_dir,
        source=world.source,
        invocation={"argv": ["lab", "holdout"], "user": "tester"},
    )
    assert '"user": "tester"' in world.ledger.get_holdout("a1")["invocation_json"]


def test_holdout_hash_changes_when_any_value_changes():
    index = pd.bdate_range("2022-01-03", periods=5)
    frame = pd.DataFrame({"A": [1.0, 2.0, np.nan, 4.0, 5.0]}, index=index)
    changed = frame.copy()
    changed.iloc[2, 0] = 3.0
    assert holdout_sha256(frame) == holdout_sha256(frame.copy())
    assert holdout_sha256(frame) != holdout_sha256(changed)


def test_a_candidate_cannot_persist_the_holdout_window_and_keeps_its_claim(world, tmp_path):
    # R3-SEC-03, probe P7b: string dispatch to to_csv has no refused name in the code, so the static
    # filter passes it and the open() guard refuses it in the child. The idea's holdout is still
    # spent (DECISION-14), so it cannot be retried.
    target = tmp_path / "holdout_copy.csv"
    code = (
        "import pandas as pd\n\ndef signal(prices):\n"
        f"    prices.agg('to_csv', 0, {str(target)!r})\n"
        "    return prices * 0.0 + 1.0 / prices.shape[1]\n"
    )
    candidate = add_attempt(world, "h6", code)
    results = holdout(world, [candidate])
    assert not target.exists()
    assert results[0].status == "error"
    assert world.ledger.get_holdout("h6")["status"] == "error"
    with pytest.raises(HoldoutRefused, match="already been run"):
        holdout(world, [candidate])


def test_code_the_current_filter_refuses_is_refused_before_any_claim(world):
    # R5-F3, R4-SEC-02: a candidate validated under an older filter is checked again before any claim,
    # because attribute reads have no runtime layer. Nothing is claimed and nothing is requested.
    code = (
        "import pandas as pd\n\ndef signal(prices):\n"
        "    g = (x for x in [1])\n    _ = g.gi_frame\n"
        "    return prices * 0.0 + 1.0 / prices.shape[1]\n"
    )
    candidate = add_attempt(world, "fr1", code)
    source = FakeSource(world.frame)
    with pytest.raises(HoldoutRefused, match="static filter"):
        holdout(world, [candidate], source=source)
    assert world.ledger.holdouts() == []
    assert source.calls == []


def test_a_missing_holdout_row_fails_only_its_own_candidate(world, monkeypatch):
    # R4-F2 request (R3-AGT-15): finish_holdout raises when no row matches. That must not turn the
    # other candidates into errors. Here the row for "a1" is missing; "b1" must still complete.
    a = add_attempt(world, "a1", GOOD)
    b = add_attempt(world, "b1", VARIANT)
    real_finish = world.ledger.finish_holdout

    def missing_for_a1(attempt_id, status, metrics, data_sha256=None):
        if attempt_id == "a1":
            raise ValueError("no holdout row for attempt 'a1'; nothing was recorded")
        return real_finish(attempt_id, status, metrics, data_sha256)

    monkeypatch.setattr(world.ledger, "finish_holdout", missing_for_a1)
    results = holdout(world, [a, b])
    by_id = {r.attempt_id: r for r in results}
    assert by_id["a1"].status == "error"
    assert "could not be recorded" in by_id["a1"].error
    assert by_id["b1"].status == "complete"


def test_the_holdout_record_keeps_the_lag_one_sharpe_like_validation(world):
    # R4-STAT-09 decision (R5-F1 request): sharpe_lag1 is a diagnostic that the engine sets for every
    # backtest and validation already stores. The holdout keeps it too, so the two records match.
    candidate = add_attempt(world, "a1", GOOD)
    results = holdout(world, [candidate])
    assert "sharpe_lag1" in results[0].metrics
