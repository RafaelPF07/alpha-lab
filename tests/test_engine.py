"""Hand-computed backtest cases. Expected values are worked out in the comments, not taken from the code."""

import inspect
import math
import warnings
from datetime import date

import numpy as np
import pandas as pd
import pytest

from alpha_lab.backtest.engine import BacktestDataError, run_backtest, scale_to_gross
from alpha_lab.backtest.holding import hold_weights
from alpha_lab.checks.lookahead import perturbation_test, run_lookahead_tests, truncation_test
from alpha_lab.config import BacktestConfig, Dates
from alpha_lab.data.splits import split_window
from alpha_lab.signals.baselines import equal_weight_long, momentum_12_1, reversal_5d

DAYS = pd.bdate_range("2020-01-06", periods=10)
COLS = ["A", "B", "C"]


def cfg(**overrides) -> BacktestConfig:
    base = {
        "cost_bps_per_side": 5.0,
        "max_gross_leverage": 1.0,
        "annualization_days": 252,
        "min_history_days": 0,
    }
    base.update(overrides)
    return BacktestConfig(**base)


def frame(rows) -> pd.DataFrame:
    return pd.DataFrame(rows, index=DAYS, columns=COLS, dtype=float)


def test_hand_worked_pnl_one_position():
    # A jumps 100 -> 110 on day 1 and stays there. Hold 0.5 of A from day 0.
    # Day 1 earns 0.5 * 10% = 0.05 gross. Turnover is |0.5 - 0| = 0.5, so cost is
    # 0.5 * 5 bps = 0.00025. Net = 0.05 - 0.00025 = 0.04975. Later days are flat.
    prices = frame([[100, 100, 100]] + [[110, 100, 100]] * 9)
    weights = frame([[0.5, 0.0, 0.0]] * 10)

    result = run_backtest(weights, prices, DAYS[1].date(), DAYS[-1].date(), cfg())

    assert result.gross_returns.iloc[0] == pytest.approx(0.05)
    assert result.turnover.iloc[0] == pytest.approx(0.5)
    assert result.net_returns.iloc[0] == pytest.approx(0.04975)
    assert result.net_returns.iloc[1:].abs().max() == pytest.approx(0.0)


def test_costs_on_known_weight_change():
    # Flat prices, so only costs move P&L. Day 0 holds all of A; from day 1 all of B.
    # Positions: day 1 holds A (turnover |1 - 0| = 1), day 2 holds B (turnover |0-1| + |1-0| = 2).
    # Costs: day 1 = 1 * 5 bps = -0.0005, day 2 = 2 * 5 bps = -0.001.
    prices = frame([[100, 100, 100]] * 10)
    weights = frame([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]] + [[0.0, 1.0, 0.0]] * 8)

    result = run_backtest(weights, prices, DAYS[1].date(), DAYS[-1].date(), cfg())

    assert result.turnover.iloc[0] == pytest.approx(1.0)
    assert result.turnover.iloc[1] == pytest.approx(2.0)
    assert result.net_returns.iloc[0] == pytest.approx(-0.0005)
    assert result.net_returns.iloc[1] == pytest.approx(-0.001)
    assert result.net_returns.iloc[2:].abs().max() == pytest.approx(0.0)


def test_leverage_scales_gross_exposure_down():
    # Weights of 1 in each name give gross 3. Engine scales them to 1/3 each.
    # Day 1: A up 10%, so gross = (1/3) * 0.10 = 0.1/3. Costs are set to zero for this test.
    prices = frame([[100, 100, 100]] + [[110, 100, 100]] * 9)
    weights = frame([[1.0, 1.0, 1.0]] * 10)

    result = run_backtest(
        weights, prices, DAYS[1].date(), DAYS[-1].date(), cfg(cost_bps_per_side=0.0)
    )

    assert result.gross_returns.iloc[0] == pytest.approx(0.1 / 3)


def test_scale_to_gross_handles_nan_inf_and_shrinks_only_when_needed():
    weights = pd.DataFrame(
        [[2.0, -2.0, 0.0], [0.2, 0.3, float("nan")], [float("inf"), 0.0, 0.0]],
        index=DAYS[:3],
        columns=COLS,
    )
    scaled = scale_to_gross(weights, max_gross=1.0)

    assert scaled.iloc[0].tolist() == pytest.approx([0.5, -0.5, 0.0])  # gross 4 -> 1
    assert scaled.iloc[1].tolist() == pytest.approx([0.2, 0.3, 0.0])  # gross 0.5, unchanged
    assert scaled.iloc[2].tolist() == pytest.approx([0.0, 0.0, 0.0])  # inf became 0


def test_weight_set_on_last_train_day_earns_first_validation_day():
    # A is flat until day 5, then rises 10%. The weight is set only on day 4,
    # the last train day. The day-5 return is realised in validation, so it must
    # count there and not in train.
    dates = Dates(
        start=DAYS[0].date(),
        train_end=DAYS[4].date(),
        validation_end=DAYS[7].date(),
        holdout_end=DAYS[9].date(),
    )
    prices = frame([[100, 100, 100]] * 5 + [[110, 100, 100]] * 5)
    weights = frame([[0.0, 0.0, 0.0]] * 10)
    weights.loc[DAYS[4], "A"] = 1.0

    train_lo, train_hi = split_window(dates, "train")
    val_lo, val_hi = split_window(dates, "validation")
    train = run_backtest(weights, prices, train_lo, train_hi, cfg(cost_bps_per_side=0.0))
    validation = run_backtest(weights, prices, val_lo, val_hi, cfg(cost_bps_per_side=0.0))

    assert train.gross_returns.abs().sum() == pytest.approx(0.0)
    assert validation.gross_returns.iloc[0] == pytest.approx(0.10)
    assert validation.gross_returns.index[0] == DAYS[5]


def test_min_history_days_are_dropped_from_output():
    prices = frame([[100, 100, 100]] * 10)
    weights = frame([[0.0, 0.0, 0.0]] * 10)

    result = run_backtest(weights, prices, DAYS[0].date(), DAYS[-1].date(), cfg(min_history_days=3))

    assert result.net_returns.index[0] == DAYS[3]


def test_nan_price_inside_window_raises():
    # B is held, so a NaN on its price cannot be scored as no return.
    prices = frame([[100, 100, 100]] * 10)
    prices.loc[DAYS[5], "B"] = float("nan")
    weights = frame([[0.0, 1.0, 0.0]] * 10)

    with pytest.raises(ValueError, match="NaN"):
        run_backtest(weights, prices, DAYS[1].date(), DAYS[-1].date(), cfg())


def test_weights_with_mismatched_shape_raise():
    prices = frame([[100, 100, 100]] * 10)
    weights = frame([[0.0, 0.0, 0.0]] * 10).drop(columns="C")

    with pytest.raises(ValueError, match="same index and columns"):
        run_backtest(weights, prices, date(2020, 1, 7), date(2020, 1, 17), cfg())


def test_backtest_data_error_is_a_value_error():
    assert issubclass(BacktestDataError, ValueError)


def test_nan_on_the_day_before_the_window_raises():
    # The first in-window return needs yesterday's price. A NaN there used to be
    # scored silently as a zero return.
    prices = frame([[100, 100, 100]] * 10)
    prices.loc[DAYS[0], "B"] = float("nan")
    weights = frame([[0.0, 1.0, 0.0]] * 10)  # B is held into the window

    with pytest.raises(BacktestDataError, match="NaN"):
        run_backtest(weights, prices, DAYS[1].date(), DAYS[-1].date(), cfg())


def test_window_that_runs_past_the_last_price_raises():
    prices = frame([[100, 100, 100]] * 10)
    weights = frame([[0.0, 0.0, 0.0]] * 10)
    too_late = (DAYS[-1] + pd.Timedelta(days=30)).date()

    with pytest.raises(BacktestDataError, match="end"):
        run_backtest(weights, prices, DAYS[1].date(), too_late, cfg())


def test_window_that_starts_before_the_first_price_raises():
    prices = frame([[100, 100, 100]] * 10)
    weights = frame([[0.0, 0.0, 0.0]] * 10)
    too_early = (DAYS[0] - pd.Timedelta(days=30)).date()

    with pytest.raises(BacktestDataError, match="start"):
        run_backtest(weights, prices, too_early, DAYS[-1].date(), cfg())


def test_six_days_of_slack_at_both_window_edges_is_allowed():
    # Weekends and market holidays: a window may end up to a week past the first or last price.
    prices = frame([[100, 100, 100]] * 10)
    weights = frame([[0.0, 0.0, 0.0]] * 10)
    slack_end = (DAYS[-1] + pd.Timedelta(days=6)).date()
    slack_start = (DAYS[0] - pd.Timedelta(days=6)).date()

    assert len(run_backtest(weights, prices, DAYS[1].date(), slack_end, cfg()).net_returns) == 9
    assert len(run_backtest(weights, prices, slack_start, DAYS[-1].date(), cfg()).net_returns) == 9


@pytest.mark.parametrize("overflow_days", [8, 14])
def test_overflow_past_the_window_tolerance_raises_at_either_edge(overflow_days):
    # R3-QA-05(d). The tolerance is 7 days, so 8 days past either edge must raise. (A tolerance of
    # 14 days would let a 14-day overflow through, which is what the tests below catch.)
    prices = frame([[100, 100, 100]] * 10)
    weights = frame([[0.0, 0.0, 0.0]] * 10)
    too_late = (DAYS[-1] + pd.Timedelta(days=overflow_days)).date()
    too_early = (DAYS[0] - pd.Timedelta(days=overflow_days)).date()

    with pytest.raises(BacktestDataError, match="end"):
        run_backtest(weights, prices, DAYS[1].date(), too_late, cfg())
    with pytest.raises(BacktestDataError, match="start"):
        run_backtest(weights, prices, too_early, DAYS[-1].date(), cfg())


def test_window_tolerance_is_seven_calendar_days():
    from alpha_lab.backtest.engine import WINDOW_TOLERANCE_DAYS

    assert WINDOW_TOLERANCE_DAYS == 7


def _weekday_index_with_block_removed(n: int, start: int, count: int) -> pd.DatetimeIndex:
    full = pd.bdate_range("2020-01-06", periods=n)
    keep = np.ones(n, dtype=bool)
    keep[start : start + count] = False
    return full[keep]


@pytest.mark.parametrize("count", [5, 6])
def test_a_gap_of_more_than_four_weekdays_raises(count):
    # R3-STAT-05, R4-STAT-03. A is about to jump 50% after the gap, and a return across the gap would
    # book the whole move as one day, so the engine must refuse the prices. Five weekdays is the first
    # size over the limit, so it is tested as well as six (the boundary is pinned).
    index = _weekday_index_with_block_removed(30, start=10, count=count)
    prices = pd.DataFrame(100.0, index=index, columns=COLS)
    prices.iloc[10:, 0] = 150.0
    weights = pd.DataFrame(0.0, index=index, columns=COLS)
    weights.iloc[:, 0] = 1.0

    with pytest.raises(BacktestDataError, match="gap"):
        run_backtest(weights, prices, index[0].date(), index[-1].date(), cfg())


def test_a_gap_before_the_window_does_not_refuse_the_window():
    # R4-STAT-06. A six-weekday gap ends before the window starts. No return in the window touches it,
    # so the window is scored. Before the fix the whole cache was checked and the window was refused.
    index = _weekday_index_with_block_removed(30, start=2, count=6)
    prices = pd.DataFrame(100.0, index=index, columns=COLS)
    weights = pd.DataFrame(0.0, index=index, columns=COLS)
    window_start = index[9]

    result = run_backtest(weights, prices, window_start.date(), index[-1].date(), cfg())

    assert len(result.net_returns) == int((index >= window_start).sum())


@pytest.mark.parametrize("count", [1, 4])
def test_a_gap_of_up_to_four_weekdays_is_accepted(count):
    # A holiday looks the same as a missing day without a trading calendar, so short gaps pass.
    index = _weekday_index_with_block_removed(30, start=10, count=count)
    prices = pd.DataFrame(100.0, index=index, columns=COLS)
    weights = pd.DataFrame(0.0, index=index, columns=COLS)

    result = run_backtest(weights, prices, index[0].date(), index[-1].date(), cfg())

    assert len(result.net_returns) == len(index) - 1


def test_window_with_fewer_than_two_scored_days_raises():
    prices = frame([[100, 100, 100]] * 10)
    weights = frame([[0.0, 0.0, 0.0]] * 10)

    with pytest.raises(BacktestDataError):
        run_backtest(weights, prices, DAYS[5].date(), DAYS[5].date(), cfg())


@pytest.mark.parametrize("signal", [momentum_12_1, reversal_5d, equal_weight_long])
def test_net_returns_before_a_cutoff_do_not_depend_on_later_prices(signal):
    # End to end: signal, then holding, then engine. Scrambling prices after the cutoff
    # must not move any net return up to and including the cutoff. STAT-14: three signals, not one.
    rng = np.random.default_rng(11)
    index = pd.bdate_range("2015-01-01", periods=420)
    cols = [f"T{i}" for i in range(6)]
    prices = pd.DataFrame(
        100 * np.exp(np.cumsum(rng.normal(0, 0.01, (420, 6)), axis=0)), index=index, columns=cols
    )
    cutoff = 330
    perturbed = prices.copy()
    after = perturbed.index > index[cutoff]
    perturbed.loc[after] = perturbed.loc[after].to_numpy() * (
        1 + rng.normal(0, 0.05, (int(after.sum()), 6))
    )

    def net(px: pd.DataFrame) -> pd.Series:
        held = hold_weights(signal(px), 21)
        return run_backtest(held, px, index[1].date(), index[-1].date(), cfg()).net_returns

    base, changed = net(prices), net(perturbed)
    up_to_cutoff = index[1 : cutoff + 1]
    later = index[cutoff + 1 :]
    pd.testing.assert_series_equal(base.loc[up_to_cutoff], changed.loc[up_to_cutoff])
    assert not np.allclose(base.loc[later], changed.loc[later])  # the check is not vacuous


def late_leak_from(leak_from: pd.Timestamp):
    """Equal weight, except on and after `leak_from`, where it trades on tomorrow's return.

    The leak is keyed on a fixed date, not on the length of the input, as a real leak would be.
    """

    def signal(prices: pd.DataFrame) -> pd.DataFrame:
        weights = pd.DataFrame(1.0 / prices.shape[1], index=prices.index, columns=prices.columns)
        tomorrow = prices.pct_change(fill_method=None).shift(-1)
        late = prices.index >= leak_from
        weights.loc[late] = np.sign(tomorrow.loc[late]).fillna(0.0).to_numpy()
        return weights

    return signal


def _random_prices(seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    index = pd.bdate_range("2019-01-01", periods=400)
    return pd.DataFrame(
        100 * np.exp(np.cumsum(rng.normal(0.0003, 0.012, (400, 10)), axis=0)),
        index=index,
        columns=[f"T{i}" for i in range(10)],
    )


def test_lookahead_tests_see_a_leak_in_the_last_tenth_of_the_rows():
    # Row 368 of 400 is 92%. The old cutoffs stopped at 90% (row 360), so nothing compared the leak.
    prices = _random_prices(3)

    results = run_lookahead_tests(late_leak_from(prices.index[368]), prices)

    assert not all(r.passed for r in results), results


@pytest.mark.parametrize("seed", [3, 11, 19, 27, 35])
def test_truncation_catches_a_late_leak_for_every_price_seed(seed):
    # STAT-14: one seed was not enough. The cutoff at row 380 (95%) sees the leak on row 380 for any
    # prices, so the truncation test must fail for each seed, not only for the one above.
    prices = _random_prices(seed)

    assert not truncation_test(late_leak_from(prices.index[368]), prices).passed


def test_default_cutoff_count_is_five():
    # R3-QA-05(c): the number of cutoffs is part of the test's power, so pin the default.
    for test in (truncation_test, perturbation_test):
        assert inspect.signature(test).parameters["n_cutoffs"].default == 5


@pytest.mark.parametrize("signal", [momentum_12_1, reversal_5d, equal_weight_long])
@pytest.mark.parametrize("seed", [3, 11, 19])
def test_baselines_pass_both_lookahead_tests_for_several_seeds(signal, seed):
    results = run_lookahead_tests(signal, _random_prices(seed))

    assert all(r.passed for r in results), results


def test_reversal_buys_losers_and_momentum_buys_winners():
    cols = ["T0", "T1", "T2", "T3", "T4"]
    # reversal_5d: T0 fell 20% over the last five days and T4 rose 20%. Reversal buys T0.
    days = pd.bdate_range("2020-01-01", periods=6)
    path = np.full((6, 5), 100.0)
    path[5, 0] = 80.0
    path[5, 4] = 120.0
    reversal = reversal_5d(pd.DataFrame(path, index=days, columns=cols)).iloc[-1]
    assert reversal["T0"] > 0 > reversal["T4"]

    # momentum_12_1: T0 rose 0.1% a day for a year and T4 fell by the same amount.
    # Momentum buys the winner T0 and sells the loser T4.
    days = pd.bdate_range("2019-01-01", periods=260)
    drift = np.array([0.001, 0.0, 0.0, 0.0, -0.001])
    path = 100.0 * np.exp(np.arange(260)[:, None] * drift[None, :])
    momentum = momentum_12_1(pd.DataFrame(path, index=days, columns=cols)).iloc[-1]
    assert momentum["T0"] > 0 > momentum["T4"]


def test_nan_on_a_zero_weight_column_scores_as_no_return():
    # Coordinator request (R4-F3). A ticker that is flat across a gap is not exposed to it, so a
    # validation-only or holdout-only gap must not fail the window. Its NaN returns earn nothing.
    weights = frame([[0.5, 0.0, 0.5]] * 10)
    gapped = frame([[100, 100, 100]] * 10)
    gapped.loc[DAYS[5:8], "B"] = float("nan")
    filled = gapped.copy()
    filled["B"] = 100.0

    with_gap = run_backtest(weights, gapped, DAYS[1].date(), DAYS[-1].date(), cfg())
    no_gap = run_backtest(weights, filled, DAYS[1].date(), DAYS[-1].date(), cfg())

    pd.testing.assert_series_equal(with_gap.net_returns, no_gap.net_returns)


def test_nan_on_a_column_that_is_exited_on_the_gap_still_raises():
    # R4-STAT-02. B is held through day 3 and its position is zero from day 4 on. Day 5 is the NaN
    # day: flat today, but held yesterday, so the sale on day 4 would trade across the NaN. The
    # engine must refuse it. The check that looks only at today would score this without error.
    prices = frame([[100, 100, 100]] * 10)
    prices.loc[DAYS[5], "B"] = float("nan")
    weights = frame([[0.0, 0.0, 0.0]] * 10)
    weights.loc[DAYS[:4], "B"] = 1.0  # positions are these shifted one day: held on days 1 to 4

    with pytest.raises(BacktestDataError, match="NaN"):
        run_backtest(weights, prices, DAYS[1].date(), DAYS[-1].date(), cfg())


def test_nan_on_a_column_that_is_held_on_the_gap_still_raises():
    # The held-today case, kept as the second half of R4-STAT-02: B is held on day 5 itself.
    prices = frame([[100, 100, 100]] * 10)
    prices.loc[DAYS[5], "B"] = float("nan")
    weights = frame([[0.0, 0.0, 0.0]] * 10)
    weights.loc[DAYS[:5], "B"] = 1.0  # positions are these shifted one day, so B is held on day 5

    with pytest.raises(BacktestDataError, match="NaN"):
        run_backtest(weights, prices, DAYS[1].date(), DAYS[-1].date(), cfg())


def test_inf_on_a_flat_column_scores_as_no_return():
    # R4-STAT-10. An infinite return on a column with no position earns nothing, as a NaN does. The
    # engine zeroes it explicitly, so no NaN reaches the sum, and no numpy warning is raised.
    # This passed before the fix as well: the defect was the comment and the reliance on skipna.
    weights = frame([[0.5, 0.0, 0.5]] * 10)
    infinite = frame([[100, 100, 100]] * 10)
    infinite.loc[DAYS[5], "B"] = float("inf")
    filled = infinite.copy()
    filled["B"] = 100.0

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        with_inf = run_backtest(weights, infinite, DAYS[1].date(), DAYS[-1].date(), cfg())
    no_inf = run_backtest(weights, filled, DAYS[1].date(), DAYS[-1].date(), cfg())

    pd.testing.assert_series_equal(with_inf.net_returns, no_inf.net_returns)
    assert np.isfinite(with_inf.gross_returns).all()


def _random_weights_and_prices(seed: int = 4):
    rng = np.random.default_rng(seed)
    index = pd.bdate_range("2018-01-01", periods=120)
    cols = [f"T{i}" for i in range(5)]
    prices = pd.DataFrame(
        100 * np.exp(np.cumsum(rng.normal(0.0002, 0.01, (120, 5)), axis=0)),
        index=index,
        columns=cols,
    )
    weights = pd.DataFrame(rng.normal(0.0, 0.3, (120, 5)), index=index, columns=cols)
    return weights, prices


def test_execution_lag_zero_is_the_default():
    weights, prices = _random_weights_and_prices()
    lo, hi = prices.index[1].date(), prices.index[-1].date()

    default = run_backtest(weights, prices, lo, hi, cfg())
    explicit = run_backtest(weights, prices, lo, hi, cfg(), execution_lag=0)

    pd.testing.assert_series_equal(default.net_returns, explicit.net_returns)
    assert default.metrics.sharpe == explicit.metrics.sharpe


def test_execution_lag_one_is_the_same_weights_one_bar_later():
    # The fill moves one bar, so lag 1 is the default run on weights shifted down one more row. The
    # returns, costs and turnover must then match exactly.
    weights, prices = _random_weights_and_prices()
    lo, hi = prices.index[1].date(), prices.index[-1].date()

    lagged = run_backtest(weights, prices, lo, hi, cfg(), execution_lag=1)
    shifted = run_backtest(weights.shift(1).fillna(0.0), prices, lo, hi, cfg())

    pd.testing.assert_series_equal(lagged.net_returns, shifted.net_returns)
    pd.testing.assert_series_equal(lagged.turnover, shifted.turnover)


def test_execution_lag_one_moves_a_one_day_move_to_the_next_bar():
    # A is 100 on day 0 and 110 from day 1 on. Buying A on day 0 earns the jump on day 1 at lag 0.
    # At lag 1 the same position is only taken on day 1, so the jump is not earned at all. Costs are
    # zero for this case.
    prices = frame([[100, 100, 100]] + [[110, 100, 100]] * 9)
    weights = frame([[0.0, 0.0, 0.0]] * 10)
    weights.loc[DAYS[0], "A"] = 1.0

    default = run_backtest(
        weights, prices, DAYS[1].date(), DAYS[-1].date(), cfg(cost_bps_per_side=0.0)
    )
    lagged = run_backtest(
        weights,
        prices,
        DAYS[1].date(),
        DAYS[-1].date(),
        cfg(cost_bps_per_side=0.0),
        execution_lag=1,
    )

    assert default.net_returns.iloc[0] == pytest.approx(0.1)
    assert lagged.net_returns.iloc[0] == pytest.approx(0.0)
    assert lagged.net_returns.abs().max() == pytest.approx(0.0)


def test_default_metrics_carry_the_lag_one_sharpe():
    # STAT-07. The default run also records the Sharpe the same weights would get one bar later.
    weights, prices = _random_weights_and_prices()
    lo, hi = prices.index[1].date(), prices.index[-1].date()

    default = run_backtest(weights, prices, lo, hi, cfg())
    lagged = run_backtest(weights, prices, lo, hi, cfg(), execution_lag=1)

    assert default.metrics.sharpe_lag1 == pytest.approx(lagged.metrics.sharpe)
    assert default.metrics.sharpe_lag1 != pytest.approx(default.metrics.sharpe)  # not vacuous


def test_lag_one_sharpe_is_nan_when_lag_one_execution_would_be_refused():
    # Lag 0 is fine: B is flat on day 5 and on day 4, so the NaN is no return. At lag 1 B is still
    # held on day 4 and flat on day 5, so the NaN would be crossed by a sale. The default result must
    # stand, the lag-one Sharpe is NaN, and an explicit lag-one run refuses the window.
    prices = frame([[100, 100, 100]] * 10)
    prices.loc[DAYS[5], "B"] = float("nan")
    weights = frame([[0.0, 0.0, 0.0]] * 10)
    weights.loc[DAYS[2], "B"] = 1.0

    default = run_backtest(weights, prices, DAYS[1].date(), DAYS[-1].date(), cfg())

    assert math.isnan(default.metrics.sharpe_lag1)
    with pytest.raises(BacktestDataError, match="NaN"):
        run_backtest(weights, prices, DAYS[1].date(), DAYS[-1].date(), cfg(), execution_lag=1)


@pytest.mark.parametrize("bad_lag", [-1, 1.5, True])
def test_execution_lag_must_be_a_whole_number_of_bars_zero_or_more(bad_lag):
    prices = frame([[100, 100, 100]] * 10)
    weights = frame([[0.0, 0.0, 0.0]] * 10)

    with pytest.raises(ValueError, match="execution_lag"):
        run_backtest(weights, prices, DAYS[1].date(), DAYS[-1].date(), cfg(), execution_lag=bad_lag)
