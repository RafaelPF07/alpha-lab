import numpy as np
import pandas as pd

from alpha_lab.backtest.engine import run_backtest
from alpha_lab.backtest.holding import hold_weights
from alpha_lab.config import BacktestConfig


def make(n=60, k=4, seed=2):
    rng = np.random.default_rng(seed)
    index = pd.bdate_range("2020-01-01", periods=n)
    cols = [f"T{i}" for i in range(k)]
    weights = pd.DataFrame(rng.normal(size=(n, k)), index=index, columns=cols)
    return weights.sub(weights.mean(axis=1), axis=0)


def test_holding_period_of_one_changes_nothing():
    w = make()
    pd.testing.assert_frame_equal(hold_weights(w, 1), w)


def test_held_weight_comes_only_from_the_latest_rebalance_day():
    w = make()
    held = hold_weights(w, 5)
    for t in range(len(w)):
        last_rebalance = (t // 5) * 5
        pd.testing.assert_series_equal(held.iloc[t], w.iloc[last_rebalance], check_names=False)


def test_held_weights_never_use_a_later_day():
    # Changing prices or weights after day t must not change the held weight on day t.
    w = make()
    before = hold_weights(w, 5)
    changed = w.copy()
    changed.iloc[30:] = 99.0
    after = hold_weights(changed, 5)
    pd.testing.assert_frame_equal(before.iloc[:30], after.iloc[:30])


def long_a_with_nan_on_day_five() -> pd.DataFrame:
    # Long A on every day, except NaN on day 5, which is a rebalance day when holding_days=5.
    index = pd.bdate_range("2020-01-06", periods=12)
    w = pd.DataFrame(0.0, index=index, columns=["A", "B"])
    w["A"] = 1.0
    w.iloc[5, 0] = np.nan
    return w


def test_nan_on_a_rebalance_day_is_flat_for_the_whole_holding_period():
    # The engine treats NaN as flat. A NaN on day 5 must not carry day 0's weight forward.
    held = hold_weights(long_a_with_nan_on_day_five(), 5)
    assert held["A"].iloc[:5].tolist() == [1.0] * 5
    assert held["A"].iloc[5:10].tolist() == [0.0] * 5


def test_nan_rebalance_weight_does_not_earn_a_later_price_jump():
    # A jumps 10% on day 6. The NaN on day 5 is flat, so the jump must not be earned.
    index = pd.bdate_range("2020-01-06", periods=12)
    prices = pd.DataFrame(100.0, index=index, columns=["A", "B"])
    prices.iloc[6:, 0] = 110.0
    cfg = BacktestConfig(
        cost_bps_per_side=0.0, max_gross_leverage=1.0, annualization_days=252, min_history_days=0
    )

    held = hold_weights(long_a_with_nan_on_day_five(), 5)
    net = run_backtest(held, prices, index[1].date(), index[-1].date(), cfg).net_returns

    assert abs((1.0 + net).prod() - 1.0) < 1e-12


def test_nan_and_infinite_weights_are_flat_at_every_holding_period():
    w = make()
    w.iloc[7, 1] = np.nan
    w.iloc[10, 2] = np.inf
    for holding_days in (1, 5):
        assert np.isfinite(hold_weights(w, holding_days).to_numpy()).all()
    assert hold_weights(w, 1).iloc[7, 1] == 0.0
    assert hold_weights(w, 1).iloc[10, 2] == 0.0


def test_all_nan_signal_holds_zero_at_every_holding_period():
    w = make() * np.nan
    zero = pd.DataFrame(0.0, index=w.index, columns=w.columns)
    pd.testing.assert_frame_equal(hold_weights(w, 1), zero)
    pd.testing.assert_frame_equal(hold_weights(w, 5), zero)


def test_longer_holding_period_cuts_turnover():
    rng = np.random.default_rng(4)
    index = pd.bdate_range("2019-01-01", periods=400)
    prices = pd.DataFrame(
        100 * np.exp(np.cumsum(rng.normal(0, 0.01, (400, 5)), axis=0)),
        index=index,
        columns=[f"T{i}" for i in range(5)],
    )
    raw = make(n=400, k=5)
    raw.index = index
    raw.columns = prices.columns
    cfg = BacktestConfig(
        cost_bps_per_side=5.0, max_gross_leverage=1.0, annualization_days=252, min_history_days=0
    )
    daily = run_backtest(raw, prices, index[1].date(), index[-1].date(), cfg).metrics
    held = run_backtest(
        hold_weights(raw, 21), prices, index[1].date(), index[-1].date(), cfg
    ).metrics
    assert held.avg_daily_turnover < daily.avg_daily_turnover / 5
