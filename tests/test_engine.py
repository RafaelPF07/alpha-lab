"""Hand-computed backtest cases. Expected values are worked out in the comments, not taken from the code."""

from datetime import date

import pandas as pd
import pytest

from alpha_lab.backtest.engine import run_backtest, scale_to_gross
from alpha_lab.config import BacktestConfig, Dates
from alpha_lab.data.splits import split_window

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
    prices = frame([[100, 100, 100]] * 10)
    prices.loc[DAYS[5], "B"] = float("nan")
    weights = frame([[0.0, 0.0, 0.0]] * 10)

    with pytest.raises(ValueError, match="NaN"):
        run_backtest(weights, prices, DAYS[1].date(), DAYS[-1].date(), cfg())


def test_weights_with_mismatched_shape_raise():
    prices = frame([[100, 100, 100]] * 10)
    weights = frame([[0.0, 0.0, 0.0]] * 10).drop(columns="C")

    with pytest.raises(ValueError, match="same index and columns"):
        run_backtest(weights, prices, date(2020, 1, 7), date(2020, 1, 17), cfg())
