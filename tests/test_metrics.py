import math

import pandas as pd
import pytest

from alpha_lab.backtest.metrics import compute_metrics, deflated_sharpe_ratio


def test_sharpe_matches_hand_calculation():
    # r = [0.02, 0.0]. Mean 0.01. Sample std (ddof=1) = sqrt((0.01^2 + 0.01^2) / 1) = sqrt(2e-4).
    # Sharpe = 0.01 / sqrt(2e-4) * sqrt(252) = 11.22497...
    m = compute_metrics(pd.Series([0.02, 0.0]), pd.Series([0.0, 0.0]), 252)
    assert m.sharpe == pytest.approx(0.01 / math.sqrt(2e-4) * math.sqrt(252), rel=1e-9)


def test_max_drawdown_hand_example():
    # Equity goes 1.00 -> 1.02 -> 0.969. The drawdown from the 1.02 peak is 0.969 / 1.02 - 1 = -5%.
    m = compute_metrics(pd.Series([0.02, -0.05]), pd.Series([0.0, 0.0]), 252)
    assert m.max_drawdown == pytest.approx(-0.05)


def test_hit_rate_and_turnover():
    m = compute_metrics(pd.Series([0.02, 0.0, 0.01, -0.01]), pd.Series([0.1, 0.3, 0.2, 0.4]), 252)
    assert m.hit_rate == pytest.approx(0.5)
    assert m.avg_daily_turnover == pytest.approx(0.25)
    assert m.n_days == 4


def test_annualised_return_is_compounded():
    # 252 days of +0.1% compound to 1.001^252 in one year.
    m = compute_metrics(pd.Series([0.001] * 252), pd.Series([0.0] * 252), 252)
    assert m.ann_return == pytest.approx(1.001**252 - 1.0)


def test_deflated_sharpe_is_a_stub_until_phase_3():
    with pytest.raises(NotImplementedError):
        deflated_sharpe_ratio()
