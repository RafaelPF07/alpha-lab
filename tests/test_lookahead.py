import pandas as pd
import pytest

from alpha_lab.checks.lookahead import perturbation_test, run_lookahead_tests, truncation_test
from alpha_lab.signals.baselines import SIGNALS, cheater


@pytest.mark.parametrize("name", ["momentum_12_1", "reversal_5d", "equal_weight_long"])
def test_baselines_pass_both_lookahead_tests(name, random_prices):
    results = run_lookahead_tests(SIGNALS[name], random_prices)
    assert all(r.passed for r in results), results


def test_cheater_fails_truncation(random_prices):
    result = truncation_test(cheater, random_prices)
    assert not result.passed
    assert result.first_offending_date is not None
    assert result.ticker in random_prices.columns


def test_cheater_fails_perturbation(random_prices):
    result = perturbation_test(cheater, random_prices)
    assert not result.passed
    assert result.first_offending_date is not None
    assert result.ticker in random_prices.columns


def test_signal_with_wrong_shape_is_rejected(random_prices):
    def bad(prices: pd.DataFrame) -> pd.DataFrame:
        return prices.iloc[:-1]

    with pytest.raises(ValueError, match="same index and columns"):
        truncation_test(bad, random_prices)
