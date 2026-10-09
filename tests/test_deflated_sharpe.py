import math

import numpy as np
import pandas as pd
import pytest

from alpha_lab.backtest.metrics import (
    EULER_GAMMA,
    deflated_sharpe_ratio,
    expected_max_sharpe,
    moments,
    per_period_sharpe,
)


def _norm_cdf(x: float) -> float:
    """Standard normal CDF from erf. Independent of statistics.NormalDist."""
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def _norm_ppf(p: float) -> float:
    """Inverse standard normal CDF by bisection on _norm_cdf. Independent of the library."""
    lo, hi = -40.0, 40.0
    for _ in range(200):
        mid = (lo + hi) / 2
        if _norm_cdf(mid) < p:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


def test_expected_max_matches_independent_calculation():
    n, v = 50, 0.02
    e = math.e
    independent = math.sqrt(v) * (
        (1 - EULER_GAMMA) * _norm_ppf(1 - 1 / n) + EULER_GAMMA * _norm_ppf(1 - 1 / (n * e))
    )
    assert expected_max_sharpe(n, v) == pytest.approx(independent, rel=1e-9)


def test_expected_max_is_close_to_a_known_order_statistic():
    # The expected maximum of 100 standard normal draws is about 2.508 (order-statistic tables).
    # The Gumbel-based formula is an approximation, so allow a tolerance of 0.05.
    assert expected_max_sharpe(100, 1.0) == pytest.approx(2.508, abs=0.05)


def test_no_trials_means_no_deflation():
    assert expected_max_sharpe(1, 1.0) == 0.0


def test_a_sharpe_equal_to_sr0_gives_probability_one_half():
    n, v = 200, 0.001
    sr0 = expected_max_sharpe(n, v)
    assert deflated_sharpe_ratio(sr0, 1000, 0.0, 3.0, n, v) == pytest.approx(0.5, abs=1e-12)


def test_deflation_reduces_significance_compared_with_a_single_trial():
    raw = deflated_sharpe_ratio(0.1, 1000, 0.0, 3.0, n_trials=1, sharpe_variance=0.001)
    deflated = deflated_sharpe_ratio(0.1, 1000, 0.0, 3.0, n_trials=500, sharpe_variance=0.001)
    assert raw > 0.99
    assert deflated < raw


def test_monte_carlo_noise_best_of_500_is_not_significant():
    """Reviewer requirement: 500 pure-noise series. The best one looks good raw, not after deflation."""
    rng = np.random.default_rng(20260101)
    n_trials, n_days = 500, 1000
    noise = rng.normal(loc=0.0, scale=0.01, size=(n_trials, n_days))  # zero true edge
    frame = pd.DataFrame(noise)

    trial_sharpes = [per_period_sharpe(frame.iloc[i]) for i in range(n_trials)]
    best = int(np.argmax(trial_sharpes))
    best_returns = frame.iloc[best]
    best_sr = trial_sharpes[best]
    variance = float(np.var(trial_sharpes, ddof=1))
    skew, kurt = moments(best_returns)

    # The raw Sharpe ratio of the best noise series is high (it is the maximum of 500 draws).
    assert best_sr * math.sqrt(252) > 1.0  # about 1.5 annualised: looks good for pure noise

    naive = deflated_sharpe_ratio(best_sr, n_days, skew, kurt, n_trials=1, sharpe_variance=0.0)
    honest = deflated_sharpe_ratio(
        best_sr, n_days, skew, kurt, n_trials=n_trials, sharpe_variance=variance
    )
    # Naively, the best noise series looks significant. Corrected for 500 trials, it must not.
    assert naive > 0.95
    assert honest < 0.95, (
        f"implementation is wrong if noise is called significant (DSR={honest:.3f})"
    )
    assert honest > 0.05, f"implementation is wrong if noise is never plausible (DSR={honest:.3f})"


def test_skew_and_kurtosis_enter_the_variance_term():
    # Variance term = 1 - g3 * SR + (g4 - 1) / 4 * SR^2 = 1 + 0.15 + 0.0125 = 1.1625.
    # DSR = Phi(0.1 * sqrt(799) / sqrt(1.1625)) = 0.995625. Ignoring the moments gives 0.997648.
    assert deflated_sharpe_ratio(0.1, 800, -1.5, 6.0, 1, 0.0) == pytest.approx(0.995625, abs=1e-6)


def test_a_non_positive_variance_term_gives_nan():
    # Variance term = 1 - 2 * 1.0 + (3 - 1) / 4 * 1.0^2 = -0.5, so the normal approximation is void.
    assert math.isnan(deflated_sharpe_ratio(1.0, 100, 2.0, 3.0, 1, 0.0))


def test_dsr_uses_n_minus_one_observations():
    # Sharpe 0.1 over 101 returns, so T - 1 = 100. One trial, so no deflation.
    # Variance term = 1 + (3 - 1) / 4 * 0.1^2 = 1.005, so z = 0.1 * sqrt(100) / sqrt(1.005).
    z = 0.1 * math.sqrt(100) / math.sqrt(1.005)
    assert deflated_sharpe_ratio(0.1, 101, 0.0, 3.0, 1, 0.0) == pytest.approx(
        _norm_cdf(z), abs=1e-12
    )


def test_moments_of_a_normal_sample_are_close_to_normal_values():
    rng = np.random.default_rng(1)
    skew, kurt = moments(pd.Series(rng.normal(size=200_000)))
    assert skew == pytest.approx(0.0, abs=0.02)
    assert kurt == pytest.approx(3.0, abs=0.05)  # non-excess kurtosis


def test_too_few_observations_is_rejected():
    with pytest.raises(ValueError):
        deflated_sharpe_ratio(0.1, 1, 0.0, 3.0, 10, 0.1)
