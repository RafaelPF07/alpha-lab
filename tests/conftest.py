import socket

import numpy as np
import pandas as pd
import pytest


@pytest.fixture(autouse=True)
def block_network(monkeypatch):
    """Tests must never open a network connection (no yfinance, no Anthropic API)."""

    def refuse(*args, **kwargs):
        raise RuntimeError("tests must not open network connections")

    monkeypatch.setattr(socket.socket, "connect", refuse)
    monkeypatch.setattr(socket, "create_connection", refuse)


@pytest.fixture
def random_prices() -> pd.DataFrame:
    """Ten tickers, 400 business days, positive random-walk prices. Seeded."""
    rng = np.random.default_rng(7)
    index = pd.bdate_range("2019-01-01", periods=400)
    log_returns = rng.normal(0.0003, 0.012, size=(len(index), 10))
    prices = 100.0 * np.exp(np.cumsum(log_returns, axis=0))
    return pd.DataFrame(prices, index=index, columns=[f"T{i}" for i in range(10)])
