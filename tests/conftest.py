import socket

import numpy as np
import pandas as pd
import pytest

LOOPBACK = {"127.0.0.1", "::1", "localhost"}


@pytest.fixture(autouse=True)
def block_network(monkeypatch):
    """Tests must never reach the internet (no yfinance, no Anthropic API).

    Loopback stays allowed, because asyncio on Windows builds its self-pipe on
    127.0.0.1. Any other destination raises.
    """
    real_connect = socket.socket.connect
    real_create_connection = socket.create_connection

    def refuse(host):
        raise RuntimeError(f"tests must not open network connections (tried {host!r})")

    def guarded_connect(self, address):
        host = address[0] if isinstance(address, tuple) else address
        if host not in LOOPBACK:
            refuse(host)
        return real_connect(self, address)

    def guarded_create_connection(address, *args, **kwargs):
        host = address[0]
        if host not in LOOPBACK:
            refuse(host)
        return real_create_connection(address, *args, **kwargs)

    monkeypatch.setattr(socket.socket, "connect", guarded_connect)
    monkeypatch.setattr(socket, "create_connection", guarded_create_connection)


@pytest.fixture
def random_prices() -> pd.DataFrame:
    """Ten tickers, 400 business days, positive random-walk prices. Seeded."""
    rng = np.random.default_rng(7)
    index = pd.bdate_range("2019-01-01", periods=400)
    log_returns = rng.normal(0.0003, 0.012, size=(len(index), 10))
    prices = 100.0 * np.exp(np.cumsum(log_returns, axis=0))
    return pd.DataFrame(prices, index=index, columns=[f"T{i}" for i in range(10)])
