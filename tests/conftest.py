import socket

import numpy as np
import pandas as pd
import pytest

LOOPBACK = {"127.0.0.1", "::1", "localhost"}


@pytest.fixture(autouse=True)
def block_network(monkeypatch):
    """Tests must never reach the internet (no yfinance, no Anthropic API).

    Loopback stays allowed, because asyncio on Windows builds its self-pipe on
    127.0.0.1. Any other destination raises. Every socket call that takes an
    address is covered: connect, connect_ex, sendto and create_connection.
    """
    real_connect = socket.socket.connect
    real_connect_ex = socket.socket.connect_ex
    real_sendto = socket.socket.sendto
    real_create_connection = socket.create_connection

    def refuse(host):
        raise RuntimeError(f"tests must not open network connections (tried {host!r})")

    def check(address):
        host = address[0] if isinstance(address, tuple) else address
        if host not in LOOPBACK:
            refuse(host)

    def guarded_connect(self, address):
        check(address)
        return real_connect(self, address)

    def guarded_connect_ex(self, address):
        check(address)
        return real_connect_ex(self, address)

    def guarded_sendto(self, data, *args):
        # sendto(data, address) or sendto(data, flags, address): the address is last.
        if args:
            check(args[-1])
        return real_sendto(self, data, *args)

    def guarded_create_connection(address, *args, **kwargs):
        check(address)
        return real_create_connection(address, *args, **kwargs)

    monkeypatch.setattr(socket.socket, "connect", guarded_connect)
    monkeypatch.setattr(socket.socket, "connect_ex", guarded_connect_ex)
    monkeypatch.setattr(socket.socket, "sendto", guarded_sendto)
    monkeypatch.setattr(socket, "create_connection", guarded_create_connection)


@pytest.fixture
def random_prices() -> pd.DataFrame:
    """Ten tickers, 400 business days, positive random-walk prices. Seeded."""
    rng = np.random.default_rng(7)
    index = pd.bdate_range("2019-01-01", periods=400)
    log_returns = rng.normal(0.0003, 0.012, size=(len(index), 10))
    prices = 100.0 * np.exp(np.cumsum(log_returns, axis=0))
    return pd.DataFrame(prices, index=index, columns=[f"T{i}" for i in range(10)])
