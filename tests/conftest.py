import socket

import numpy as np
import pandas as pd
import pytest

LOOPBACK = {"127.0.0.1", "::1", "localhost"}


@pytest.fixture(autouse=True)
def block_network(monkeypatch):
    """Stops accidental network use inside the test process. It is a test convenience, not isolation.

    It patches the Python socket functions that take an address: connect, connect_ex, sendto and
    sendmsg on socket.socket, create_connection, and the name lookups getaddrinfo, gethostbyname,
    gethostbyname_ex and gethostbyaddr. A call to any host outside LOOPBACK raises RuntimeError
    before the operating system is asked. C-level code and subprocesses are not covered. Loopback
    stays allowed, because asyncio on Windows builds its self-pipe on 127.0.0.1.
    """
    real_connect = socket.socket.connect
    real_connect_ex = socket.socket.connect_ex
    real_sendto = socket.socket.sendto
    real_create_connection = socket.create_connection
    real_getaddrinfo = socket.getaddrinfo
    real_gethostbyname = socket.gethostbyname
    real_gethostbyname_ex = socket.gethostbyname_ex
    real_gethostbyaddr = socket.gethostbyaddr
    # sendmsg does not exist on Windows, so it is patched only where it exists.
    real_sendmsg = getattr(socket.socket, "sendmsg", None)

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

    def guarded_sendmsg(self, buffers, *args, **kwargs):
        # sendmsg(buffers[, ancdata[, flags[, address]]]): address is the fourth argument. On a
        # connected socket it is absent or None, and there is no new destination to check.
        address = kwargs.get("address", args[2] if len(args) > 2 else None)
        if address is not None:
            check(address)
        return real_sendmsg(self, buffers, *args, **kwargs)

    def guarded_create_connection(address, *args, **kwargs):
        check(address)
        return real_create_connection(address, *args, **kwargs)

    def name_check(host):
        # None is the wildcard host for a passive bind. It makes no DNS query, so it is allowed.
        if host is None:
            return
        if isinstance(host, bytes):
            host = host.decode("ascii", "replace")
        if host not in LOOPBACK:
            refuse(host)

    def guarded_getaddrinfo(host, *args, **kwargs):
        name_check(host)
        return real_getaddrinfo(host, *args, **kwargs)

    def guarded_gethostbyname(host):
        name_check(host)
        return real_gethostbyname(host)

    def guarded_gethostbyname_ex(host):
        name_check(host)
        return real_gethostbyname_ex(host)

    def guarded_gethostbyaddr(host):
        name_check(host)
        return real_gethostbyaddr(host)

    monkeypatch.setattr(socket.socket, "connect", guarded_connect)
    monkeypatch.setattr(socket.socket, "connect_ex", guarded_connect_ex)
    monkeypatch.setattr(socket.socket, "sendto", guarded_sendto)
    if real_sendmsg is not None:
        monkeypatch.setattr(socket.socket, "sendmsg", guarded_sendmsg)
    monkeypatch.setattr(socket, "create_connection", guarded_create_connection)
    monkeypatch.setattr(socket, "getaddrinfo", guarded_getaddrinfo)
    monkeypatch.setattr(socket, "gethostbyname", guarded_gethostbyname)
    monkeypatch.setattr(socket, "gethostbyname_ex", guarded_gethostbyname_ex)
    monkeypatch.setattr(socket, "gethostbyaddr", guarded_gethostbyaddr)


@pytest.fixture
def random_prices() -> pd.DataFrame:
    """Ten tickers, 400 business days, positive random-walk prices. Seeded."""
    rng = np.random.default_rng(7)
    index = pd.bdate_range("2019-01-01", periods=400)
    log_returns = rng.normal(0.0003, 0.012, size=(len(index), 10))
    prices = 100.0 * np.exp(np.cumsum(log_returns, axis=0))
    return pd.DataFrame(prices, index=index, columns=[f"T{i}" for i in range(10)])
