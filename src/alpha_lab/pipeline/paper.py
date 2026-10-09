"""Paper trading: out of scope for now.

Connecting a broker would mean: a paper-account API key in .env, a daily job that reads
the approved signal's latest weights, converts them to share counts, and places orders
against the paper account only. This module does not place orders.
"""


def place_paper_orders(*args: object, **kwargs: object) -> None:
    raise NotImplementedError("paper trading is out of scope; see README")
