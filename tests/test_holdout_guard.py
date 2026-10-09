"""Holdout protection, checked on the code and on the files (STAT-04, QA-09).

The AST scans below are tripwires for the spellings they know. They do not prove that no code
path can reach holdout data. What they do enforce is that the holdout file does not exist, that
the holdout window is requested from one function only, and that only the loader and the sandbox
read Parquet. The runtime tests check the same things while the loader runs.
"""

import ast
from datetime import timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from alpha_lab.config import load_settings, load_universe
from alpha_lab.data.loader import (
    PRE_HOLDOUT_FILE,
    fetch_and_cache,
    load_prices,
    write_cache,
    write_fingerprint,
)

SRC = Path(__file__).resolve().parents[1] / "src" / "alpha_lab"

# Files that may use each sensitive spelling. Anything else in src/ fails the scan.
ALLOWED = {
    "holdout_end": {"config.py", "data/splits.py", "pipeline/holdout.py"},
    "holdout_split": {"data/splits.py", "pipeline/holdout.py"},
    "download": {"data/loader.py", "pipeline/holdout.py"},
    "read_parquet": {"data/loader.py", "checks/sandbox.py", "checks/sandbox_runner.py"},
}
RETIRED_NAMES = {"HOLDOUT_FILE", "read_holdout_cache", "check_holdout_caller", "holdout_slice"}


def _modules():
    for path in sorted(SRC.rglob("*.py")):
        yield path.relative_to(SRC).as_posix(), ast.parse(path.read_text(encoding="utf-8"))


def _callee(node: ast.Call) -> str | None:
    if isinstance(node.func, ast.Name):
        return node.func.id
    if isinstance(node.func, ast.Attribute):
        return node.func.attr
    return None


def test_no_module_names_the_holdout_file_or_its_retired_readers():
    for rel, tree in _modules():
        for node in ast.walk(tree):
            if isinstance(node, ast.Constant) and isinstance(node.value, str):
                assert "prices_holdout" not in node.value, rel
            name = node.id if isinstance(node, ast.Name) else getattr(node, "attr", None)
            if name is not None:
                assert name not in RETIRED_NAMES, f"{rel}:{getattr(node, 'lineno', '?')}"


def test_the_holdout_end_date_is_read_only_where_the_holdout_lives():
    for rel, tree in _modules():
        for node in ast.walk(tree):
            if isinstance(node, ast.Attribute) and node.attr == "holdout_end":
                assert rel in ALLOWED["holdout_end"], f"{rel}:{node.lineno}"


def test_the_holdout_window_is_split_only_in_the_holdout_modules():
    for rel, tree in _modules():
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call) or _callee(node) not in {"split_window", "get_split"}:
                continue
            literals = [a.value for a in node.args if isinstance(a, ast.Constant)]
            literals += [k.value.value for k in node.keywords if isinstance(k.value, ast.Constant)]
            if "holdout" in literals:
                assert rel in ALLOWED["holdout_split"], f"{rel}:{node.lineno}"


def test_only_the_holdout_module_requests_prices_for_the_holdout_window():
    for rel, tree in _modules():
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and _callee(node) == "fetch_adjusted_close":
                assert rel in ALLOWED["download"], f"{rel}:{node.lineno}"


def test_only_the_loader_and_the_sandbox_read_parquet():
    for rel, tree in _modules():
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and _callee(node) == "read_parquet":
                assert rel in ALLOWED["read_parquet"], f"{rel}:{node.lineno}"


class RecordingSource:
    """A price source for tests. Records every request and returns flat synthetic prices."""

    def __init__(self):
        self.calls = []

    def fetch_adjusted_close(self, tickers, start, end_exclusive):
        self.calls.append((list(tickers), start, end_exclusive))
        index = pd.bdate_range(start, end_exclusive - timedelta(days=1))
        values = 100.0 + np.arange(len(index))[:, None] * np.ones((1, len(tickers)))
        return pd.DataFrame(values, index=index, columns=list(tickers))


def test_fetch_never_asks_for_the_holdout_window(tmp_path):
    settings, universe = load_settings(), load_universe()
    source = RecordingSource()
    fetch_and_cache(settings, universe, source, tmp_path)
    assert source.calls
    latest_allowed = settings.dates.validation_end + timedelta(days=1)
    for _tickers, _start, end_exclusive in source.calls:
        assert end_exclusive <= latest_allowed


def test_fetch_writes_no_holdout_file(tmp_path):
    settings, universe = load_settings(), load_universe()
    fetch_and_cache(settings, universe, RecordingSource(), tmp_path)
    names = sorted(p.name for p in tmp_path.iterdir())
    assert names == sorted([PRE_HOLDOUT_FILE, "cache_fingerprint.json", "cleaning_report.json"])


def test_load_prices_reads_only_the_pre_holdout_file(tmp_path, monkeypatch):
    settings, universe = load_settings(), load_universe()
    index = pd.bdate_range(settings.dates.start, settings.dates.validation_end)
    frame = pd.DataFrame(100.0, index=index, columns=list(universe.tickers))
    write_cache(frame, settings.dates, tmp_path)
    write_fingerprint(settings, universe, tmp_path)

    opened = []
    real_read_parquet = pd.read_parquet

    def recording_read_parquet(path, *args, **kwargs):
        opened.append(str(path))
        return real_read_parquet(path, *args, **kwargs)

    monkeypatch.setattr(pd, "read_parquet", recording_read_parquet)
    loaded = load_prices(settings, universe, data_dir=tmp_path)
    assert opened and all(PRE_HOLDOUT_FILE in p for p in opened), opened
    assert loaded.index.max() <= pd.Timestamp(settings.dates.validation_end)
