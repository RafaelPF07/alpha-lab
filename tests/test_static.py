import pytest

from alpha_lab.checks.static import static_check

ALLOWED = ["numpy", "pandas", "math"]

GOOD = """
import numpy as np
import pandas as pd

def signal(prices: pd.DataFrame) -> pd.DataFrame:
    score = prices / prices.shift(5) - 1.0
    return score.rank(axis=1, pct=True).sub(0.5).fillna(0.0)
"""


def check(code):
    return static_check(code, ALLOWED)


def with_statement(statement: str) -> str:
    """GOOD with one statement added at the top of the function body."""
    return GOOD.replace("return score", f"{statement}\n    return score")


def test_good_code_passes_without_warnings():
    report = check(GOOD)
    assert report.ok
    assert report.warnings == []


def test_disallowed_import_is_rejected():
    assert not check("import os\n" + GOOD).ok
    assert not check("from scipy import stats\n" + GOOD).ok


@pytest.mark.parametrize(
    "statement",
    [
        "_ = open('x')",
        "_ = eval('1')",
        "_ = exec('1')",
        "_ = compile('1', 'x', 'eval')",
        "_ = getattr(pd, 'x')",
        "_ = setattr(pd, 'x', 1)",
        "delattr(pd, 'x')",
        "_ = globals()",
        "_ = locals()",
        "_ = vars(pd)",
        "_ = type(score)",
        "_ = object()",
        "breakpoint()",
        "_ = input()",
        "help(pd)",
        "_ = __import__('os')",
    ],
)
def test_forbidden_calls_are_rejected(statement):
    report = check(with_statement(statement))
    assert not report.ok, statement


def test_forbidden_call_message_names_the_call():
    report = check(with_statement("_ = open('x')"))
    assert any("call to open() is forbidden" in e for e in report.errors), report.errors


@pytest.mark.parametrize(
    "statement",
    ["ev = eval", "f = open", "g = getattr", "ctor = type", "lookup = vars"],
)
def test_builtin_aliases_are_rejected(statement):
    # Assigning a forbidden builtin to another name is how the old filter was bypassed.
    report = check(with_statement(statement))
    assert not report.ok, statement
    assert any("reference to builtin" in e for e in report.errors), report.errors


def test_dunder_names_and_keys_are_rejected():
    for statement in [
        "_ = score.__class__",
        "_ = __builtins__['__import__']('io')",
        "_ = {}['__class__']",
        "_ = __loader__",
    ]:
        report = check(with_statement(statement))
        assert not report.ok, statement


@pytest.mark.parametrize(
    "statement",
    [
        "_ = pd.read_csv('/etc/passwd')",
        "_ = pd.read_excel('x')",
        "_ = pd.ExcelFile('x')",
        "_ = np.load('x')",
        "_ = np.fromfile('x', dtype=float)",
        "_ = np.loadtxt('x')",
        "_ = np.lib.format.open_memmap('x')",
        "score.to_parquet('x')",
        "score.to_csv('x')",
    ],
)
def test_file_io_is_rejected(statement):
    report = check(with_statement(statement))
    assert not report.ok, statement


def test_file_io_message_names_the_method():
    report = check(with_statement("_ = pd.read_csv('/etc/passwd')"))
    assert any("'read_csv'" in e for e in report.errors), report.errors


@pytest.mark.parametrize(
    "statement",
    [
        "from pandas import read_csv",
        "from numpy import fromfile",
        "from numpy import ctypeslib",
        "from pandas import *",
        "from numpy.lib import npyio",
        "from numpy.lib.npyio import DataSource",
        "import numpy.ctypeslib",
        "import numpy.lib.npyio",
    ],
)
def test_import_forms_that_name_forbidden_members_are_rejected(statement):
    report = check(with_statement(statement))
    assert not report.ok, statement


@pytest.mark.parametrize(
    "name",
    [
        "os",
        "sys",
        "subprocess",
        "socket",
        "_socket",
        "io",
        "_io",
        "importlib",
        "ctypes",
        "builtins",
        "gc",
        "pathlib",
    ],
)
def test_forbidden_module_names_are_rejected(name):
    report = check(with_statement(f"_ = {name}.x"))
    assert not report.ok, name


def test_pandas_io_namespace_is_rejected():
    report = check(with_statement("_ = pd.io.common.get_handle('x', 'r')"))
    assert not report.ok
    assert any("'io'" in e for e in report.errors), report.errors


def test_numpy_datasource_is_rejected():
    # np.lib.npyio.DataSource opens URLs and files. The socket guard is the runtime layer.
    report = check(with_statement("_ = np.lib.npyio.DataSource().open('http://127.0.0.1/x')"))
    assert not report.ok
    assert any("npyio" in e for e in report.errors), report.errors
    assert any("DataSource" in e for e in report.errors), report.errors


def test_private_attributes_are_rejected():
    report = check(with_statement("_ = score._mgr"))
    assert not report.ok


def test_signal_column_attribute_is_allowed():
    # Names such as `signal` are common column names, so only the forbidden set is blocked.
    report = check(with_statement("_ = score.signal"))
    assert report.ok, report.errors


def test_dunder_access_is_rejected():
    report = check(with_statement("_ = score.__class__"))
    assert not report.ok


def test_missing_signal_is_rejected():
    assert not check("import pandas as pd\n\ndef other(prices):\n    return prices\n").ok


def test_signal_with_wrong_arity_is_rejected():
    assert not check("import pandas as pd\n\ndef signal(a, b):\n    return a\n").ok


def test_negative_shift_is_a_warning_not_an_error():
    report = check(GOOD.replace("prices.shift(5)", "prices.shift(-1)"))
    assert report.ok
    assert any("suspected lookahead" in w for w in report.warnings)


def test_syntax_error_is_reported():
    report = check("def signal(prices:\n")
    assert not report.ok
    assert "syntax error" in report.errors[0]


def test_pandas_methods_that_merely_start_with_to_are_allowed():
    report = check(with_statement("_ = score.to_frame()"))
    assert report.ok, report.errors
