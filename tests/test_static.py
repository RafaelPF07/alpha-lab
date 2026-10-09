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


# ---- R4: spellings from the round-3 probes (R3-SEC-01, -02, -03). Each one must be refused. -----


@pytest.mark.parametrize(
    "statement",
    [
        # R3-SEC-01: process creation through numpy's test helpers
        "_ = numpy.testing.extbuild.run_subprocess(['cmd', '/c', 'echo', 'x'])",
        "from numpy.testing import extbuild",
        "import numpy.testing",
        "from numpy.testing._private.extbuild import run_subprocess",
        "import numpy.distutils",
        "_ = score.process_spawn_check(run_subprocess)",
        "_ = os_like.Popen(['x'])",
        "_ = score.system('echo x')",
        # R3-SEC-02: file and network I/O through pandas' bundled pyarrow and numpy's f2py
        "_ = numpy.f2py.crackfortran.fileinput.input(files=['x'])",
        "_ = numpy.f2py.crackfortran.codecs.open('x', 'r', 'utf-8')",
        "import codecs",
        "from numpy.f2py import crackfortran",
        "_ = pandas.compat.pyarrow.pa.OSFile('x', 'rb')",
        "_ = pandas.compat.pyarrow",
        "_ = pa.fs.S3FileSystem(endpoint_override='127.0.0.1:1')",
        "import pyarrow",
        "from pyarrow import fs",
        "import fileinput",
        # R3-SEC-03: writes that persist the holdout window
        "_ = score.to_string(buf='x')",
        "_ = score.to_string('x')",
        "_ = score.rank(path_or_buf='x')",
        "_ = score.rank(filename='x')",
        "score.savefig('x')",
    ],
)
def test_round_three_spellings_are_refused(statement):
    report = check(with_statement(statement))
    assert not report.ok, statement


def test_file_keyword_message_names_the_keyword():
    report = check(with_statement("_ = score.rank(buf='x')"))
    assert any("'buf'" in e for e in report.errors), report.errors


def test_runtime_denylist_matches_the_static_list():
    # The child refuses the same names at runtime. A name in only one list is a gap in one layer.
    import importlib.util

    from alpha_lab.checks import sandbox
    from alpha_lab.checks.static import FORBIDDEN_MODULE_NAMES

    spec = importlib.util.spec_from_file_location("runner_under_test", sandbox.RUNNER)
    runner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runner)
    assert runner.DENIED_IMPORT_PARTS == FORBIDDEN_MODULE_NAMES


def test_future_import_is_refused():
    # The dead allowlist entry is gone. The dunder rule refuses __future__ on its own.
    assert not check("from __future__ import annotations\n" + GOOD).ok


# ---- R5-F3: frame attributes and the modules added in round five (R4-SEC-01, R4-SEC-02) ----------


@pytest.mark.parametrize(
    "attribute",
    ["f_back", "f_locals", "f_globals", "f_code", "gi_frame", "gi_code", "cr_frame", "ag_frame", "tb_frame"],
)
def test_frame_attributes_are_refused(attribute):
    # R4-SEC-02: a frame reached through a generator, coroutine or traceback exposes the runner's locals.
    report = check(with_statement(f"_ = g.{attribute}"))
    assert not report.ok, attribute
    assert any("frame attribute" in e for e in report.errors), report.errors


def test_the_frame_route_probe_is_refused_statically():
    # R4-SEC-02, probe P12 as written by the reviewer: the generator frame, then f_back, then f_locals.
    code = (
        "def signal(prices):\n"
        "    holder = []\n"
        "    def gen():\n"
        "        yield holder[0].gi_frame.f_back\n"
        "    g = gen()\n"
        "    holder.append(g)\n"
        "    sf = next(g)\n"
        "    mf = sf.f_back\n"
        "    return mf.f_locals['real_import']\n"
    )
    report = check(code)
    assert not report.ok
    assert len(report.errors) >= 3, report.errors


@pytest.mark.parametrize(
    "statement",
    [
        # R4-SEC-01: modules that reach a file or a process by a route the import guard had let through
        "_ = _winapi.CreateProcess(None, 'x', None, None, False, 0, None, None, None)",
        "_ = numpy.winapi.CreateProcess(None, 'x', None, None, False, 0, None, None, None)",
        "_ = score.posixsubprocess",
        "_ = msvcrt.getch()",
        "_ = score.winreg",
        "_ = score.pyio",
        "import _pyio",
    ],
)
def test_process_and_file_modules_added_in_round_five_are_refused(statement):
    report = check(with_statement(statement))
    assert not report.ok, statement


def test_ordinary_attribute_names_are_not_caught_by_the_frame_prefixes():
    # The prefixes are f_, gi_, cr_, ag_ and tb_. Common names that merely contain them are fine.
    report = check(with_statement("_ = score.fillna(0.0).ffill().tb"))
    assert report.ok, report.errors


@pytest.mark.parametrize(
    "statement",
    [
        # R4-SEC-02: DataFrame.eval and query resolve @names from the calling frame, so a string
        # reaches frame objects with no attribute name in the code (probe: prices.eval("@g.gi_frame")).
        "_ = prices.eval('@score')",
        "_ = prices.eval('@g.gi_frame')",
        "_ = prices.query('A > 0')",
        "_ = pandas.eval('score')",
        "from pandas import eval",
    ],
)
def test_string_evaluators_are_refused(statement):
    report = check(with_statement(statement))
    assert not report.ok, statement
    assert any("string evaluator" in e for e in report.errors), report.errors
