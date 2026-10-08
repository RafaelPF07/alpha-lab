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


def test_good_code_passes_without_warnings():
    report = check(GOOD)
    assert report.ok
    assert report.warnings == []


def test_disallowed_import_is_rejected():
    assert not check("import os\n" + GOOD).ok
    assert not check("from scipy import stats\n" + GOOD).ok


def test_forbidden_calls_are_rejected():
    for call in [
        "open('x')",
        "eval('1')",
        "exec('1')",
        "compile('1', 'x', 'eval')",
        "getattr(pd, 'x')",
        "globals()",
        "locals()",
        "__import__('os')",
    ]:
        report = check(GOOD.replace("return score", f"_ = {call}\n    return score"))
        assert not report.ok, call


def test_file_io_methods_are_rejected():
    report = check(GOOD.replace("return score", "_ = pd.read_csv('/etc/passwd')\n    return score"))
    assert not report.ok
    assert any("read_csv" in e for e in report.errors)
    report = check(GOOD.replace("return score", "score.to_parquet('x')\n    return score"))
    assert not report.ok


def test_module_names_are_rejected():
    report = check(GOOD.replace("return score", "_ = sys.path\n    return score"))
    assert not report.ok


def test_dunder_access_is_rejected():
    report = check(GOOD.replace("return score", "_ = score.__class__\n    return score"))
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
