import subprocess
import sys
import time

import numpy as np
import pandas as pd
import pytest

from alpha_lab.checks.sandbox import SandboxedSignal, SandboxFailure, kill_process_tree

ALLOWED = ["numpy", "pandas", "math"]


@pytest.fixture
def prices():
    index = pd.bdate_range("2020-01-01", periods=30)
    return pd.DataFrame(
        np.linspace(100, 110, 30)[:, None] * np.ones((1, 3)), index=index, columns=["A", "B", "C"]
    )


def run(code, prices, timeout=20):
    return SandboxedSignal(code, timeout, ALLOWED).run(prices)


def test_good_signal_runs_and_returns_weights(prices):
    code = """
import pandas as pd

def signal(prices: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(1.0 / prices.shape[1], index=prices.index, columns=prices.columns)
"""
    outcome = run(code, prices)
    assert outcome.status == "ok"
    assert outcome.weights.shape == prices.shape
    assert outcome.weights.iloc[0].tolist() == pytest.approx([1 / 3] * 3)


def test_wrong_index_is_reported(prices):
    code = "import pandas as pd\n\ndef signal(prices):\n    return prices.iloc[:-1] * 0.0\n"
    outcome = run(code, prices)
    assert outcome.status == "wrong_output"
    assert "index mismatch" in outcome.message


def test_wrong_columns_are_reported(prices):
    code = "import pandas as pd\n\ndef signal(prices):\n    return (prices * 0.0).rename(columns={'A': 'Z'})\n"
    outcome = run(code, prices)
    assert outcome.status == "wrong_output"
    assert "columns" in outcome.message


def test_all_zero_weights_are_reported(prices):
    code = "import pandas as pd\n\ndef signal(prices):\n    return prices * 0.0\n"
    assert run(code, prices).status == "all_zero"


def test_exception_is_returned_with_traceback(prices):
    code = "def signal(prices):\n    raise ValueError('boom')\n"
    outcome = run(code, prices)
    assert outcome.status == "exception"
    assert outcome.error_type == "ValueError"
    assert "boom" in outcome.message
    assert "ValueError" in outcome.traceback


def test_runtime_guard_blocks_file_access_from_agent_code(prices):
    # Static checks would reject this first. This tests the runtime layer on its own.
    code = "def signal(prices):\n    open('whatever.txt')\n    return prices * 0.0\n"
    outcome = run(code, prices)
    assert outcome.status == "exception"
    assert outcome.error_type == "PermissionError"


def test_runtime_guard_blocks_disallowed_imports(prices):
    code = "import subprocess\n\ndef signal(prices):\n    return prices * 0.0\n"
    outcome = run(code, prices)
    assert outcome.status == "exception"
    assert outcome.error_type == "ImportError"


def test_timeout_kills_the_run_and_reports_it(prices):
    code = "def signal(prices):\n    while True:\n        pass\n"
    started = time.perf_counter()
    outcome = run(code, prices, timeout=3)
    assert outcome.status == "timeout"
    assert time.perf_counter() - started < 30


def test_callable_raises_on_failure(prices):
    with pytest.raises(SandboxFailure):
        SandboxedSignal("def signal(prices):\n    raise KeyError('x')\n", 20, ALLOWED)(prices)


def _pid_alive(pid: int) -> bool:
    if sys.platform == "win32":
        out = subprocess.run(
            ["tasklist", "/FI", f"PID eq {pid}"], capture_output=True, text=True, check=False
        ).stdout
        return str(pid) in out
    try:
        import os

        os.kill(pid, 0)
        return True
    except OSError:
        return False


def test_kill_process_tree_takes_out_child_processes(tmp_path):
    pid_file = tmp_path / "child.pid"
    helper = (
        "import subprocess, sys, time\n"
        f"child = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(60)'])\n"
        f"open(r'{pid_file}', 'w').write(str(child.pid))\n"
        "time.sleep(60)\n"
    )
    kwargs = {} if sys.platform == "win32" else {"start_new_session": True}
    proc = subprocess.Popen([sys.executable, "-c", helper], **kwargs)
    deadline = time.time() + 20
    while not pid_file.exists() and time.time() < deadline:
        time.sleep(0.1)
    child_pid = int(pid_file.read_text())
    assert _pid_alive(child_pid)

    kill_process_tree(proc)

    deadline = time.time() + 10
    while _pid_alive(child_pid) and time.time() < deadline:
        time.sleep(0.1)
    assert not _pid_alive(child_pid)
