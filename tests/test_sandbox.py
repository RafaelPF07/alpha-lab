import contextlib
import csv
import http.server
import io
import os
import subprocess
import sys
import textwrap
import threading
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from alpha_lab.checks import sandbox
from alpha_lab.checks.sandbox import (
    MAX_HEADER_BYTES,
    RUNNER,
    ContainedRun,
    SandboxedSignal,
    SandboxFailure,
    _encode_prices,
    _minimal_env,
    _parse_result,
    _WindowsJob,
    kill_process_tree,
    run_contained,
)
from alpha_lab.config import PROJECT_ROOT

ALLOWED = ["numpy", "pandas", "math"]
SRC = PROJECT_ROOT / "src"
GOOD_CODE = """
import pandas as pd

def signal(prices: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(1.0 / prices.shape[1], index=prices.index, columns=prices.columns)
"""
GRANDCHILD = "import time; time.sleep(120)"


@pytest.fixture
def prices():
    index = pd.bdate_range("2020-01-01", periods=30)
    return pd.DataFrame(
        np.linspace(100, 110, 30)[:, None] * np.ones((1, 3)), index=index, columns=["A", "B", "C"]
    )


def run(code, prices, timeout=20):
    return SandboxedSignal(code, timeout, ALLOWED).run(prices)


# ---- outcomes -------------------------------------------------------------------------------


def test_good_signal_runs_and_returns_weights(prices):
    outcome = run(GOOD_CODE, prices)
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


def test_traceback_keeps_only_agent_frames(prices):
    code = (
        "def helper():\n    raise ValueError('boom')\n\n"
        "def signal(prices):\n    helper()\n    return prices\n"
    )
    outcome = run(code, prices)
    assert outcome.status == "exception"
    assert "agent_signal.py" in outcome.traceback
    assert "sandbox_runner" not in outcome.traceback
    assert str(PROJECT_ROOT) not in outcome.traceback


def test_callable_raises_on_failure(prices):
    with pytest.raises(SandboxFailure):
        SandboxedSignal("def signal(prices):\n    raise KeyError('x')\n", 20, ALLOWED)(prices)


# ---- runtime guards inside the child (not a boundary; see DECISION-4) ------------------------


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


def test_explicit_dunder_import_is_refused_at_runtime(prices):
    # SEC-03: an explicit __import__ call passes globals=None to the guard. It must still be refused.
    code = "def signal(prices):\n    __import__('subprocess')\n    return prices * 0.0\n"
    outcome = run(code, prices)
    assert outcome.status == "exception"
    assert outcome.error_type == "ImportError"


@pytest.mark.parametrize("name", ["vars", "type", "eval", "getattr", "globals"])
def test_withheld_builtins_are_not_visible_to_agent_code(prices, name):
    # Defence in depth behind the static filter: the names are simply not in the agent's builtins.
    code = f"def signal(prices):\n    {name}\n    return prices * 0.0\n"
    outcome = run(code, prices)
    assert outcome.status == "exception"
    assert outcome.error_type == "NameError", outcome


def test_ordinary_builtins_still_work_for_agent_code(prices):
    code = (
        "import pandas as pd\n\n"
        "def signal(prices):\n"
        "    n = len(range(3)) + int(float(2)) + sum(sorted([3, 1]))\n"
        "    pairs = list(zip([1, 2], [3, 4]))\n"
        "    assert isinstance(n, int) and pairs and max(1, 2) == 2 and abs(-1) == 1\n"
        "    return prices * 0.0 + n\n"
    )
    outcome = run(code, prices)
    assert outcome.status == "ok", outcome


@contextlib.contextmanager
def loopback_server():
    """A local HTTP server that records every request. It only listens on 127.0.0.1."""
    hits: list[str] = []

    class Handler(http.server.BaseHTTPRequestHandler):
        def do_GET(self):
            hits.append(self.path)
            self.send_response(200)
            self.send_header("Content-Type", "text/csv")
            self.end_headers()
            self.wfile.write(b"a,b\n1,2\n")

        def log_message(self, *args):
            pass

    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_address[1]}", hits
    finally:
        server.shutdown()
        server.server_close()


def test_socket_guard_stops_a_pandas_url_read(prices):
    with loopback_server() as (url, hits):
        code = (
            "import pandas as pd\n\ndef signal(prices):\n"
            f"    pd.read_csv({url + '/x.csv'!r})\n    return prices * 0.0\n"
        )
        outcome = run(code, prices)
    assert outcome.status == "exception"
    assert hits == []


def test_socket_guard_stops_numpy_datasource(prices):
    # QA-05: DataSource passes the static filter in older versions. The child's socket guard must stop it.
    with loopback_server() as (url, hits):
        code = (
            "import numpy as np\n\ndef signal(prices):\n"
            f"    np.lib.npyio.DataSource().open({url + '/x.csv'!r})\n    return prices * 0.0\n"
        )
        outcome = run(code, prices)
    assert outcome.status == "exception"
    assert hits == []


def test_sandbox_writes_only_the_agent_code_to_its_work_directory(prices, monkeypatch):
    # DECISION-2: prices travel on stdin, so the work directory holds the agent's code and nothing else.
    seen: dict[str, list[str]] = {}
    real_run_contained = sandbox.run_contained

    def spy(cmd, *, cwd, **kwargs):
        seen["files"] = sorted(p.name for p in Path(cwd).iterdir())
        return real_run_contained(cmd, cwd=cwd, **kwargs)

    monkeypatch.setattr(sandbox, "run_contained", spy)
    assert run(GOOD_CODE, prices).ok
    assert seen["files"] == ["agent_signal.py"]


def test_child_watchdog_ends_the_child_without_the_parent(tmp_path, prices):
    # SEC-04: the child exits on its own deadline. Here the deadline is 0 + 5 seconds.
    code_file = tmp_path / "agent_signal.py"
    code_file.write_text("def signal(prices):\n    while True:\n        pass\n", encoding="utf-8")
    cmd = [sys.executable, "-I", str(RUNNER), str(code_file), "numpy,pandas,math", "0"]
    result = run_contained(
        cmd,
        cwd=tmp_path,
        env=_minimal_env(),
        stdin_data=_encode_prices(prices),
        timeout_seconds=90,
    )
    assert not result.timed_out
    assert result.returncode in (3, -9)  # 3 from os._exit on Windows, SIGKILL on POSIX


# ---- process containment ------------------------------------------------------------------------


def _spawner_script(pid_file: Path, linger: bool) -> str:
    """Source for a process that starts a grandchild, publishes both pids atomically, then waits."""
    wait = "pass" if linger else "time.sleep(120)"
    return (
        "import os, subprocess, sys, time\n"
        f"child = subprocess.Popen([sys.executable, '-c', {GRANDCHILD!r}])\n"
        f"tmp = {str(pid_file) + '.tmp'!r}\n"
        "with open(tmp, 'w') as f:\n"
        "    f.write(f'{os.getpid()} {child.pid}')\n"
        f"os.replace(tmp, {str(pid_file)!r})\n"
        f"{wait}\n"
    )


def _read_pids(pid_file: Path, seconds: float = 30.0) -> tuple[int, int]:
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        if pid_file.exists():
            leader, grandchild = pid_file.read_text(encoding="utf-8").split()
            return int(leader), int(grandchild)
        time.sleep(0.1)
    raise AssertionError("the helper never published its pids")


def _pid_alive(pid: int) -> bool:
    if sys.platform == "win32":
        out = subprocess.run(
            ["tasklist", "/FI", f"PID eq {pid}", "/FO", "CSV", "/NH"],
            capture_output=True,
            text=True,
            check=False,
        ).stdout
        return any(len(row) > 1 and row[1].strip() == str(pid) for row in csv.reader(io.StringIO(out)))
    stat = Path(f"/proc/{pid}/stat")
    if stat.exists():
        # Field 3 is the state. State Z (zombie) has exited and counts as dead.
        state = stat.read_text(encoding="utf-8").rsplit(")", 1)[1].split()[0]
        return state != "Z"
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True


def _wait_dead(pid: int, seconds: float = 20.0) -> bool:
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        if not _pid_alive(pid):
            return True
        time.sleep(0.2)
    return not _pid_alive(pid)


def _kill_quietly(pid: int) -> None:
    with contextlib.suppress(OSError, subprocess.SubprocessError):
        if sys.platform == "win32":
            subprocess.run(
                ["taskkill", "/F", "/T", "/PID", str(pid)], capture_output=True, check=False
            )
        else:
            os.kill(pid, 9)


def test_grandchild_is_killed_on_timeout(tmp_path):
    pid_file = tmp_path / "pids.txt"
    cmd = [sys.executable, "-c", _spawner_script(pid_file, linger=False)]
    try:
        result = run_contained(
            cmd, cwd=tmp_path, env=_minimal_env(), stdin_data=b"", timeout_seconds=8
        )
        leader, grandchild = _read_pids(pid_file)
        assert result.timed_out
        assert _wait_dead(leader)
        assert _wait_dead(grandchild)
    finally:
        if pid_file.exists():
            for pid in _read_pids(pid_file):
                _kill_quietly(pid)


def test_grandchild_holding_the_pipes_does_not_block_or_survive(tmp_path):
    # SEC-04 (probes P08, P09): the leader returns a result while a grandchild keeps the pipes open.
    pid_file = tmp_path / "pids.txt"
    cmd = [sys.executable, "-c", _spawner_script(pid_file, linger=True)]
    try:
        started = time.monotonic()
        result = run_contained(
            cmd, cwd=tmp_path, env=_minimal_env(), stdin_data=b"", timeout_seconds=60
        )
        elapsed = time.monotonic() - started
        _, grandchild = _read_pids(pid_file)
        assert not result.timed_out
        assert result.returncode == 0
        assert elapsed < 30
        assert _wait_dead(grandchild)
    finally:
        if pid_file.exists():
            for pid in _read_pids(pid_file):
                _kill_quietly(pid)


@pytest.mark.skipif(sys.platform != "win32", reason="the parent-death guarantee is a Job Object feature")
def test_no_grandchild_survives_when_the_parent_process_is_killed(tmp_path):
    pid_file = tmp_path / "pids.txt"
    parent_code = textwrap.dedent(
        f"""
        import sys
        sys.path.insert(0, {str(SRC)!r})
        from alpha_lab.checks.sandbox import run_contained, _minimal_env
        run_contained(
            [sys.executable, "-c", {_spawner_script(pid_file, linger=False)!r}],
            cwd={str(tmp_path)!r}, env=_minimal_env(), stdin_data=b"", timeout_seconds=600,
        )
        """
    )
    parent = subprocess.Popen([sys.executable, "-c", parent_code])
    leader = grandchild = None
    try:
        leader, grandchild = _read_pids(pid_file)
        assert _pid_alive(leader) and _pid_alive(grandchild)
        parent.kill()
        parent.wait(timeout=30)
        assert _wait_dead(leader, seconds=30)
        assert _wait_dead(grandchild, seconds=30)
    finally:
        parent.kill()
        for pid in (leader, grandchild):
            if pid is not None:
                _kill_quietly(pid)


@pytest.mark.skipif(sys.platform != "win32", reason="Job Objects are Windows only")
def test_windows_job_has_kill_on_close_and_memory_limits():
    job = _WindowsJob.create()
    assert job is not None
    try:
        flags = job.limit_flags()
        assert flags & sandbox.JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
        assert flags & sandbox.JOB_OBJECT_LIMIT_PROCESS_MEMORY
        assert flags & sandbox.JOB_OBJECT_LIMIT_JOB_MEMORY
        assert flags & sandbox.JOB_OBJECT_LIMIT_ACTIVE_PROCESS
    finally:
        job.close()


@pytest.mark.skipif(sys.platform != "win32", reason="Job Objects are Windows only")
def test_windows_run_is_contained_by_a_job_object(tmp_path):
    result = run_contained(
        [sys.executable, "-c", "pass"],
        cwd=tmp_path,
        env=_minimal_env(),
        stdin_data=b"",
        timeout_seconds=30,
    )
    assert result.containment == "job-object"


@pytest.mark.skipif(sys.platform != "win32", reason="taskkill fallback is Windows only")
def test_taskkill_fallback_kills_the_tree_on_timeout(tmp_path, monkeypatch):
    monkeypatch.setattr(sandbox._WindowsJob, "create", classmethod(lambda cls: None))
    pid_file = tmp_path / "pids.txt"
    cmd = [sys.executable, "-c", _spawner_script(pid_file, linger=False)]
    try:
        result = run_contained(
            cmd, cwd=tmp_path, env=_minimal_env(), stdin_data=b"", timeout_seconds=8
        )
        leader, grandchild = _read_pids(pid_file)
        assert result.timed_out
        assert result.containment == "taskkill"
        assert _wait_dead(leader)
        assert _wait_dead(grandchild)
    finally:
        if pid_file.exists():
            for pid in _read_pids(pid_file):
                _kill_quietly(pid)


def test_kill_process_tree_takes_out_child_processes(tmp_path):
    pid_file = tmp_path / "child.pid"
    helper = (
        "import subprocess, sys, time, os\n"
        f"child = subprocess.Popen([sys.executable, '-c', {GRANDCHILD!r}])\n"
        f"tmp = {str(pid_file) + '.tmp'!r}\n"
        "open(tmp, 'w').write(str(child.pid))\n"
        f"os.replace(tmp, {str(pid_file)!r})\n"
        "time.sleep(60)\n"
    )
    kwargs = {} if sys.platform == "win32" else {"start_new_session": True}
    proc = subprocess.Popen([sys.executable, "-c", helper], **kwargs)
    try:
        deadline = time.time() + 20
        while not pid_file.exists() and time.time() < deadline:
            time.sleep(0.1)
        child_pid = int(pid_file.read_text(encoding="utf-8"))
        assert _pid_alive(child_pid)

        kill_process_tree(proc)

        assert _wait_dead(child_pid, seconds=10)
    finally:
        if proc.poll() is None:
            proc.kill()


# ---- bounded output and the parent's trust in the child --------------------------------------


def test_stdout_is_capped_at_the_limit(tmp_path):
    cmd = [sys.executable, "-c", "import sys; sys.stdout.write('x' * 2000000)"]
    result = run_contained(
        cmd,
        cwd=tmp_path,
        env=_minimal_env(),
        stdin_data=b"",
        timeout_seconds=30,
        max_stdout=1000,
    )
    assert result.stdout_overflow
    assert len(result.stdout) == 1000


def test_stderr_keeps_only_the_tail(tmp_path):
    cmd = [sys.executable, "-c", "import sys; sys.stderr.write('a' * 5000 + 'END')"]
    result = run_contained(
        cmd,
        cwd=tmp_path,
        env=_minimal_env(),
        stdin_data=b"",
        timeout_seconds=30,
        max_stderr=100,
    )
    assert len(result.stderr) <= 100
    assert result.stderr.endswith(b"END")


def _frame_run(stdout: bytes, returncode: int = 0, overflow: bool = False) -> ContainedRun:
    return ContainedRun(
        timed_out=False,
        returncode=returncode,
        stdout=stdout,
        stdout_overflow=overflow,
        stderr=b"",
        containment="test",
    )


def test_malformed_child_output_never_escapes_as_an_exception(prices):
    # SEC-06: anything the child writes is parsed defensively. Each case must give an outcome.
    body = b"not-a-parquet"
    not_parquet = b'{"status": "ok", "bytes": %d}\n' % len(body) + body
    cases = {
        "not json": (_frame_run(b"not json\n"), "SandboxCrash"),
        "no output": (_frame_run(b""), "SandboxCrash"),
        "header too long": (_frame_run(b"x" * (MAX_HEADER_BYTES + 1)), "SandboxCrash"),
        "body length mismatch": (
            _frame_run(b'{"status": "ok", "bytes": 999}\nabc'),
            "SandboxCrash",
        ),
        "ok then nonzero exit": (
            _frame_run(b'{"status": "ok", "bytes": 3}\nabc', returncode=3),
            "SandboxCrash",
        ),
        "unreadable weights": (_frame_run(not_parquet), "wrong_output"),
        "output over the limit": (_frame_run(b"", overflow=True), "wrong_output"),
    }
    for name, (run_result, expected) in cases.items():
        outcome = _parse_result(run_result, prices)
        got = outcome.error_type if outcome.status == "exception" else outcome.status
        assert got == expected, (name, outcome)


def _sandbox_dirs() -> set[str]:
    import tempfile

    return {p for p in os.listdir(tempfile.gettempdir()) if p.startswith("alpha-sandbox-")}


def test_cleanup_failure_does_not_escape_run(prices, monkeypatch):
    # SEC-05: a file that stays locked in the work directory must not raise out of run().
    # shutil reports each failure through a callback, so the locks are simulated on unlink/rmdir.
    import tempfile

    before = _sandbox_dirs()
    real_unlink, real_rmdir = os.unlink, os.rmdir

    def locked(real):
        def guarded(path, *args, **kwargs):
            if "alpha-sandbox-" in str(path):
                raise PermissionError(32, "file in use", str(path))
            return real(path, *args, **kwargs)

        return guarded

    monkeypatch.setattr(os, "unlink", locked(real_unlink))
    monkeypatch.setattr(os, "rmdir", locked(real_rmdir))
    monkeypatch.setattr(sandbox.time, "sleep", lambda _seconds: None)
    try:
        outcome = run(GOOD_CODE, prices)
    finally:
        monkeypatch.undo()
        for name in _sandbox_dirs() - before:  # leave nothing behind from the simulated lock
            sandbox.shutil.rmtree(Path(tempfile.gettempdir()) / name, ignore_errors=True)
    assert outcome.status == "ok"


@pytest.mark.skipif(sys.platform != "win32", reason="the retry is a Windows-only behaviour")
def test_windows_cleanup_is_retried_once(prices, monkeypatch):
    real_rmtree = sandbox.shutil.rmtree
    calls = []

    def fail_once(path, *args, **kwargs):
        calls.append(path)
        if len(calls) == 1:
            raise PermissionError(32, "file in use", str(path))
        return real_rmtree(path, *args, **kwargs)

    monkeypatch.setattr(sandbox.shutil, "rmtree", fail_once)
    monkeypatch.setattr(sandbox.time, "sleep", lambda _seconds: None)
    outcome = run(GOOD_CODE, prices)
    assert outcome.status == "ok"
    # The retry is the second call on the same work directory. TemporaryDirectory's own
    # cleanup adds a further call afterwards, which is why the check is on the first two.
    assert len(calls) >= 2
    assert calls[0] == calls[1]
