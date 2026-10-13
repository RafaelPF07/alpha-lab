import contextlib
import csv
import http.server
import io
import os
import socket
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
        "from pathlib import Path\n"
        f"child = subprocess.Popen([sys.executable, '-c', {GRANDCHILD!r}])\n"
        f"tmp = {str(pid_file) + '.tmp'!r}\n"
        "Path(tmp).write_text(f'{os.getpid()} {child.pid}')\n"
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
        "from pathlib import Path\n"
        f"child = subprocess.Popen([sys.executable, '-c', {GRANDCHILD!r}])\n"
        f"tmp = {str(pid_file) + '.tmp'!r}\n"
        "Path(tmp).write_text(str(child.pid))\n"
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
    created: list[str] = []
    real_mkdtemp = tempfile.mkdtemp

    def record(*args, **kwargs):
        path = real_mkdtemp(*args, **kwargs)
        created.append(path)
        return path

    monkeypatch.setattr(tempfile, "mkdtemp", record)
    try:
        outcome = run(GOOD_CODE, prices)
    finally:
        monkeypatch.undo()
        # Remove only the directory this test created. Other processes share the temp folder,
        # and their sandbox runs need their own directories (R3-QA-01).
        for directory in created:
            sandbox.shutil.rmtree(directory, ignore_errors=True)
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
    # One cleanup path: the failed attempt and its retry, on the same work directory, and nothing more.
    assert len(calls) == 2
    assert calls[0] == calls[1]


# ---- R4 (R3-SEC-01, -02, -03, -05): the child's runtime layer, with the static filter bypassed.
# SandboxedSignal.run does not run the static filter, so these tests exercise the child alone.


def test_process_creation_through_numpy_testing_is_refused(tmp_path, prices):
    # R3-SEC-01, probe X2. The child would write a marker file, so the test checks that the
    # process never ran, not which exception stopped it. On Windows the call reaches the process
    # guard and raises PermissionError. On Linux numpy.testing is loaded lazily and the import
    # guard can fail first with AttributeError, which also stops the process (CI, Ubuntu).
    marker = tmp_path / "child_ran.txt"
    command = [sys.executable, "-c", f"open({str(marker)!r}, 'w').write('ran')"]
    code = (
        "import numpy\n\ndef signal(prices):\n"
        f"    numpy.testing.extbuild.run_subprocess({command!r})\n"
        "    return prices * 0.0\n"
    )
    outcome = run(code, prices)
    assert outcome.status == "exception", outcome
    assert outcome.error_type in {"PermissionError", "AttributeError"}, outcome
    assert not marker.exists(), "the child process ran"


def test_pyarrow_is_not_loaded_in_the_child(tmp_path, prices):
    # R3-SEC-02, probe B03: the canary read through pandas' pyarrow must fail.
    canary = tmp_path / "canary.txt"
    canary.write_text("R4-CANARY-text\n", encoding="utf-8")
    code = (
        "import pandas\n\ndef signal(prices):\n"
        f"    raise ValueError(pandas.compat.pyarrow.pa.OSFile({str(canary)!r}, 'rb').read().decode())\n"
    )
    outcome = run(code, prices)
    assert outcome.status == "exception"
    assert "R4-CANARY" not in (outcome.message or "")
    assert outcome.error_type == "AttributeError"


def test_pyarrow_cannot_be_imported_by_agent_code(prices):
    code = "import pyarrow\n\ndef signal(prices):\n    return prices * 0.0\n"
    outcome = run(code, prices)
    assert outcome.status == "exception"
    assert outcome.error_type == "ImportError"


def test_a_dotted_import_of_a_denied_part_is_refused(prices):
    # R3-SEC-01 and -02: numpy is allowed, but numpy.testing is not.
    code = "import numpy.testing\n\ndef signal(prices):\n    return prices * 0.0\n"
    outcome = run(code, prices)
    assert outcome.status == "exception"
    assert outcome.error_type == "ImportError"


def test_to_string_cannot_write_a_file(tmp_path, prices):
    # R3-SEC-03, probe H6: a candidate must not persist the frame it was given (the file is refused).
    target = tmp_path / "written.txt"
    code = (
        "import pandas\n\ndef signal(prices):\n"
        f"    prices.to_string(buf={str(target)!r})\n    return prices * 0.0 + 1.0\n"
    )
    outcome = run(code, prices)
    assert not target.exists()
    assert outcome.status == "exception", outcome
    assert outcome.error_type == "PermissionError"


@pytest.mark.parametrize(
    "reader",
    [
        # R3-SEC-02 and -05, probes B05 and B06: stdlib file readers reached through numpy.
        "''.join(numpy.f2py.crackfortran.fileinput.input(files=[{path!r}]))",
        "numpy.f2py.crackfortran.codecs.open({path!r}, 'r', 'utf-8').read()",
    ],
)
def test_stdlib_readers_reached_through_numpy_cannot_read_a_file(tmp_path, prices, reader):
    canary = tmp_path / "canary.txt"
    canary.write_text("R4-CANARY-text\n", encoding="utf-8")
    code = (
        "import numpy\n\ndef signal(prices):\n"
        f"    raise ValueError(str({reader.format(path=str(canary))}))\n"
    )
    outcome = run(code, prices)
    assert outcome.status == "exception"
    assert "R4-CANARY" not in (outcome.message or "")


def test_time_zone_conversion_still_runs_under_the_open_guard(prices):
    # R3-SEC-05 trade-off: the stack-wide open() guard must not refuse ordinary library work.
    code = (
        "import pandas as pd\n\ndef signal(prices):\n"
        "    idx = pd.DatetimeIndex(prices.index).tz_localize('UTC').tz_convert('US/Eastern')\n"
        "    return prices * 0.0 + 1.0 / 3\n"
    )
    outcome = run(code, prices)
    assert outcome.status == "ok", outcome


def test_outcome_names_the_containment_that_ran(prices):
    # R3-SEC-10: a fallback to taskkill must be visible to the owner through the outcome.
    outcome = run(GOOD_CODE, prices)
    assert outcome.containment in {"job-object", "process-group", "taskkill"}


def test_scrub_ignores_case_and_replaces_the_repo_path():
    # R3-SEC-15: Windows paths are case-insensitive, so an upper-case copy is scrubbed too.
    upper = str(PROJECT_ROOT).upper()
    forward = str(PROJECT_ROOT).replace("\\", "/")
    scrubbed = sandbox._scrub(f"error in {upper} and {forward}")
    assert scrubbed == "error in <path> and <path>"


@pytest.mark.skipif(sys.platform != "win32", reason="8.3 short names are a Windows feature")
def test_scrub_replaces_the_eight_dot_three_form_of_a_path():
    short = sandbox._short_path(str(PROJECT_ROOT))
    if not short or short == str(PROJECT_ROOT):
        pytest.skip("this volume has no 8.3 short name for the repository path")
    assert sandbox._scrub(f"at {short}") == "at <path>"


def test_the_child_environment_carries_no_price_data_or_secrets(prices, monkeypatch):
    # R3-SEC-03: prices travel on stdin only. The child's environment has the four system variables
    # and nothing else, so a key or a holdout value in the parent's environment never reaches it.
    seen: dict[str, dict[str, str]] = {}
    real_run_contained = sandbox.run_contained

    def spy(cmd, *, cwd, env, **kwargs):
        seen["env"] = dict(env)
        return real_run_contained(cmd, cwd=cwd, env=env, **kwargs)

    monkeypatch.setenv("ALPHA_R4_FAKE_SECRET", "not-a-real-key-for-tests")
    monkeypatch.setattr(sandbox, "run_contained", spy)
    assert run(GOOD_CODE, prices).ok
    assert set(seen["env"]) <= {"SYSTEMROOT", "PATH", "TEMP", "TMP"}
    assert "ALPHA_R4_FAKE_SECRET" not in seen["env"]


# ---- R4-F6 items: the runner layer, traversal, and one owner for the work directory --------------


def _load_runner():
    import importlib.util

    spec = importlib.util.spec_from_file_location("runner_layer_under_test", RUNNER)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)  # the module guards its main() call, so this runs no agent code
    return module


def test_runner_layer_denies_import_parts_by_name():
    runner = _load_runner()
    assert runner._is_denied("testing")
    assert runner._is_denied("_socket")  # a leading underscore does not get past the name check
    assert runner._is_denied("__future__")
    assert not runner._is_denied("numpy")
    assert not runner._is_denied("pandas")


def test_runner_layer_open_guard_sees_agent_code_below_library_frames():
    # R3-SEC-05: a library frame sits between the agent and open(). The stack walk must still see it.
    runner = _load_runner()

    def library_open():  # stands in for a library frame that calls open()
        return runner._agent_on_stack()

    namespace = {"__name__": runner.AGENT_MODULE, "library_open": library_open}
    exec("def agent_entry():\n    return library_open()\n", namespace)  # noqa: S102
    assert namespace["agent_entry"]() is True
    assert runner._agent_on_stack() is False  # this test module is not agent code


def test_agent_code_cannot_read_through_dot_dot_segments(tmp_path, prices):
    # Path traversal: a path that climbs out of its directory is refused by the open() guard.
    secret = tmp_path / "secret.npy"
    np.save(secret, np.arange(3))
    traversal = str(tmp_path / "sub" / ".." / "secret.npy")
    code = (
        "import numpy as np\n\ndef signal(prices):\n"
        f"    np.load({traversal!r})\n    return prices * 0.0\n"
    )
    outcome = run(code, prices)
    assert outcome.status == "exception", outcome
    assert outcome.error_type == "PermissionError"


def test_a_normal_run_leaves_no_work_directory_behind(prices, monkeypatch):
    # The work directory has one cleanup path. Track the directory this run creates. The temp folder
    # is shared with other sandbox runs, so a before/after listing of it is not a reliable check.
    created: list[str] = []
    real_mkdtemp = sandbox.tempfile.mkdtemp

    def record(*args, **kwargs):
        path = real_mkdtemp(*args, **kwargs)
        created.append(path)
        return path

    monkeypatch.setattr(sandbox.tempfile, "mkdtemp", record)
    assert run(GOOD_CODE, prices).ok
    assert len(created) == 1
    assert not Path(created[0]).exists()


def test_the_repository_root_in_an_error_message_becomes_a_placeholder(prices):
    # Item (f): the repository root never reaches an agent-facing message.
    code = (
        "def signal(prices):\n"
        f"    raise ValueError('bad path ' + {str(PROJECT_ROOT)!r})\n"
    )
    outcome = run(code, prices)
    assert outcome.status == "exception", outcome
    assert "<path>" in outcome.message
    assert str(PROJECT_ROOT) not in outcome.message


def test_an_output_pipe_still_being_read_is_reported_as_its_own_outcome(prices, monkeypatch):
    # R3-QA-01: the reader threads did not finish, so the frame may be short. That is not a crash.
    def unfinished(cmd, *, cwd, env, stdin_data, timeout_seconds, **kwargs):
        return ContainedRun(
            timed_out=False,
            returncode=0,
            stdout=b"",
            stdout_overflow=False,
            stderr=b"",
            containment="test",
            readers_done=False,
        )

    monkeypatch.setattr(sandbox, "run_contained", unfinished)
    outcome = run(GOOD_CODE, prices)
    assert outcome.status == "exception"
    assert outcome.error_type == "SandboxOutputIncomplete"


# ---- R5-F3: routes that pass their own globals to __import__ and read frames (R4-SEC-01, -02, -06) --
# Probes P0 to P12 are in scratchpad\review\round4\sec4. These tests run through SandboxedSignal.run,
# which does not run the static filter, so each one exercises the runtime layer alone. The frame
# attribute gi_frame is used only to reach the builtins table. The guard must refuse the import
# whatever the caller passes as globals.

GUARD_PRELUDE = (
    "def signal(prices):\n"
    "    g = (x for x in [1])\n"
    "    imp = g.gi_frame.f_globals.get('__builtins__').get('__import__')\n"
)


@contextlib.contextmanager
def _accepting_listener():
    """A TCP listener on 127.0.0.1 that records every connection it accepts."""
    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.bind(("127.0.0.1", 0))
    srv.listen(5)
    srv.settimeout(0.2)
    accepted: list[int] = []
    stop = threading.Event()

    def accept_loop() -> None:
        while not stop.is_set():
            try:
                conn, _ = srv.accept()
            except OSError:
                continue
            accepted.append(1)
            conn.close()

    thread = threading.Thread(target=accept_loop, daemon=True)
    thread.start()
    try:
        yield srv.getsockname()[1], accepted
    finally:
        stop.set()
        thread.join(timeout=5)
        srv.close()


def test_p0_control_import_without_a_globals_argument_is_refused(prices):
    # The control: the guard refuses a direct __import__ from agent code. This held on HEAD too.
    outcome = run(GUARD_PRELUDE + "    imp('os')\n    return prices * 0.0\n", prices)
    assert outcome.status == "exception", outcome
    assert outcome.error_type == "ImportError"
    assert "not allowed in the sandbox" in outcome.message


def test_p1_explicit_globals_naming_numpy_does_not_import_os(prices):
    # R4-SEC-01, probe P1. On HEAD the guard read the importer from the globals argument, so os imported.
    code = GUARD_PRELUDE + (
        "    m = imp('os', {'__name__': 'numpy'})\n"
        "    raise ValueError('IMPORTED os, name=' + m.name)\n"
    )
    outcome = run(code, prices)
    assert outcome.error_type == "ImportError", outcome
    assert "not allowed in the sandbox" in outcome.message


@pytest.mark.parametrize("module", ["os", "_io", "_socket", "_winapi", "ctypes"])
def test_explicit_globals_cannot_choose_the_importer(prices, module):
    # R4-SEC-01 and R4-SEC-06. The refusal text is the guard's own, so a module that is missing on
    # this platform (_winapi on POSIX) raises a different message and does not pass.
    code = GUARD_PRELUDE + (
        f"    m = imp({module!r}, {{'__name__': 'numpy'}})\n    raise ValueError('IMPORTED')\n"
    )
    outcome = run(code, prices)
    assert outcome.error_type == "ImportError", outcome
    assert "not allowed in the sandbox" in outcome.message, outcome.message


def test_p2_os_reads_a_file_through_the_globals_argument_is_refused(tmp_path, prices):
    # R4-SEC-01, probe P2: os.open and os.read, with no open() call for the guard to see.
    canary = tmp_path / "canary.txt"
    canary.write_text("R4-CANARY-text\n", encoding="utf-8")
    code = GUARD_PRELUDE + (
        "    m = imp('os', {'__name__': 'numpy'})\n"
        f"    fd = m.open({str(canary)!r}, m.O_RDONLY)\n"
        "    data = m.read(fd, 200)\n"
        "    m.close(fd)\n"
        "    raise ValueError('READ ' + data.decode())\n"
    )
    outcome = run(code, prices)
    assert "R4-CANARY" not in (outcome.message or "")
    assert outcome.error_type == "ImportError", outcome


def test_p3b_tcp_connect_through_the_private_socket_module_is_refused(prices):
    # R4-SEC-01, probe P3b: _socket.SocketType is not the class the runner patched.
    with _accepting_listener() as (port, accepted):
        code = GUARD_PRELUDE + (
            "    m = imp('_socket', {'__name__': 'numpy'})\n"
            "    s = m.SocketType(m.AF_INET, m.SOCK_STREAM)\n"
            f"    s.connect(('127.0.0.1', {port}))\n"
            "    s.close()\n"
            "    raise ValueError('CONNECTED')\n"
        )
        outcome = run(code, prices)
    assert accepted == [], "the loopback listener accepted a connection from agent code"
    assert outcome.error_type == "ImportError", outcome


@pytest.mark.skipif(sys.platform != "win32", reason="WinExec is a Windows API")
def test_p4b_process_creation_through_ctypes_is_refused(tmp_path, prices):
    # R4-SEC-01, probe P4b. The command writes a marker file, so the test shows whether it ran.
    marker = tmp_path / "spawned.txt"
    command = ("cmd /c echo R5-PROBE > " + str(marker)).encode()
    code = GUARD_PRELUDE + (
        "    m = imp('ctypes', {'__name__': 'numpy'})\n"
        f"    m.windll.kernel32.WinExec({command!r}, 0)\n"
        "    raise ValueError('SPAWNED')\n"
    )
    outcome = run(code, prices)
    time.sleep(2)  # WinExec returns before the spawned command has written anything
    assert not marker.exists()
    assert outcome.error_type == "ImportError", outcome


def test_p6_a_file_read_through_private_io_is_refused(tmp_path, prices):
    # R4-SEC-01, probe P6: _io.FileIO reads the file without the open() guard.
    canary = tmp_path / "canary.txt"
    canary.write_text("R4-CANARY-text\n", encoding="utf-8")
    code = GUARD_PRELUDE + (
        "    m = imp('_io', {'__name__': 'numpy'})\n"
        f"    raise ValueError('READ ' + m.FileIO({str(canary)!r}).read().decode())\n"
    )
    outcome = run(code, prices)
    assert "R4-CANARY" not in (outcome.message or "")
    assert outcome.error_type == "ImportError", outcome


def test_p10b_a_static_clean_write_of_the_frame_through_private_io_is_refused(tmp_path, prices):
    # R4-SEC-01 and R3-SEC-03, probe P10b: the frame written with no refused name in the code.
    target = tmp_path / "frame.csv"
    code = GUARD_PRELUDE + (
        "    m = imp('_io', {'__name__': 'numpy'})\n"
        f"    f = m.open({str(target)!r}, 'w')\n"
        "    prices.agg('to_csv', 0, f)\n"
        "    f.close()\n"
        "    return prices * 0.0\n"
    )
    outcome = run(code, prices)
    assert not target.exists(), "the frame was written to disk by agent code"
    assert outcome.error_type == "ImportError", outcome


def test_p12_the_frame_route_to_the_runner_main_locals_is_refused(prices):
    # R4-SEC-02, probe P12: frame.f_back from a running generator reaches main(), whose locals held
    # the real import function. The runtime layer must not hand it out.
    code = (
        "def signal(prices):\n"
        "    holder = []\n"
        "    def gen():\n"
        "        yield holder[0].gi_frame.f_back\n"
        "    g = gen()\n"
        "    holder.append(g)\n"
        "    sf = next(g)\n"
        "    mf = sf.f_back\n"
        "    real_import = mf.f_locals['real_import']\n"
        "    os_mod = real_import('os')\n"
        "    raise ValueError('main() locals reached; os module name=' + os_mod.name)\n"
    )
    outcome = run(code, prices)
    assert "main() locals reached" not in (outcome.message or "")
    assert outcome.status == "exception", outcome
