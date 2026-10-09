"""Run agent code in a child process: a timeout, process containment, bounded output and a checked result.

What each platform enforces:
- Windows: the child is assigned to a Job Object with KILL_ON_JOB_CLOSE, a per-process memory
  limit, a job-wide memory limit and an active-process limit. Terminating the job kills every
  process in it, and closing the last handle does the same, so the tree also dies if this
  process is killed. If the job cannot be created or the child cannot be assigned to it, the
  timeout falls back to `taskkill /T`, which misses grandchildren that have already been orphaned.
- POSIX: the child leads its own process group. The whole group is killed on timeout and after
  the child exits, so grandchildren do not outlive the run. If this process dies first, nothing
  here kills the group; the child's own watchdog (sandbox_runner.py) covers that case.

Neither platform restricts what the child can read, write or connect to. The child runs as the
same operating-system user, so it can read anything that user can read, including .env.
"""

from __future__ import annotations

import contextlib
import functools
import io
import json
import os
import shutil
import signal as posix_signal
import subprocess
import sys
import tempfile
import threading
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from alpha_lab.config import PROJECT_ROOT

RUNNER = Path(__file__).with_name("sandbox_runner.py")

MAX_STDOUT_BYTES = 32 * 1024 * 1024
MAX_STDERR_BYTES = 64 * 1024
MAX_HEADER_BYTES = 4096
MAX_TRACEBACK_CHARS = 3000
PROCESS_MEMORY_BYTES = 1024**3
JOB_MEMORY_BYTES = 2 * 1024**3
ACTIVE_PROCESS_LIMIT = 16
JOIN_SECONDS = 5.0

JOB_OBJECT_LIMIT_ACTIVE_PROCESS = 0x00000008
JOB_OBJECT_LIMIT_PROCESS_MEMORY = 0x00000100
JOB_OBJECT_LIMIT_JOB_MEMORY = 0x00000200
JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE = 0x00002000
JOB_OBJECT_EXTENDED_LIMIT_INFORMATION = 9  # JOBOBJECTINFOCLASS value
PROCESS_TERMINATE = 0x0001
PROCESS_SET_QUOTA = 0x0100


@dataclass(frozen=True)
class SandboxOutcome:
    status: str  # ok | timeout | exception | wrong_output | all_zero
    weights: pd.DataFrame | None = None
    error_type: str | None = None
    message: str | None = None
    traceback: str | None = None

    @property
    def ok(self) -> bool:
        return self.status == "ok"


class SandboxFailure(RuntimeError):
    def __init__(self, outcome: SandboxOutcome) -> None:
        super().__init__(f"sandbox {outcome.status}: {outcome.message or outcome.error_type}")
        self.outcome = outcome


@dataclass(frozen=True)
class ContainedRun:
    timed_out: bool
    returncode: int | None
    stdout: bytes
    stdout_overflow: bool
    stderr: bytes
    containment: str  # job-object | process-group | taskkill


def kill_process_tree(proc: subprocess.Popen) -> None:
    """Kill a process and everything it started. Safe to call on a finished process.

    The tree is killed even when the direct child has already exited, because its descendants
    may still be running. Windows uses taskkill /T; POSIX kills the process group.
    """
    if os.name == "nt":
        subprocess.run(
            ["taskkill", "/F", "/T", "/PID", str(proc.pid)],
            capture_output=True,
            check=False,
        )
    else:
        with contextlib.suppress(ProcessLookupError, PermissionError):
            os.killpg(proc.pid, posix_signal.SIGKILL)
    with contextlib.suppress(OSError):
        proc.kill()
    proc.wait()


@functools.cache
def _win32_job_api():
    """ctypes bindings for the Job Object calls. Built on first use, on Windows only."""
    import ctypes
    from ctypes import wintypes

    class IO_COUNTERS(ctypes.Structure):
        _fields_ = [
            ("ReadOperationCount", ctypes.c_ulonglong),
            ("WriteOperationCount", ctypes.c_ulonglong),
            ("OtherOperationCount", ctypes.c_ulonglong),
            ("ReadTransferCount", ctypes.c_ulonglong),
            ("WriteTransferCount", ctypes.c_ulonglong),
            ("OtherTransferCount", ctypes.c_ulonglong),
        ]

    class BASIC_LIMIT_INFORMATION(ctypes.Structure):
        _fields_ = [
            ("PerProcessUserTimeLimit", ctypes.c_longlong),
            ("PerJobUserTimeLimit", ctypes.c_longlong),
            ("LimitFlags", wintypes.DWORD),
            ("MinimumWorkingSetSize", ctypes.c_size_t),
            ("MaximumWorkingSetSize", ctypes.c_size_t),
            ("ActiveProcessLimit", wintypes.DWORD),
            ("Affinity", ctypes.c_size_t),
            ("PriorityClass", wintypes.DWORD),
            ("SchedulingClass", wintypes.DWORD),
        ]

    class EXTENDED_LIMIT_INFORMATION(ctypes.Structure):
        _fields_ = [
            ("BasicLimitInformation", BASIC_LIMIT_INFORMATION),
            ("IoInfo", IO_COUNTERS),
            ("ProcessMemoryLimit", ctypes.c_size_t),
            ("JobMemoryLimit", ctypes.c_size_t),
            ("PeakProcessMemoryUsed", ctypes.c_size_t),
            ("PeakJobMemoryUsed", ctypes.c_size_t),
        ]

    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    handle = wintypes.HANDLE
    kernel32.CreateJobObjectW.argtypes = [ctypes.c_void_p, wintypes.LPCWSTR]
    kernel32.CreateJobObjectW.restype = handle
    kernel32.SetInformationJobObject.argtypes = [
        handle,
        ctypes.c_int,
        ctypes.c_void_p,
        wintypes.DWORD,
    ]
    kernel32.SetInformationJobObject.restype = wintypes.BOOL
    kernel32.QueryInformationJobObject.argtypes = [
        handle,
        ctypes.c_int,
        ctypes.c_void_p,
        wintypes.DWORD,
        ctypes.c_void_p,
    ]
    kernel32.QueryInformationJobObject.restype = wintypes.BOOL
    kernel32.AssignProcessToJobObject.argtypes = [handle, handle]
    kernel32.AssignProcessToJobObject.restype = wintypes.BOOL
    kernel32.TerminateJobObject.argtypes = [handle, wintypes.UINT]
    kernel32.TerminateJobObject.restype = wintypes.BOOL
    kernel32.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
    kernel32.OpenProcess.restype = handle
    kernel32.CloseHandle.argtypes = [handle]
    kernel32.CloseHandle.restype = wintypes.BOOL
    return kernel32, EXTENDED_LIMIT_INFORMATION


class _WindowsJob:
    name = "job-object"

    def __init__(self, handle: int, limit_type) -> None:
        self._handle = handle
        self._limit_type = limit_type

    @classmethod
    def create(cls) -> _WindowsJob | None:
        """A job that kills its processes when its last handle closes. None if it cannot be made."""
        import ctypes

        kernel32, limit_type = _win32_job_api()
        handle = kernel32.CreateJobObjectW(None, None)
        if not handle:
            return None
        info = limit_type()
        info.BasicLimitInformation.LimitFlags = (
            JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
            | JOB_OBJECT_LIMIT_ACTIVE_PROCESS
            | JOB_OBJECT_LIMIT_PROCESS_MEMORY
            | JOB_OBJECT_LIMIT_JOB_MEMORY
        )
        info.BasicLimitInformation.ActiveProcessLimit = ACTIVE_PROCESS_LIMIT
        info.ProcessMemoryLimit = PROCESS_MEMORY_BYTES
        info.JobMemoryLimit = JOB_MEMORY_BYTES
        configured = kernel32.SetInformationJobObject(
            handle,
            JOB_OBJECT_EXTENDED_LIMIT_INFORMATION,
            ctypes.byref(info),
            ctypes.sizeof(info),
        )
        if not configured:
            kernel32.CloseHandle(handle)
            return None
        return cls(handle, limit_type)

    def attach(self, pid: int) -> bool:
        kernel32, _ = _win32_job_api()
        process = kernel32.OpenProcess(PROCESS_SET_QUOTA | PROCESS_TERMINATE, False, pid)
        if not process:
            return False
        try:
            return bool(kernel32.AssignProcessToJobObject(self._handle, process))
        finally:
            kernel32.CloseHandle(process)

    def limit_flags(self) -> int:
        """The LimitFlags the kernel holds for this job. Used by the tests."""
        import ctypes

        kernel32, _ = _win32_job_api()
        info = self._limit_type()
        ok = kernel32.QueryInformationJobObject(
            self._handle,
            JOB_OBJECT_EXTENDED_LIMIT_INFORMATION,
            ctypes.byref(info),
            ctypes.sizeof(info),
            None,
        )
        if not ok:
            raise OSError(ctypes.get_last_error(), "QueryInformationJobObject failed")
        return int(info.BasicLimitInformation.LimitFlags)

    def kill(self) -> None:
        kernel32, _ = _win32_job_api()
        kernel32.TerminateJobObject(self._handle, 1)

    def close(self) -> None:
        kernel32, _ = _win32_job_api()
        kernel32.CloseHandle(self._handle)


class _PosixGroup:
    name = "process-group"

    def __init__(self, pgid: int) -> None:
        self._pgid = pgid

    def kill(self) -> None:
        with contextlib.suppress(ProcessLookupError, PermissionError):
            os.killpg(self._pgid, posix_signal.SIGKILL)

    def close(self) -> None:
        pass


class _TaskkillTree:
    name = "taskkill"

    def __init__(self, proc: subprocess.Popen) -> None:
        self._proc = proc

    def kill(self) -> None:
        kill_process_tree(self._proc)

    def close(self) -> None:
        pass


def _contain(proc: subprocess.Popen):
    """Choose the strongest containment this platform gives. Never raises."""
    if os.name != "nt":
        return _PosixGroup(proc.pid)
    try:
        job = _WindowsJob.create()
    except Exception:  # noqa: BLE001 - no job object means the taskkill fallback, not a failure
        job = None
    if job is not None:
        if job.attach(proc.pid):
            return job
        job.close()
    return _TaskkillTree(proc)


class _Capture(threading.Thread):
    """Drain one pipe on a thread. Keeps the head (stdout) or the tail (stderr), at most `limit`."""

    def __init__(self, stream, limit: int, keep_tail: bool) -> None:
        super().__init__(daemon=True)
        self._stream = stream
        self._limit = limit
        self._keep_tail = keep_tail
        self._buf = bytearray()
        self._total = 0

    def run(self) -> None:
        try:
            while True:
                chunk = self._stream.read1(65536)
                if not chunk:
                    break
                self._total += len(chunk)
                if self._keep_tail:
                    self._buf += chunk
                    excess = len(self._buf) - self._limit
                    if excess > 0:
                        del self._buf[:excess]
                elif len(self._buf) < self._limit:
                    self._buf += chunk[: self._limit - len(self._buf)]
        except (OSError, ValueError):
            pass
        finally:
            with contextlib.suppress(OSError, ValueError):
                self._stream.close()

    @property
    def data(self) -> bytes:
        return bytes(self._buf)

    @property
    def overflow(self) -> bool:
        return not self._keep_tail and self._total > self._limit


def _feed(stream, data: bytes) -> None:
    try:
        stream.write(data)
        stream.flush()
    except (OSError, ValueError):  # the child exited first; BrokenPipeError is an OSError
        pass
    finally:
        with contextlib.suppress(OSError, ValueError):
            stream.close()


def run_contained(
    cmd: list[str],
    *,
    cwd: Path,
    env: dict[str, str],
    stdin_data: bytes,
    timeout_seconds: float,
    max_stdout: int = MAX_STDOUT_BYTES,
    max_stderr: int = MAX_STDERR_BYTES,
) -> ContainedRun:
    """Run `cmd` with its whole process tree contained. Nothing it started outlives this call."""
    popen_kwargs: dict = {} if os.name == "nt" else {"start_new_session": True}
    proc = subprocess.Popen(
        cmd,
        cwd=cwd,
        env=env,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        **popen_kwargs,
    )
    containment = _contain(proc)
    out = _Capture(proc.stdout, max_stdout, keep_tail=False)
    err = _Capture(proc.stderr, max_stderr, keep_tail=True)
    feeder = threading.Thread(target=_feed, args=(proc.stdin, stdin_data), daemon=True)
    for thread in (out, err, feeder):
        thread.start()

    timed_out = False
    try:
        proc.wait(timeout=timeout_seconds)
    except subprocess.TimeoutExpired:
        timed_out = True
    finally:
        containment.kill()  # the whole tree, even when the leader has already exited
        proc.wait()
        for thread in (out, err, feeder):
            thread.join(timeout=JOIN_SECONDS)
        containment.close()
    return ContainedRun(
        timed_out=timed_out,
        returncode=proc.returncode,
        stdout=out.data,
        stdout_overflow=out.overflow,
        stderr=err.data,
        containment=containment.name,
    )


def _minimal_env() -> dict[str, str]:
    keep = ("SYSTEMROOT", "PATH", "TEMP", "TMP")
    return {k: os.environ[k] for k in keep if k in os.environ}


def _encode_prices(prices: pd.DataFrame) -> bytes:
    buffer = io.BytesIO()
    prices.to_parquet(buffer)
    return buffer.getvalue()


def _scrub(text: str) -> str:
    """Replace the repo, temp and home directories with placeholders. Prompts never see them."""
    roots = {str(PROJECT_ROOT), tempfile.gettempdir()}
    with contextlib.suppress(RuntimeError):
        roots.add(str(Path.home()))
    for root in sorted(roots, key=len, reverse=True):
        for variant in {root, root.replace("\\", "/")}:
            text = text.replace(variant, "<path>")
    return text


def _text(value, limit: int) -> str | None:
    if value is None:
        return None
    return _scrub(str(value))[:limit]


def _remove_workdir(path: Path) -> None:
    """Remove the work directory. A Windows file can stay locked for a moment, so retry once.

    TemporaryDirectory(ignore_cleanup_errors=True) is the last word. If the directory still cannot
    be removed, it is left behind instead of raised out of run(). It holds only the agent's code.
    """
    for attempt in (1, 2):
        try:
            shutil.rmtree(path)
            return
        except FileNotFoundError:
            return
        except OSError:
            if attempt == 2 or os.name != "nt":
                return
            time.sleep(0.5)


def _split_frame(stdout: bytes) -> tuple[dict, bytes] | None:
    newline = stdout.find(b"\n")
    if newline < 0 or newline > MAX_HEADER_BYTES:
        return None
    try:
        header = json.loads(stdout[:newline].decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        return None
    if not isinstance(header, dict):
        return None
    return header, stdout[newline + 1 :]


def _crash(stderr: bytes, message: str) -> SandboxOutcome:
    return SandboxOutcome(
        status="exception",
        error_type="SandboxCrash",
        message=message,
        traceback=_scrub(stderr.decode("utf-8", "replace")[-MAX_TRACEBACK_CHARS:]),
    )


def _parse_result(run: ContainedRun, prices: pd.DataFrame) -> SandboxOutcome:
    """Turn the child's framed stdout into an outcome. Nothing is trusted until it parses."""
    if run.stdout_overflow:
        limit_mb = MAX_STDOUT_BYTES // (1024 * 1024)
        return SandboxOutcome(status="wrong_output", message=f"signal output exceeds {limit_mb} MB")
    frame = _split_frame(run.stdout)
    if frame is None:
        return _crash(run.stderr, "the sandbox process exited without a result")
    header, body = frame
    if header.get("status") != "ok":
        return SandboxOutcome(
            status="exception",
            error_type=_text(header.get("error_type"), 200),
            message=_text(header.get("message"), 1000),
            traceback=_text(header.get("traceback"), MAX_TRACEBACK_CHARS),
        )
    if run.returncode != 0 or header.get("bytes") != len(body):
        return _crash(run.stderr, "the sandbox reported a result it did not finish writing")
    try:
        weights = pd.read_parquet(io.BytesIO(body))
    except Exception as exc:  # noqa: BLE001 - any unreadable output is a wrong output
        return SandboxOutcome(
            status="wrong_output",
            error_type=type(exc).__name__,
            message=f"output unreadable: {exc}",
        )
    return _validate_output(weights, prices)


class SandboxedSignal:
    """A callable `signal(prices) -> weights` that runs the agent's code in a subprocess."""

    def __init__(self, code: str, timeout_seconds: int, allowed_imports: list[str]) -> None:
        self.code = code
        self.timeout_seconds = timeout_seconds
        self.allowed_imports = allowed_imports

    def run(self, prices: pd.DataFrame) -> SandboxOutcome:
        # Prices go in on stdin and the result comes back on stdout. The only file written to
        # the work directory is the agent's own code, so holdout data never reaches the disk.
        with tempfile.TemporaryDirectory(
            prefix="alpha-sandbox-", ignore_cleanup_errors=True
        ) as tmp:
            workdir = Path(tmp)
            try:
                return self._run_in(workdir, prices)
            except OSError as exc:  # the child could not be started
                return SandboxOutcome(
                    status="exception",
                    error_type="SandboxLaunchError",
                    message=_scrub(f"the sandbox could not be started: {exc}"),
                )
            finally:
                _remove_workdir(workdir)

    def _run_in(self, workdir: Path, prices: pd.DataFrame) -> SandboxOutcome:
        code_path = workdir / "agent_signal.py"
        code_path.write_text(self.code, encoding="utf-8")
        cmd = [
            sys.executable,
            "-I",
            str(RUNNER),
            str(code_path),
            ",".join(self.allowed_imports),
            str(int(self.timeout_seconds)),
        ]
        run = run_contained(
            cmd,
            cwd=workdir,
            env=_minimal_env(),
            stdin_data=_encode_prices(prices),
            timeout_seconds=self.timeout_seconds,
        )
        if run.timed_out:
            return SandboxOutcome(
                status="timeout",
                message=f"signal() did not finish within {self.timeout_seconds} seconds",
            )
        return _parse_result(run, prices)

    def __call__(self, prices: pd.DataFrame) -> pd.DataFrame:
        outcome = self.run(prices)
        if not outcome.ok:
            raise SandboxFailure(outcome)
        assert outcome.weights is not None
        return outcome.weights


def _validate_output(weights: pd.DataFrame, prices: pd.DataFrame) -> SandboxOutcome:
    try:
        weights = weights.astype("float64")
    except Exception as exc:  # noqa: BLE001 - any unusable output is a wrong output
        return SandboxOutcome(
            status="wrong_output",
            error_type=type(exc).__name__,
            message=f"output unreadable: {exc}",
        )

    if not weights.index.equals(prices.index):
        return SandboxOutcome(
            status="wrong_output",
            message=(
                f"index mismatch: expected {len(prices.index)} rows from {prices.index[0].date()} "
                f"to {prices.index[-1].date()}, got {len(weights.index)} rows"
            ),
        )
    if list(weights.columns) != list(prices.columns):
        return SandboxOutcome(
            status="wrong_output",
            message=f"columns must match the input tickers {list(prices.columns)}",
        )
    values = weights.fillna(0.0).to_numpy()
    if not np.isfinite(values).all():
        return SandboxOutcome(status="wrong_output", message="weights contain infinite values")
    if not values.any():
        return SandboxOutcome(status="all_zero", message="every weight is zero")
    return SandboxOutcome(status="ok", weights=weights)
