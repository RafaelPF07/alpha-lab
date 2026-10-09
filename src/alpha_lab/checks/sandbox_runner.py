"""Child-process entry point for sandboxed agent code.

Run as: python -I sandbox_runner.py CODE_PATH ALLOWED_CSV TIMEOUT_SECONDS

Do not import this from the package. It runs as a script, in an isolated interpreter, and
imports nothing from alpha_lab.

Protocol:
  stdin   the prices, as a NumPy .npz archive (dates as int64 nanoseconds, column names, float64
          values). The archive is read with allow_pickle=False. Holdout data is never written to a
          file by this runner, but the agent code receives it and can write it (see below).
  stdout  one JSON header line, then (only for status "ok") the weights in the same .npz form.
          Agent print() output is redirected to stderr, so it cannot corrupt the frame.
  stderr  agent output and interpreter warnings. The parent keeps only the tail.

Layers, in order:
  1. The parent runs the static AST checks first (checks/static.py). They also refuse frame
     attributes (f_back, f_locals, gi_frame and the like). No runtime hook sees an attribute read,
     so this static refusal is the only layer for that route (R4-SEC-02).
  2. The parent kills the whole process tree on timeout, and on Windows the child is in a Job
     Object (checks/sandbox.py).
  3. This process limits its own address space (POSIX only) and starts a watchdog. The watchdog
     can end a Python-level loop without the parent, but not C code that holds the GIL, so the
     parent's timeout is the real deadline.
  4. pyarrow is hidden before pandas is imported. pandas then has no pyarrow, so no module holds
     a pyarrow file or HTTP client (R3-SEC-02).
  5. Process creation is refused: subprocess.Popen and the os.system, os.spawn* and os.exec*
     families (R3-SEC-01).
  6. Imports from agent code are restricted to the allowlist, and no dotted import may name a
     denied part (for example numpy.testing or numpy.f2py). The importer is the frame that runs
     the import. The globals argument of __import__ is ignored, so a caller cannot name numpy as
     its importer (R4-SEC-01).
  7. open() is refused for agent code on any frame of the call stack, so a library called by
     agent code cannot open a file for it (R3-SEC-05).
  8. Python-level socket connections, datagram sends and name lookups are refused.

Layers 4 to 8 are checks inside one Python process. They are not an operating-system boundary.
The child runs as the same user, so it can still reach anything that user can read. A module that
the import guard does not name is not stopped if agent code reaches it by another route, and a
library can write a file without going through open(). A C-level client that does not use the
socket module is not covered by layer 8. Nothing here is isolation. See the README sandbox
section and DECISION-4.
"""

import builtins
import contextlib
import io
import json
import os
import signal
import socket
import subprocess
import sys
import threading
import traceback

AGENT_MODULE = "agent_signal"
AGENT_FILENAME = "agent_signal.py"
ADDRESS_SPACE_BYTES = 8 * 1024**3
WATCHDOG_SLACK_SECONDS = 5

# Modules hidden before pandas is imported. Setting sys.modules[name] to None makes an import of
# that name raise ImportError, and pandas then runs without it.
HIDDEN_MODULES = ("pyarrow",)

# Dotted-name parts that agent code may not import, at runtime. This must cover the names in
# checks/static.py FORBIDDEN_MODULE_NAMES. tests/test_sandbox.py checks that it does.
DENIED_IMPORT_PARTS = frozenset(
    {
        "os", "sys", "subprocess", "socket", "io", "pathlib", "builtins", "importlib", "ctypes",
        "gc", "posix", "nt", "pickle", "marshal", "shutil", "tempfile", "multiprocessing",
        "concurrent", "asyncio", "runpy", "urllib", "http", "ssl", "ftplib", "smtplib",
        "webbrowser", "npyio", "ctypeslib", "DataSource", "testing", "extbuild", "distutils",
        "f2py", "crackfortran", "fileinput", "codecs", "linecache", "tokenize", "logging", "glob",
        "zipfile", "tarfile", "gzip", "bz2", "lzma", "mmap", "sqlite3", "dbm", "shelve", "site",
        "pkgutil", "zipimport", "pyarrow", "pa", "fs", "flight", "Popen", "run_subprocess",
        "check_call", "check_output", "system", "popen", "startfile", "fork", "forkpty",
        "posix_spawn", "posix_spawnp", "spawnl", "spawnle", "spawnlp", "spawnlpe", "spawnv",
        "spawnve", "spawnvp", "spawnvpe", "execl", "execle", "execlp", "execlpe", "execv",
        "execve", "execvp", "execvpe",
        # Private C modules that reach processes, files and the registry without the names above
        # (R4-SEC-01: _winapi.CreateProcess and ctypes.windll.kernel32.WinExec were reachable).
        "winapi", "posixsubprocess", "msvcrt", "winreg", "pyio",
    }
)

# os functions that start a process. They are replaced in the child.
PROCESS_FUNCTIONS = (
    "system", "popen", "startfile", "fork", "forkpty", "posix_spawn", "posix_spawnp",
    "spawnl", "spawnle", "spawnlp", "spawnlpe", "spawnv", "spawnve", "spawnvp", "spawnvpe",
    "execl", "execle", "execlp", "execlpe", "execv", "execve", "execvp", "execvpe",
)

# Builtins that agent code cannot name. Library code (pandas, numpy) keeps the full set.
WITHHELD_BUILTINS = frozenset(
    {
        "eval",
        "exec",
        "compile",
        "getattr",
        "setattr",
        "delattr",
        "globals",
        "locals",
        "vars",
        "type",
        "object",
        "breakpoint",
        "input",
        "help",
        "memoryview",
        "__loader__",
        "__spec__",
    }
)


def _agent_builtins(import_guard, open_guard) -> dict:
    """The builtins table for agent code: everything except WITHHELD_BUILTINS, with guards."""
    allowed = {
        name: value for name, value in vars(builtins).items() if name not in WITHHELD_BUILTINS
    }
    allowed["__import__"] = import_guard
    allowed["open"] = open_guard
    return allowed


def _refuse(*args, **kwargs):
    raise PermissionError("network access is not allowed in the sandbox")


def _refuse_process(*args, **kwargs):
    raise PermissionError("process creation is not allowed in the sandbox")


def _refuse_process_creation() -> None:
    """Replace every Python-level way to start a process. subprocess.run, call and check_output
    all reach Popen through the subprocess module's own name, so one replacement covers them."""
    subprocess.Popen = _refuse_process
    for name in PROCESS_FUNCTIONS:
        if hasattr(os, name):
            setattr(os, name, _refuse_process)


def _is_denied(part: str) -> bool:
    return part.startswith("__") or part.lstrip("_") in DENIED_IMPORT_PARTS


def _agent_on_stack() -> bool:
    """True if agent code is anywhere on the call stack, not only as the immediate caller."""
    frame = sys._getframe(1)
    while frame is not None:
        if frame.f_globals.get("__name__") == AGENT_MODULE:
            return True
        frame = frame.f_back
    return False


def _limit_address_space() -> None:
    """Self-applied limit. Setting soft and hard to the same value means the child cannot raise it."""
    if os.name == "nt":
        return  # Windows has no resource module; the Job Object limits memory there.
    import resource

    with contextlib.suppress(ValueError, OSError):
        resource.setrlimit(resource.RLIMIT_AS, (ADDRESS_SPACE_BYTES, ADDRESS_SPACE_BYTES))


def _start_watchdog(seconds: float) -> None:
    def expire() -> None:
        if os.name != "nt" and os.getpgrp() == os.getpid():
            # We lead our own process group (the parent started us with a new session), so this
            # reaches our grandchildren too. The guard stops the call from touching a shared group.
            with contextlib.suppress(OSError):
                os.killpg(os.getpgrp(), signal.SIGKILL)
        os._exit(3)

    timer = threading.Timer(seconds, expire)
    timer.daemon = True
    timer.start()


def _agent_traceback(exc: BaseException) -> str:
    """Only the frames from agent code. Runner frames and absolute paths are left out."""
    frames = [f for f in traceback.extract_tb(exc.__traceback__) if f.filename == AGENT_FILENAME]
    lines = ["Traceback (agent code only):\n"] if frames else []
    lines += traceback.format_list(frames)
    lines += traceback.format_exception_only(type(exc), exc)
    return "".join(lines)


def _write_frame(stream, header: dict, body: bytes = b"") -> None:
    stream.write(json.dumps(header).encode("utf-8") + b"\n" + body)
    stream.flush()


def _decode_frame(data: bytes, np, pd):
    """Prices (or weights) from the .npz form. Pickles are refused."""
    with np.load(io.BytesIO(data), allow_pickle=False) as arrays:
        index = pd.DatetimeIndex(arrays["index"].view("datetime64[ns]"), name="date")
        columns = [str(c) for c in arrays["columns"]]
        values = arrays["values"]
    return pd.DataFrame(values, index=index, columns=columns)


def _encode_frame(frame, np, pd) -> bytes:
    if not isinstance(frame.index, pd.DatetimeIndex):
        raise TypeError("signal must return a DataFrame indexed by dates")
    buffer = io.BytesIO()
    np.savez(
        buffer,
        index=frame.index.as_unit("ns").asi8,
        columns=np.array([str(c) for c in frame.columns], dtype=str),
        values=frame.to_numpy(dtype="float64"),
    )
    return buffer.getvalue()


def _install_guards(allowed: set[str]):
    """Install the import and open guards, and return them for the agent's builtins table.

    The real __import__ and open are closure cells of this function, not locals of main(). A frame
    that agent code reaches through f_back or f_locals therefore cannot hand them out (R4-SEC-02).
    """
    real_import = builtins.__import__
    real_open = builtins.open

    def guarded_import(name, globals=None, locals=None, fromlist=(), level=0):
        # The importer is the frame that runs the import. The globals argument belongs to the
        # caller, so it is ignored: an explicit call cannot name numpy and reach the real import
        # (R4-SEC-01). The real import gets the real caller's globals, which library code needs
        # for its own relative imports.
        frame = sys._getframe(1)
        if frame.f_globals.get("__name__") == AGENT_MODULE:
            top = name.split(".")[0]
            if level or top not in allowed:
                raise ImportError(f"import of {name!r} is not allowed in the sandbox")
            if any(_is_denied(part) for part in name.split(".")):
                raise ImportError(f"import of {name!r} is not allowed in the sandbox")
            if any(_is_denied(part) for part in (fromlist or ())):
                raise ImportError(f"import from {name!r} is not allowed in the sandbox")
        return real_import(name, frame.f_globals, None, fromlist, level)

    def guarded_open(*args, **kwargs):
        if _agent_on_stack():
            raise PermissionError("file access is not allowed in the sandbox")
        return real_open(*args, **kwargs)

    builtins.__import__ = guarded_import
    builtins.open = guarded_open
    return guarded_import, guarded_open


def main() -> None:
    code_path, allowed_csv, timeout_csv = sys.argv[1:4]
    allowed = set(allowed_csv.split(",")) - {""}

    _limit_address_space()
    _start_watchdog(int(timeout_csv) + WATCHDOG_SLACK_SECONDS)

    result_stream = sys.stdout.buffer  # the frame goes to the real stdout
    sys.stdout = sys.stderr  # agent print() calls go to stderr

    for name in HIDDEN_MODULES:
        sys.modules[name] = None  # before pandas is imported, so pandas never loads pyarrow
    import numpy as np
    import pandas as pd  # imported before the guards go up

    try:
        with open(code_path, encoding="utf-8") as f:
            source = f.read()
        prices = _decode_frame(sys.stdin.buffer.read(), np, pd)
        _refuse_process_creation()

        guarded_import, guarded_open = _install_guards(allowed)
        for name in ("connect", "connect_ex", "sendto", "sendmsg"):
            setattr(socket.socket, name, _refuse)
        for name in (
            "create_connection",
            "getaddrinfo",
            "gethostbyname",
            "gethostbyname_ex",
            "gethostbyaddr",
        ):
            setattr(socket, name, _refuse)

        agent_builtins = _agent_builtins(guarded_import, guarded_open)
        namespace = {"__name__": AGENT_MODULE, "__builtins__": agent_builtins}
        # exec is the point of this runner. Static checks and the guards above make it safe enough.
        exec(compile(source, AGENT_FILENAME, "exec"), namespace)  # noqa: S102
        signal_fn = namespace.get("signal")
        if not callable(signal_fn):
            raise TypeError("signal is not defined as a callable")

        weights = signal_fn(prices.copy())
        if not isinstance(weights, pd.DataFrame):
            raise TypeError(f"signal must return a DataFrame, got {type(weights).__name__}")
        body = _encode_frame(weights, np, pd)
        _write_frame(result_stream, {"status": "ok", "bytes": len(body)}, body)
    except BaseException as exc:  # noqa: BLE001 - report everything to the parent
        _write_frame(
            result_stream,
            {
                "status": "exception",
                "error_type": type(exc).__name__,
                "message": str(exc)[:1000],
                "traceback": _agent_traceback(exc)[-3000:],
            },
        )


if __name__ == "__main__":
    main()
