"""Child-process entry point for sandboxed agent code.

Run as: python -I sandbox_runner.py CODE_PATH ALLOWED_CSV TIMEOUT_SECONDS

Do not import this from the package. It runs as a script, in an isolated interpreter, and
imports nothing from alpha_lab.

Protocol:
  stdin   the prices, as Parquet bytes. Nothing holdout-related is ever written to a file.
  stdout  one JSON header line, then (only for status "ok") the weights as Parquet bytes.
          Agent print() output is redirected to stderr, so it cannot corrupt the frame.
  stderr  agent output and interpreter warnings. The parent keeps only the tail.

Layers, in order:
  1. The parent runs the static AST checks first (checks/static.py).
  2. The parent kills the whole process tree on timeout (checks/sandbox.py).
  3. This process limits its own address space (POSIX only) and starts a watchdog that
     exits on its own deadline, so the time limit does not depend on the parent.
  4. Imports from agent code are restricted to the allowlist.
  5. Direct open() calls from agent code are refused.
  6. Socket connections, datagram sends and name lookups are refused.

Layers 4 to 6 are checks inside one Python process. They are not an operating-system boundary:
the child runs as the same user and can still reach anything that user can read or connect to
through modules the static filter does not name. See the README sandbox section.
"""

import builtins
import contextlib
import io
import json
import os
import signal
import socket
import sys
import threading
import traceback

AGENT_MODULE = "agent_signal"
AGENT_FILENAME = "agent_signal.py"
ADDRESS_SPACE_BYTES = 8 * 1024**3
WATCHDOG_SLACK_SECONDS = 5


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


def main() -> None:
    code_path, allowed_csv, timeout_csv = sys.argv[1:4]
    allowed = set(allowed_csv.split(",")) - {""}

    _limit_address_space()
    _start_watchdog(int(timeout_csv) + WATCHDOG_SLACK_SECONDS)

    result_stream = sys.stdout.buffer  # the frame goes to the real stdout
    sys.stdout = sys.stderr  # agent print() calls go to stderr

    import pandas as pd  # imported before the guards go up

    try:
        with open(code_path, encoding="utf-8") as f:
            source = f.read()
        prices = pd.read_parquet(io.BytesIO(sys.stdin.buffer.read()))

        real_import = builtins.__import__
        real_open = builtins.open

        def guarded_import(name, globals=None, locals=None, fromlist=(), level=0):
            # An explicit __import__('x') call passes globals=None, so look at the caller's frame.
            importer = (globals or sys._getframe(1).f_globals).get("__name__")
            if importer == AGENT_MODULE:
                top = name.split(".")[0]
                if level or top not in allowed:
                    raise ImportError(f"import of {name!r} is not allowed in the sandbox")
            return real_import(name, globals, locals, fromlist, level)

        def guarded_open(*args, **kwargs):
            if sys._getframe(1).f_globals.get("__name__") == AGENT_MODULE:
                raise PermissionError("file access is not allowed in the sandbox")
            return real_open(*args, **kwargs)

        builtins.__import__ = guarded_import
        builtins.open = guarded_open
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
        buffer = io.BytesIO()
        weights.to_parquet(buffer)
        body = buffer.getvalue()
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
