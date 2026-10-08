"""Child-process entry point for sandboxed agent code. Run as: python -I sandbox_runner.py CODE IN OUT RESULT ALLOWED

Do not import this from the package. It runs as a script, in an isolated
interpreter, and imports nothing from alpha_lab.

Layers, in order:
  1. The parent runs the static AST checks first.
  2. This process runs the code with a timeout enforced by the parent.
  3. Imports from agent code are restricted to the allowlist.
  4. Direct open() calls from agent code are refused.
  5. Socket connections are refused.
This is not a hardened sandbox. Python cannot be fully sandboxed in-process.
"""

import builtins
import json
import socket
import sys
import traceback

AGENT_MODULE = "agent_signal"


def _refuse(*args, **kwargs):
    raise PermissionError("network access is not allowed in the sandbox")


def main() -> None:
    code_path, in_path, out_path, result_path, allowed_csv = sys.argv[1:6]
    allowed = set(allowed_csv.split(",")) - {""}

    import pandas as pd  # imported before the guards go up

    def write_result(payload: dict) -> None:
        with open(result_path, "w", encoding="utf-8") as f:
            json.dump(payload, f)

    try:
        with open(code_path, encoding="utf-8") as f:
            source = f.read()
        prices = pd.read_parquet(in_path)

        real_import = builtins.__import__
        real_open = builtins.open

        def guarded_import(name, globals=None, locals=None, fromlist=(), level=0):
            importer = (globals or {}).get("__name__")
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
        socket.socket.connect = _refuse
        socket.create_connection = _refuse

        namespace = {"__name__": AGENT_MODULE}
        # exec is the point of this runner. Static checks and the guards above make it safe enough.
        exec(compile(source, "agent_signal.py", "exec"), namespace)  # noqa: S102
        signal_fn = namespace.get("signal")
        if not callable(signal_fn):
            raise TypeError("signal is not defined as a callable")

        weights = signal_fn(prices.copy())
        if not isinstance(weights, pd.DataFrame):
            raise TypeError(f"signal must return a DataFrame, got {type(weights).__name__}")
        weights.to_parquet(out_path)
        write_result({"status": "ok"})
    except BaseException as exc:  # noqa: BLE001 - report everything to the parent
        write_result(
            {
                "status": "exception",
                "error_type": type(exc).__name__,
                "message": str(exc)[:1000],
                "traceback": traceback.format_exc()[-3000:],
            }
        )


if __name__ == "__main__":
    main()
