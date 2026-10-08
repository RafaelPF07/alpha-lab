"""Run agent code in a separate Python process with a timeout, then validate its output.

Windows has no `resource` module, so memory and CPU limits are not enforced.
The only hard limit is the wall-clock timeout, and a timeout kills the whole
process tree (taskkill /T on Windows, a process group on POSIX).
"""

from __future__ import annotations

import contextlib
import json
import os
import signal as posix_signal
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

RUNNER = Path(__file__).with_name("sandbox_runner.py")


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


def kill_process_tree(proc: subprocess.Popen) -> None:
    """Kill a process and everything it started. Safe to call on a finished process."""
    if proc.poll() is not None:
        return
    if os.name == "nt":
        subprocess.run(
            ["taskkill", "/F", "/T", "/PID", str(proc.pid)],
            capture_output=True,
            check=False,
        )
    else:
        with contextlib.suppress(ProcessLookupError):
            os.killpg(proc.pid, posix_signal.SIGKILL)
    with contextlib.suppress(OSError):
        proc.kill()
    proc.wait()


def _minimal_env() -> dict[str, str]:
    keep = ("SYSTEMROOT", "PATH", "TEMP", "TMP")
    return {k: os.environ[k] for k in keep if k in os.environ}


class SandboxedSignal:
    """A callable `signal(prices) -> weights` that runs the agent's code in a subprocess."""

    def __init__(self, code: str, timeout_seconds: int, allowed_imports: list[str]) -> None:
        self.code = code
        self.timeout_seconds = timeout_seconds
        self.allowed_imports = allowed_imports

    def run(self, prices: pd.DataFrame) -> SandboxOutcome:
        with tempfile.TemporaryDirectory(prefix="alpha-sandbox-") as tmp:
            workdir = Path(tmp)
            code_path = workdir / "agent_signal.py"
            in_path = workdir / "prices.parquet"
            out_path = workdir / "weights.parquet"
            result_path = workdir / "result.json"

            code_path.write_text(self.code, encoding="utf-8")
            prices.to_parquet(in_path)

            cmd = [
                sys.executable,
                "-I",
                str(RUNNER),
                str(code_path),
                str(in_path),
                str(out_path),
                str(result_path),
                ",".join(self.allowed_imports),
            ]
            popen_kwargs: dict = {}
            if os.name != "nt":
                popen_kwargs["start_new_session"] = True
            proc = subprocess.Popen(
                cmd,
                cwd=workdir,
                env=_minimal_env(),
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                **popen_kwargs,
            )
            try:
                _, stderr = proc.communicate(timeout=self.timeout_seconds)
            except subprocess.TimeoutExpired:
                kill_process_tree(proc)
                return SandboxOutcome(
                    status="timeout",
                    message=f"signal() did not finish within {self.timeout_seconds} seconds",
                )

            if not result_path.exists():
                return SandboxOutcome(
                    status="exception",
                    error_type="SandboxCrash",
                    message="the sandbox process exited without a result",
                    traceback=stderr.decode("utf-8", "replace")[-3000:],
                )
            result = json.loads(result_path.read_text(encoding="utf-8"))
            if result["status"] != "ok":
                return SandboxOutcome(
                    status="exception",
                    error_type=result.get("error_type"),
                    message=result.get("message"),
                    traceback=result.get("traceback"),
                )
            return _validate_output(out_path, prices)

    def __call__(self, prices: pd.DataFrame) -> pd.DataFrame:
        outcome = self.run(prices)
        if not outcome.ok:
            raise SandboxFailure(outcome)
        assert outcome.weights is not None
        return outcome.weights


def _validate_output(out_path: Path, prices: pd.DataFrame) -> SandboxOutcome:
    try:
        weights = pd.read_parquet(out_path)
        weights = weights.astype("float64")
    except Exception as exc:  # noqa: BLE001 - any unreadable output is a wrong output
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
    if not np.isfinite(weights.fillna(0.0).to_numpy()).all():
        return SandboxOutcome(status="wrong_output", message="weights contain infinite values")
    if not weights.fillna(0.0).to_numpy().any():
        return SandboxOutcome(status="all_zero", message="every weight is zero")
    return SandboxOutcome(status="ok", weights=weights)
