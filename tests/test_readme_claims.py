"""README claims that name code: constants, status names, file and test names, CI commands and
two front-end phrases (R4-WHOLE-11).

These tests do not judge the prose. Each one checks a sentence in README.md that names a value,
a status, a file, a test, a CI command or a page phrase, and fails when the code no longer has it.
"""

import re
from datetime import timedelta
from pathlib import Path

from alpha_lab import statuses
from alpha_lab.agents import ideas, llm
from alpha_lab.backtest import engine
from alpha_lab.checks import sandbox
from alpha_lab.data import loader
from alpha_lab.pipeline import run

ROOT = Path(__file__).resolve().parents[1]
README = (ROOT / "README.md").read_text(encoding="utf-8")
STATIC = ROOT / "src" / "alpha_lab" / "dashboard" / "static"


def test_named_constants_have_the_values_the_readme_states():
    assert engine.WINDOW_TOLERANCE_DAYS == 7
    assert "7 calendar days" in README
    assert engine.MAX_INTERIOR_GAP_WEEKDAYS == 4
    assert "more than four weekdays" in README
    assert run.STALE_RUN_AFTER == timedelta(minutes=30)
    assert "30 minutes" in README
    assert llm.MAX_ATTEMPTS == 4
    assert "at most four times" in README
    assert llm.MAX_RETRY_AFTER_SECONDS == 60.0
    assert "up to 60 seconds" in README
    assert ideas.AVOID_LIST_SIZE == 30
    assert "up to 30 earlier ideas" in README
    assert loader.CLEANING_RULE == 2
    assert "currently 2" in README


def test_sandbox_limits_have_the_values_the_readme_states():
    assert sandbox.PROCESS_MEMORY_BYTES == 1024**3
    assert "1 GiB memory limit per process" in README
    assert sandbox.JOB_MEMORY_BYTES == 2 * 1024**3
    assert "2 GiB memory limit per job" in README
    assert sandbox.MAX_STDOUT_BYTES == 32 * 1024 * 1024
    assert "32 MiB of stdout" in README
    assert sandbox.MAX_STDERR_BYTES == 64 * 1024
    assert "last 64 KiB of stderr" in README
    # The POSIX address-space limit lives in the child runner, which is not imported here.
    runner_text = (ROOT / "src" / "alpha_lab" / "checks" / "sandbox_runner.py").read_text(
        encoding="utf-8"
    )
    assert "ADDRESS_SPACE_BYTES = 8 * 1024**3" in runner_text
    assert "8 GiB address-space limit" in README


def test_run_statuses_named_in_the_readme_exist():
    section = README.split("Run statuses (`runs.status`")[1].split("- **Unexpected errors")[0]
    named = set(re.findall(r"^\| `(\w+)` \|", section, re.MULTILINE))
    assert named, "the run status table was not found"
    assert named <= statuses.RUN_STATUSES, f"unknown run statuses: {named - statuses.RUN_STATUSES}"


PATH_REF = re.compile(r"`((?:src|tests)/[\w/.]+\.py)(?:::(\w+))?`")
BARE_TEST_REF = re.compile(r"`(test_\w+)`")


def test_paths_and_test_names_in_the_readme_exist():
    all_tests = "\n".join(p.read_text(encoding="utf-8") for p in (ROOT / "tests").glob("test_*.py"))
    for path, name in PATH_REF.findall(README):
        target = ROOT / path
        assert target.is_file(), f"README names {path}, which does not exist"
        if name:
            text = target.read_text(encoding="utf-8")
            assert re.search(rf"^def {name}\b", text, re.MULTILINE), f"{path} has no test {name}"
    for name in set(BARE_TEST_REF.findall(README)):
        assert re.search(rf"^def {name}\b", all_tests, re.MULTILINE), (
            f"README names test {name}, which no test file defines"
        )


def test_the_readme_ci_commands_are_the_workflow_commands():
    workflow = (ROOT / ".github" / "workflows" / "ci.yml").read_text(encoding="utf-8")
    for command in ("uv run ruff check src tests", "uv run pytest -q", "node --version"):
        assert command in workflow, f"ci.yml lost {command!r}"
        assert command in README, f"README does not name {command!r}"


def test_front_end_phrases_the_readme_quotes_are_on_the_pages():
    research = (STATIC / "research.html").read_text(encoding="utf-8")
    assert "The holdout window is not drawn" in research
    assert "holdout window is not drawn" in README
    assert "dsr_reason" in (STATIC / "js" / "pages" / "research.js").read_text(encoding="utf-8")
    assert "dsr_reason" in (STATIC / "js" / "pages" / "signal.js").read_text(encoding="utf-8")
    assert "Holdout run stale" in (STATIC / "js" / "core" / "signal_text.js").read_text(
        encoding="utf-8"
    )
    assert "stalled" not in README, "the page does not label a run stalled"
