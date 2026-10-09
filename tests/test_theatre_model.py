"""Runs the JavaScript tests for the dashboard's browser modules under Node.

The tests live in tests/js. Each one loads the modules from src/alpha_lab/dashboard/static/js through
tests/js/_static.mjs, which copies them to a temporary folder marked as ES modules, so no package.json
is added to the served folder. The tests are skipped if Node.js is missing.
"""

import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
JS_TESTS = ROOT / "tests/js"

node = shutil.which("node")


@pytest.mark.skipif(node is None, reason="Node.js is not installed")
def test_theatre_event_model_under_node():
    result = subprocess.run(
        [node, str(JS_TESTS / "theatre_model.test.mjs")],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "all theatre model checks passed" in result.stdout


@pytest.mark.skipif(node is None, reason="Node.js is not installed")
def test_all_dashboard_js_tests_under_node():
    tests = sorted(JS_TESTS.glob("*.test.mjs"))
    assert tests, "no JavaScript tests found in tests/js"
    result = subprocess.run(
        [node, "--test", *map(str, tests)],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=300,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
