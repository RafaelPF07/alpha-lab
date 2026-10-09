"""Runs the JavaScript tests for the dashboard's browser modules under Node.

The tests live in tests/js. Each one loads the modules from src/alpha_lab/dashboard/static/js through
tests/js/_static.mjs, which copies them to a temporary folder marked as ES modules, so no package.json
is added to the served folder. The tests are skipped if Node.js is missing.
"""

import json
import shutil
import subprocess
from pathlib import Path

import pytest

from alpha_lab.dashboard.server import create_app

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


@pytest.mark.skipif(node is None, reason="Node.js is not installed")
def test_the_js_status_fixture_matches_what_the_api_serves(tmp_path):
    # The JavaScript tests load FIXTURE_STATUSES instead of the server, so this ties the two together (R4-FE-10).
    # The route is called as the app registers it: /api/statuses returns queries.status_sets() unchanged.
    module_url = (JS_TESTS / "_static.mjs").as_uri()
    script = (
        f"const m = await import({json.dumps(module_url)});"
        " console.log(JSON.stringify(m.FIXTURE_STATUSES));"
    )
    result = subprocess.run(
        [node, "--input-type=module", "-e", script],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    fixture = json.loads(result.stdout)

    app = create_app(ledger_path=tmp_path / "unused.db")
    route = next(r for r in app.routes if getattr(r, "path", None) == "/api/statuses")
    served = route.endpoint()
    assert set(fixture) <= set(served), "every list the tests use is served"
    for key, names in fixture.items():
        assert sorted(names) == sorted(served[key]), (
            f"{key}: the fixture and /api/statuses disagree"
        )
    assert not (tmp_path / "unused.db").exists(), "reading the statuses never creates a ledger"
