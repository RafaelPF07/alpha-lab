"""AST checks run before any agent code is executed.

These checks are a first filter, not the security boundary. The sandbox and the
dynamic lookahead tests do the real work. The checks are strict on purpose,
because it is cheaper to reject code early than to explain a bad run later.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass, field

FORBIDDEN_CALLS = frozenset(
    {"open", "exec", "eval", "compile", "__import__", "getattr", "globals", "locals"}
)
# pandas and numpy have I/O helpers that read or write arbitrary paths.
FORBIDDEN_ATTRIBUTE_PREFIXES = ("read_", "to_")
FORBIDDEN_MODULE_NAMES = frozenset({"os", "sys", "subprocess", "socket", "pathlib"})


@dataclass
class StaticReport:
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.errors


def _line(node: ast.AST) -> str:
    return f"line {getattr(node, 'lineno', '?')}"


def static_check(code: str, allowed_imports: list[str]) -> StaticReport:
    report = StaticReport()
    allowed = set(allowed_imports) | {"__future__"}

    try:
        tree = ast.parse(code)
    except SyntaxError as exc:
        report.errors.append(f"syntax error at line {exc.lineno}: {exc.msg}")
        return report

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                top = alias.name.split(".")[0]
                if top not in allowed:
                    report.errors.append(f"import of {alias.name!r} is not allowed ({_line(node)})")

        elif isinstance(node, ast.ImportFrom):
            if node.level:
                report.errors.append(f"relative import is not allowed ({_line(node)})")
            top = (node.module or "").split(".")[0]
            if top not in allowed:
                report.errors.append(f"import from {node.module!r} is not allowed ({_line(node)})")

        elif isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name) and node.func.id in FORBIDDEN_CALLS:
                report.errors.append(f"call to {node.func.id}() is forbidden ({_line(node)})")
            if (
                isinstance(node.func, ast.Attribute)
                and node.func.attr == "shift"
                and _has_negative_shift(node)
            ):
                # A warning, not an error: the dynamic lookahead tests decide.
                report.warnings.append(
                    f"suspected lookahead: .shift() with a negative argument ({_line(node)})"
                )

        elif isinstance(node, ast.Attribute):
            if isinstance(node.value, ast.Name) and node.value.id in FORBIDDEN_MODULE_NAMES:
                report.errors.append(
                    f"attribute access on {node.value.id!r} is forbidden ({_line(node)})"
                )
            if node.attr.startswith(FORBIDDEN_ATTRIBUTE_PREFIXES):
                report.errors.append(f"file I/O method {node.attr!r} is forbidden ({_line(node)})")
            if node.attr.startswith("__"):
                report.errors.append(f"dunder attribute {node.attr!r} is forbidden ({_line(node)})")

        elif isinstance(node, ast.Name) and node.id in FORBIDDEN_MODULE_NAMES:
            report.errors.append(f"reference to {node.id!r} is forbidden ({_line(node)})")

    if not _defines_signal(tree):
        report.errors.append("code must define a top-level function signal(prices)")

    return report


def _has_negative_shift(call: ast.Call) -> bool:
    candidates = [*call.args, *(kw.value for kw in call.keywords)]
    for arg in candidates:
        if isinstance(arg, ast.UnaryOp) and isinstance(arg.op, ast.USub):
            return True
        if isinstance(arg, ast.Constant) and isinstance(arg.value, int) and arg.value < 0:
            return True
    return False


def _defines_signal(tree: ast.Module) -> bool:
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == "signal":
            params = [*node.args.posonlyargs, *node.args.args]
            return len(params) == 1
    return False
