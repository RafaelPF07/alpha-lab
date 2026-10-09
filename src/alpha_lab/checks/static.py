"""AST checks run before any agent code is executed.

These checks are a first filter, not the security boundary. The sandbox and the
dynamic lookahead tests do the real work. The checks are strict on purpose,
because it is cheaper to reject code early than to explain a bad run later.

The filter rejects names and spellings. Every rejected name is one that can reach a file, a
socket, the import machinery or the object graph. A spelling that is not listed is not caught,
so this filter is never the only layer. Attribute reads are the exception: no runtime hook sees
them, so the frame-attribute prefixes below are the only layer for frames (R4-SEC-02). This is a
blocklist, not isolation (DECISION-4, DECISION-11).
"""

from __future__ import annotations

import ast
from dataclasses import dataclass, field

# Builtins that load code, reach the object graph, or open files. Calling one, or reading one
# into another name, is rejected. Assigning to the bare name is harmless, so it is allowed.
FORBIDDEN_BUILTINS = frozenset(
    {
        "open",
        "exec",
        "eval",
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
    }
)
# Modules and attribute names that reach the operating system, the network, the import
# machinery or the garbage collector. Rejected as a name, an attribute, or a dotted import part,
# so `pd.io`, `np.lib.npyio.DataSource` and `from numpy import ctypeslib` are all caught.
#
# The second group is library helpers that reach a file, a process or the network from inside
# numpy, pandas or the stdlib (R3-SEC-01, R3-SEC-02). A name in this list is refused wherever
# it appears, and the runtime guards in sandbox_runner.py refuse the same names. This is a
# blocklist of spellings. It is not complete, and it is not a boundary (DECISION-4, DECISION-11).
FORBIDDEN_MODULE_NAMES = frozenset(
    {
        "os",
        "sys",
        "subprocess",
        "socket",
        "io",
        "pathlib",
        "builtins",
        "importlib",
        "ctypes",
        "gc",
        "posix",
        "nt",
        "pickle",
        "marshal",
        "shutil",
        "tempfile",
        "multiprocessing",
        "concurrent",
        "asyncio",
        "runpy",
        "urllib",
        "http",
        "ssl",
        "ftplib",
        "smtplib",
        "webbrowser",
        "npyio",
        "ctypeslib",
        "DataSource",
        # numpy and stdlib helpers that open files or spawn commands
        "testing",  # numpy.testing.extbuild.run_subprocess spawns a command (R3-SEC-01)
        "extbuild",
        "distutils",
        "f2py",
        "crackfortran",  # numpy.f2py.crackfortran reaches fileinput and codecs (R3-SEC-02)
        "fileinput",
        "codecs",
        "linecache",  # reads source files through a copy of the original open()
        "tokenize",
        "logging",
        "glob",
        "zipfile",
        "tarfile",
        "gzip",
        "bz2",
        "lzma",
        "mmap",
        "sqlite3",
        "dbm",
        "shelve",
        "site",
        "pkgutil",
        "zipimport",
        # pandas' bundled pyarrow reaches files and HTTP in C (R3-SEC-02)
        "pyarrow",
        "pa",
        "fs",
        "flight",
        # process creation (R3-SEC-01)
        "Popen",
        "run_subprocess",
        "check_call",
        "check_output",
        "system",
        "popen",
        "startfile",
        "fork",
        "forkpty",
        "posix_spawn",
        "posix_spawnp",
        "spawnl",
        "spawnle",
        "spawnlp",
        "spawnlpe",
        "spawnv",
        "spawnve",
        "spawnvp",
        "spawnvpe",
        "execl",
        "execle",
        "execlp",
        "execlpe",
        "execv",
        "execve",
        "execvp",
        "execvpe",
        # private C modules that reach processes, files and the registry (R4-SEC-01)
        "winapi",  # _winapi.CreateProcess raises no audit event
        "posixsubprocess",
        "msvcrt",
        "winreg",
        "pyio",  # _pyio is a pure-Python file object route
    }
)
# Frame and traceback attributes. A frame reached through f_back or a generator's gi_frame exposes
# the caller's locals, including the runner's own functions (R4-SEC-02). No runtime hook sees an
# attribute read, so this list is the only layer for that route. A name is refused by prefix.
FORBIDDEN_FRAME_PREFIXES = ("f_", "gi_", "cr_", "ag_", "tb_")
# String evaluators. DataFrame.eval and DataFrame.query resolve @names from the calling frame, so a
# string such as "@g.gi_frame" reaches a frame object with no attribute name in the code (R4-SEC-02).
FORBIDDEN_STRING_EVALUATORS = frozenset({"eval", "query"})
# Keyword arguments that name a path or a stream. `to_string(buf=...)` writes a file (R3-SEC-03).
FORBIDDEN_FILE_KEYWORDS = frozenset(
    {"buf", "path_or_buf", "filepath_or_buffer", "fname", "filename", "file", "path"}
)
# pandas and numpy have I/O helpers that read or write arbitrary paths. Only the
# real I/O method names are blocked. A prefix like "to_" would also block to_frame.
FORBIDDEN_IO_PREFIXES = ("read_",)
FORBIDDEN_IO_METHODS = frozenset(
    {
        "to_csv",
        "to_parquet",
        "to_pickle",
        "to_excel",
        "to_json",
        "to_hdf",
        "to_sql",
        "to_feather",
        "to_stata",
        "to_clipboard",
        "to_html",
        "to_latex",
        "to_markdown",
        "to_xml",
        "to_orc",
        "to_gbq",
        "to_records_file",
        "tofile",
        "dump",
        "save",
        "savez",
        "savez_compressed",
        "savetxt",
        "load",
        "loadtxt",
        "genfromtxt",
        "recfromtxt",
        "recfromcsv",
        "fromfile",
        "fromregex",
        "memmap",
        "open_memmap",
        "write_array",
        "ExcelFile",
        "ExcelWriter",
        "HDFStore",
        # Write-side methods: DataFrame.to_string(buf=...) and Figure.savefig write to a path (R3-SEC-03)
        "to_string",
        "savefig",
        "write",
        "writelines",
    }
)


@dataclass
class StaticReport:
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.errors


def _line(node: ast.AST) -> str:
    return f"line {getattr(node, 'lineno', '?')}"


def _name_problem(name: str) -> str | None:
    """Why a bare name, an attribute, or a dotted import part is forbidden. None if it is fine."""
    if name.startswith("__"):
        return f"dunder name {name!r} is forbidden"
    if name.lstrip("_") in FORBIDDEN_MODULE_NAMES:  # also _socket, _io and similar
        return f"reference to {name!r} is forbidden"
    if name.startswith(FORBIDDEN_FRAME_PREFIXES):
        return f"frame attribute {name!r} is forbidden"
    if name in FORBIDDEN_STRING_EVALUATORS:
        return f"string evaluator {name!r} is forbidden: it resolves names from the calling frame"
    if name.startswith("_"):
        return f"private attribute {name!r} is forbidden"
    if name.startswith(FORBIDDEN_IO_PREFIXES) or name in FORBIDDEN_IO_METHODS:
        return f"file I/O method {name!r} is forbidden"
    return None


def static_check(code: str, allowed_imports: list[str]) -> StaticReport:
    report = StaticReport()
    allowed = set(allowed_imports)  # __future__ is refused by the dunder rule, so it is not listed

    try:
        tree = ast.parse(code)
    except SyntaxError as exc:
        report.errors.append(f"syntax error at line {exc.lineno}: {exc.msg}")
        return report

    called = {id(node.func) for node in ast.walk(tree) if isinstance(node, ast.Call)}

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                parts = alias.name.split(".")
                if parts[0] not in allowed:
                    report.errors.append(f"import of {alias.name!r} is not allowed ({_line(node)})")
                for part in parts:
                    problem = _name_problem(part)
                    if problem:
                        report.errors.append(f"{problem} ({_line(node)})")

        elif isinstance(node, ast.ImportFrom):
            if node.level:
                report.errors.append(f"relative import is not allowed ({_line(node)})")
            parts = node.module.split(".") if node.module else []
            if not parts or parts[0] not in allowed:
                report.errors.append(f"import from {node.module!r} is not allowed ({_line(node)})")
            for part in parts:
                problem = _name_problem(part)
                if problem:
                    report.errors.append(f"{problem} ({_line(node)})")
            for alias in node.names:
                if alias.name == "*":
                    report.errors.append(f"star import is forbidden ({_line(node)})")
                    continue
                problem = _name_problem(alias.name)
                if problem:
                    report.errors.append(f"{problem} ({_line(node)})")

        elif isinstance(node, ast.Call):
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
            problem = _name_problem(node.attr)
            if problem:
                report.errors.append(f"{problem} ({_line(node)})")

        elif isinstance(node, ast.Name):
            if node.id.startswith("__") or node.id.lstrip("_") in FORBIDDEN_MODULE_NAMES:
                report.errors.append(f"{_name_problem(node.id)} ({_line(node)})")
            elif node.id in FORBIDDEN_BUILTINS and isinstance(node.ctx, ast.Load):
                if id(node) in called:
                    report.errors.append(f"call to {node.id}() is forbidden ({_line(node)})")
                else:
                    report.errors.append(f"reference to builtin {node.id}() is forbidden ({_line(node)})")

        elif isinstance(node, ast.keyword):
            if node.arg in FORBIDDEN_FILE_KEYWORDS:
                report.errors.append(
                    f"keyword argument {node.arg!r} names a file or stream and is forbidden "
                    f"({_line(node)})"
                )

        elif isinstance(node, ast.Subscript):
            key = node.slice
            if (
                isinstance(key, ast.Constant)
                and isinstance(key.value, str)
                and key.value.startswith("__")
            ):
                report.errors.append(f"dunder key {key.value!r} is forbidden ({_line(node)})")

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
