"""Duplicate detection for ideas (fuzzy text) and code (normalised structural hash).

Code duplicates are found by hashing the program's AST, with comments, docstrings and
formatting left out. Renaming a variable or changing a number does not make a copy,
because the hash keeps names and literals. That is deliberate: a rename is cheap, so the
check stays focused on what the code computes, not on how it looks.
"""

from __future__ import annotations

import ast
import hashlib
import io
import tokenize
from difflib import SequenceMatcher

# Only used for code that does not parse. The check still runs on it, but less carefully.
IGNORED_TOKENS = frozenset(
    {
        tokenize.COMMENT,
        tokenize.NL,
        tokenize.NEWLINE,
        tokenize.INDENT,
        tokenize.DEDENT,
        tokenize.ENCODING,
        tokenize.ENDMARKER,
    }
)

_DOCSTRING_OWNERS = (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)


def normalised_code_hash(code: str) -> str:
    """Hash of a program's structure. Comments, docstrings and formatting do not change it.

    Programs that differ only in layout, indentation width, comments or docstrings get the same
    hash. A change to a name, a literal, an operator or the block structure changes it. Code that
    does not parse is hashed from its tokens instead.
    """
    try:
        tree = ast.parse(code)
    except (SyntaxError, ValueError):  # ValueError: the source contains a null byte
        return _sha256(" ".join(_tokens(code)))
    _strip_docstrings(tree)
    return _sha256(ast.dump(tree))


def _tokens(code: str) -> list[str]:
    try:
        return [
            tok.string
            for tok in tokenize.generate_tokens(io.StringIO(code).readline)
            if tok.type not in IGNORED_TOKENS
        ]
    except (tokenize.TokenError, IndentationError, SyntaxError):
        return code.split()  # unparseable code is still hashed, just less carefully


def _strip_docstrings(tree: ast.AST) -> None:
    """Remove string-literal first statements, which are docstrings and do not run."""
    for node in ast.walk(tree):
        if not isinstance(node, _DOCSTRING_OWNERS) or not node.body:
            continue
        first = node.body[0]
        if (
            isinstance(first, ast.Expr)
            and isinstance(first.value, ast.Constant)
            and isinstance(first.value.value, str)
        ):
            node.body = node.body[1:] or [ast.Pass()]


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def idea_similarity(
    name_a: str, desc_a: str, name_b: str, desc_b: str, floor: float = 0.0
) -> float:
    """Similarity of two ideas from 0 to 1, computed on their names and descriptions together.

    autojunk is off. The default heuristic treats any character that makes up more than 1% of a
    text of 200 characters or more as junk, so one changed word in a long description scored near
    zero. The exact ratio is computed only when the cheap upper bounds reach `floor`. A pair below
    `floor` returns that upper bound instead, so a caller must treat any value below `floor` as
    "not similar enough", not as the exact ratio.
    """
    matcher = SequenceMatcher(
        None,
        f"{name_a}. {desc_a}".lower(),
        f"{name_b}. {desc_b}".lower(),
        autojunk=False,
    )
    bound = matcher.real_quick_ratio()
    if bound < floor:
        return bound
    bound = matcher.quick_ratio()
    if bound < floor:
        return bound
    return matcher.ratio()
