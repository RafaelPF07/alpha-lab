"""Duplicate detection for ideas (fuzzy text) and code (normalised token hash)."""

from __future__ import annotations

import hashlib
import io
import tokenize
from difflib import SequenceMatcher

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


def normalised_code_hash(code: str) -> str:
    """Hash of the code's tokens with comments and whitespace removed.

    Two sources that differ only in formatting or comments get the same hash.
    Renaming a variable does not, which is deliberate: it is cheap to rename,
    so the check stays focused on what the code computes, not how it looks.
    """
    tokens = []
    try:
        for tok in tokenize.generate_tokens(io.StringIO(code).readline):
            if tok.type not in IGNORED_TOKENS:
                tokens.append(tok.string)
    except (tokenize.TokenError, IndentationError, SyntaxError):
        tokens = code.split()  # unparseable code is still hashed, just less carefully
    return hashlib.sha256(" ".join(tokens).encode("utf-8")).hexdigest()


def idea_similarity(name_a: str, desc_a: str, name_b: str, desc_b: str) -> float:
    a = f"{name_a}. {desc_a}".lower()
    b = f"{name_b}. {desc_b}".lower()
    return SequenceMatcher(None, a, b).ratio()
