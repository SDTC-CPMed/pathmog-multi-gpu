#!/usr/bin/env python
"""
Shared logging verbosity for PathMoG / MOGHET trainers.

Levels (ascending detail)
-------------------------
0 quiet  — fold start/end, resume/skip, early stop, fatal errors, final summary
1 normal — quiet + one line per epoch + best-model saves (default; good for Kaggle)
2 verbose — normal + tqdm bars + evaluation stats + soft warnings
3 debug  — verbose + per-batch loss / gradient diagnostics
"""

from __future__ import annotations

from typing import Union

LEVELS = {
    "quiet": 0,
    "normal": 1,
    "verbose": 2,
    "debug": 3,
}

# Aliases accepted on the CLI
_ALIASES = {
    "0": 0,
    "1": 1,
    "2": 2,
    "3": 3,
    "q": 0,
    "n": 1,
    "v": 2,
    "d": 3,
}


def parse_verbosity(value: Union[str, int]) -> int:
    """Convert CLI string/int to an integer verbosity level in [0, 3]."""
    if isinstance(value, int):
        return max(0, min(3, value))
    key = str(value).strip().lower()
    if key in LEVELS:
        return LEVELS[key]
    if key in _ALIASES:
        return _ALIASES[key]
    raise ValueError(
        f"Unknown verbosity '{value}'. Use quiet|normal|verbose|debug or 0|1|2|3."
    )


def log(level: int, message: str, *, min_level: int = 1) -> None:
    """Print ``message`` if current ``level`` is at least ``min_level``."""
    if level >= min_level:
        print(message, flush=True)
