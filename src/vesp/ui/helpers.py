"""Small display and file helpers shared by Mission Console pages."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any


def fmt(value: Any, digits: int = 4) -> str:
    """Format a numeric UI value, returning ``--`` for missing or invalid data."""

    try:
        number = float(value)
    except (TypeError, ValueError):
        return "--"
    return f"{number:.{digits}g}" if math.isfinite(number) else "--"


def safe_read_json(path: Path) -> tuple[dict[str, Any] | None, str | None]:
    """Read a JSON object without allowing filesystem or decode errors to escape."""

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        return None, str(exc)
    if not isinstance(payload, dict):
        return None, "expected a JSON object"
    return payload, None
