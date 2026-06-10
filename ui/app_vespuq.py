"""Launcher for the VESP-UQ Mission Console.

    python ui/app_vespuq.py

A thin caller in the spirit of the root ``run_*.py`` wrappers: it makes ``vesp`` importable
from the source tree (no install required) and starts :func:`vesp.ui.app.main`. All pages drive
the documented entry points (``python -m vesp.uq.run`` / ``python -m vesp.uq.screen`` /
``VESPUQPlugin``), so anything done in the UI is reproducible from the CLI.
"""

from __future__ import annotations

import sys
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "src"
if SRC.is_dir() and str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from vesp.ui.app import main

if __name__ == "__main__":
    raise SystemExit(main())
