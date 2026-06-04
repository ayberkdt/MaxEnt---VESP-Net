"""Run the minimum safety checklist before reporting numerical results."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _commands() -> list[list[str]]:
    py = sys.executable
    return [
        [py, "-m", "pytest", "tests/"],
        [py, "scripts/smoke_test.py"],
        [py, "-m", "vesp.training.train", "--config", "configs/discrete_single_shell.yaml"],
        [py, "-m", "vesp.training.train", "--config", "configs/discrete_multishell.yaml"],
        [py, "-m", "vesp.training.train", "--config", "configs/altitude_ood.yaml"],
        [py, "-m", "vesp.training.feasibility", "--config", "configs/feasibility_suite.yaml"],
    ]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run pre-results deterministic VESP safety checks.")
    parser.add_argument("--dry-run", action="store_true", help="Print commands without executing them.")
    args = parser.parse_args(argv)

    for index, command in enumerate(_commands(), start=1):
        display = " ".join(command)
        print(f"[{index}] {display}")
        if args.dry_run:
            continue
        completed = subprocess.run(command, cwd=ROOT)
        if completed.returncode != 0:
            print(f"PRE-RESULTS CHECK FAILED at step {index}: {display}")
            return completed.returncode

    print("PRE-RESULTS CHECK PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
