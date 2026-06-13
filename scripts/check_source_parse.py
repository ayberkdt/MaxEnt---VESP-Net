"""Decode and parse Python source files without writing bytecode."""

from __future__ import annotations

import argparse
import ast
from pathlib import Path


def iter_python_files(paths: list[str]) -> list[Path]:
    files: list[Path] = []
    for raw in paths:
        path = Path(raw)
        if path.is_file() and path.suffix == ".py":
            files.append(path)
        elif path.is_dir():
            files.extend(p for p in path.rglob("*.py") if "__pycache__" not in p.parts)
    return sorted({p.resolve() for p in files})


def check_files(files: list[Path]) -> list[str]:
    errors: list[str] = []
    for path in files:
        try:
            text = path.read_text(encoding="utf-8")
            ast.parse(text, filename=str(path))
        except Exception as exc:
            errors.append(f"{path}: {exc}")
    return errors


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="UTF-8 decode + ast.parse source check.")
    parser.add_argument("paths", nargs="+", help="Files or directories to scan")
    args = parser.parse_args(argv)

    files = iter_python_files(args.paths)
    errors = check_files(files)
    if errors:
        for error in errors:
            print(error)
        return 1
    print(f"source-parse-ok: {len(files)} files")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
