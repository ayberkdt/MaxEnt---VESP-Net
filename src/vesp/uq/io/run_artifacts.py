"""Shared run-artifact writer for VESP-UQ scripts.

The benchmark / audit / screening scripts historically wrote bare JSON/MD/CSV files with no
provenance. This helper routes every output through the atomic writers in
:mod:`vesp.common.artifacts`, injects a small ``_provenance`` block into each JSON, and writes a
``run_manifest.json`` recording the config snapshot, seed, environment, and a SHA-256 checksum +
byte size for every emitted file -- so a result can be traced back to the exact config and verified.

It uses the shared manifest builder from :mod:`vesp.common.artifacts` while keeping this
script-oriented API free of the training layout's ``checkpoints/`` side effect.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

from vesp.common.artifacts import (
    atomic_write_json,
    atomic_write_text,
    utc_now_iso,
    write_manifest,
)

MANIFEST_NAME = "run_manifest.json"


def write_run_artifacts(
    out_dir: str | Path,
    *,
    tool: str,
    config: Mapping[str, Any] | None = None,
    json_files: Mapping[str, Mapping[str, Any]] | None = None,
    text_files: Mapping[str, str] | None = None,
    artifact_files: Mapping[str, str | Path] | None = None,
    artifact_statuses: Mapping[str, Any] | None = None,
    inputs: Mapping[str, str | Path] | None = None,
    seed: Any = None,
    config_path: str | None = None,
    manifest_name: str = MANIFEST_NAME,
) -> dict:
    """Write a VESP-UQ script's outputs atomically with a provenance manifest + checksums.

    ``json_files`` maps filename -> JSON-able payload (a ``_provenance`` block is injected unless one
    is already present); ``text_files`` maps filename -> text (Markdown / CSV). ``artifact_files``
    maps logical artifact names to files that were already written (for example PNG/PDF figures);
    they are checksummed into the manifest's ``artifacts`` block without being rewritten and marked
    with ``origin: prewritten``. ``artifact_statuses`` adds optional machine-readable status fields.
    ``inputs`` maps a logical name -> path of a file the run CONSUMED (saved models, datasets,
    trajectory CSVs); each existing input is checksummed into the manifest's ``inputs`` block.
    Returns the manifest dict. Output filenames are preserved exactly; ``run_manifest.json`` is
    added alongside them.
    """

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    generated_at = utc_now_iso()
    cfg_map: Mapping[str, Any] = config if isinstance(config, Mapping) else {}
    if seed is None:
        seed = cfg_map.get("seed")
    if config_path is None:
        config_path = cfg_map.get("_config_path")
    provenance = {
        "tool": tool,
        "generated_at": generated_at,
        "seed": seed,
        "config_path": config_path,
    }

    written: list[str] = []
    for name, payload in (json_files or {}).items():
        body = dict(payload)
        body.setdefault("_provenance", provenance)
        atomic_write_json(out_dir / name, body)
        written.append(name)
    for name, text in (text_files or {}).items():
        atomic_write_text(out_dir / name, text)
        written.append(name)

    artifacts: dict[str, str | Path] = {name: out_dir / name for name in written}
    artifacts.update({str(name): path for name, path in (artifact_files or {}).items()})
    artifact_metadata = {
        name: {"origin": "generated"} for name in written
    }
    artifact_metadata.update(
        {str(name): {"origin": "prewritten"} for name in (artifact_files or {})}
    )
    return write_manifest(
        out_dir / manifest_name,
        created_at_utc=generated_at,
        config=cfg_map,
        artifacts=artifacts,
        inputs=inputs,
        artifact_statuses=artifact_statuses,
        artifact_metadata=artifact_metadata,
        manifest_metadata={
            "tool": tool,
            "seed": seed,
            "config_path": config_path,
        },
    )
