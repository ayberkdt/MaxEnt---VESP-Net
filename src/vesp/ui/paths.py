"""Repository-relative paths + filesystem scans shared by the Mission Console pages."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from vesp.ui.helpers import safe_read_json


def repo_root() -> Path:
    """The repository root (this file lives at ``<root>/src/vesp/ui/paths.py``).

    Falls back to the current working directory when the source tree layout is not
    recognizable (e.g. an unusual install); every consumer treats the result as a base for
    *defaults* only, so a wrong guess degrades to empty pickers rather than errors.
    """

    candidate = Path(__file__).resolve().parents[3]
    if (candidate / "configs").is_dir() or (candidate / "pyproject.toml").is_file():
        return candidate
    return Path.cwd()


ROOT = repo_root()
CONFIG_DIR = ROOT / "configs" / "vespuq"
OUTPUTS_DIR = ROOT / "outputs"


def list_configs() -> list[Path]:
    """The VESP-UQ run configs shipped with the repo (sorted, may be empty)."""

    if not CONFIG_DIR.is_dir():
        return []
    return sorted(CONFIG_DIR.glob("*.yaml"))


def list_models(root: Path | None = None) -> list[Path]:
    """Persisted plugin artifacts under ``outputs/`` (newest first)."""

    base = root or OUTPUTS_DIR
    if not base.is_dir():
        return []
    found: list[tuple[float, Path]] = []
    try:
        candidates = base.rglob("vespuq_plugin.pt")
        for path in candidates:
            try:
                if path.is_file():
                    found.append((path.stat().st_mtime, path))
            except OSError:
                continue
    except OSError:
        return []
    return [path for _mtime, path in sorted(found, key=lambda item: item[0], reverse=True)]


@dataclass
class RunRecord:
    """One run directory summarized from its ``run_manifest.json``."""

    run_dir: Path
    manifest_path: Path
    created_at: str = ""
    kind: str = "other"  # train | serve | other
    metrics: dict = field(default_factory=dict)
    artifacts: dict = field(default_factory=dict)
    inputs: dict = field(default_factory=dict)

    @property
    def name(self) -> str:
        return self.run_dir.name


def scan_runs(root: Path | None = None) -> list[RunRecord]:
    """All manifest-bearing run directories under ``outputs/`` (newest first).

    ``kind`` is inferred from the artifact set: a serve run writes ``screening_report_json``,
    a training run writes ``vespuq_report_json``; anything else (benchmarks, propagation,
    stage-1/2 runs) is ``other``.
    """

    base = root or OUTPUTS_DIR
    if not base.is_dir():
        return []
    records: list[RunRecord] = []
    for manifest_path in base.rglob("run_manifest.json"):
        manifest, error = safe_read_json(manifest_path)
        if error is not None or manifest is None:
            continue
        artifacts_raw = manifest.get("artifacts", {})
        metrics_raw = manifest.get("metrics", {})
        inputs_raw = manifest.get("inputs", {})
        artifacts = artifacts_raw if isinstance(artifacts_raw, dict) else {}
        metrics = metrics_raw if isinstance(metrics_raw, dict) else {}
        inputs = inputs_raw if isinstance(inputs_raw, dict) else {}
        if "screening_report_json" in artifacts:
            kind = "serve"
        elif "vespuq_report_json" in artifacts:
            kind = "train"
        else:
            kind = "other"
        records.append(
            RunRecord(
                run_dir=manifest_path.parent,
                manifest_path=manifest_path,
                created_at=str(manifest.get("created_at_utc", "")),
                kind=kind,
                metrics=metrics,
                artifacts=artifacts,
                inputs=inputs,
            )
        )
    records.sort(key=lambda r: r.created_at, reverse=True)
    return records
