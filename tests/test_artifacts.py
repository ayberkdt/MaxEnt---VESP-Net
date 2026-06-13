import json

import torch

from vesp.common.artifacts import (
    atomic_write_json,
    build_run_manifest,
    canonical_json_text,
    compute_file_sha256,
    write_manifest,
    write_run_manifest,
)


def test_atomic_write_json_and_manifest(tmp_path):
    payload_path = tmp_path / "payload.json"
    atomic_write_json(payload_path, {"b": 2, "a": 1})
    assert json.loads(payload_path.read_text(encoding="utf-8")) == {"a": 1, "b": 2}
    assert len(compute_file_sha256(payload_path)) == 64

    manifest_path = write_run_manifest(
        tmp_path,
        config={"run": "test"},
        metrics={"rmse": 0.1},
        artifacts={"payload": payload_path},
    )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["schema_version"] == "vesp_run_manifest_v1"
    assert manifest["artifacts"]["payload"]["sha256"] == compute_file_sha256(payload_path)
    assert manifest["artifacts"]["payload"]["path"] == str(payload_path)
    assert manifest["artifacts"]["payload"]["origin"] == "generated"
    assert manifest["inputs"] == {}  # additive key: present (empty) when no inputs given


def test_manifest_checksums_consumed_inputs(tmp_path):
    input_path = tmp_path / "dataset.csv"
    input_path.write_text("x,y\n1,2\n", encoding="utf-8")

    manifest_path = write_run_manifest(
        tmp_path,
        artifacts={},
        inputs={"dataset_csv": input_path, "gone": tmp_path / "missing.csv"},
    )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    entry = manifest["inputs"]["dataset_csv"]
    assert entry["sha256"] == compute_file_sha256(input_path)
    assert entry["bytes"] == input_path.stat().st_size
    assert entry["origin"] == "consumed"
    assert manifest["inputs"]["gone"]["missing"] is True


def test_shared_manifest_builder_supports_status_and_explicit_path(tmp_path):
    artifact = tmp_path / "plot.png"
    artifact.write_bytes(b"png")
    manifest = build_run_manifest(
        artifacts={"plot": artifact},
        artifact_statuses={"plot": "missing_data"},
        artifact_metadata={"plot": {"origin": "prewritten", "media_type": "image/png"}},
        manifest_metadata={"tool": "test"},
        created_at_utc="2026-06-13T00:00:00Z",
    )

    entry = manifest["artifacts"]["plot"]
    assert entry == {
        "path": str(artifact),
        "origin": "prewritten",
        "sha256": compute_file_sha256(artifact),
        "bytes": artifact.stat().st_size,
        "status": "missing_data",
        "media_type": "image/png",
    }
    assert manifest["tool"] == "test"
    assert manifest["created_at_utc"] == "2026-06-13T00:00:00Z"

    manifest_path = tmp_path / "custom_manifest.json"
    written = write_manifest(manifest_path, artifacts={"plot": artifact})
    assert json.loads(manifest_path.read_text(encoding="utf-8")) == written
    assert not (tmp_path / "checkpoints").exists()


def test_canonical_json_sorts_unordered_collections():
    first = canonical_json_text({"tags": {"beta", "alpha"}, "frozen": frozenset({3, 1, 2})})
    second = canonical_json_text({"frozen": frozenset({2, 3, 1}), "tags": {"alpha", "beta"}})

    assert first == second
    assert json.loads(first) == {"frozen": [1, 2, 3], "tags": ["alpha", "beta"]}


def test_canonical_json_converts_nonfinite_numbers_to_null():
    text = canonical_json_text(
        {
            "nan": float("nan"),
            "positive_infinity": float("inf"),
            "tensor": torch.tensor([1.0, float("-inf")]),
        }
    )

    assert "NaN" not in text
    assert "Infinity" not in text
    assert json.loads(text) == {
        "nan": None,
        "positive_infinity": None,
        "tensor": [1.0, None],
    }
