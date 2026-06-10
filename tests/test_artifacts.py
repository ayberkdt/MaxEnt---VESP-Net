import json

from vesp.common.artifacts import atomic_write_json, compute_file_sha256, write_run_manifest


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
    assert manifest["inputs"]["gone"]["missing"] is True
