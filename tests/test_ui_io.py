import json, os
import pytest
import ui_io


def test_load_bundle_ok(ui_data_dir):
    b = ui_io.load_bundle(ui_data_dir)
    assert b["meta"]["schema_version"] == 1
    assert set(b) >= {"meta", "gaps", "labels", "probs", "returns",
                      "backtest", "tables_path"}
    assert len(b["gaps"]) == 24
    assert os.path.exists(b["tables_path"])


def test_load_bundle_empty_state(tmp_path):
    assert ui_io.load_bundle(str(tmp_path / "nowhere")) is None


def test_load_bundle_schema_mismatch(ui_data_dir):
    meta_path = os.path.join(ui_data_dir, "meta.json")
    meta = json.load(open(meta_path)); meta["schema_version"] = 99
    json.dump(meta, open(meta_path, "w"))
    with pytest.raises(ui_io.SchemaError):
        ui_io.load_bundle(ui_data_dir)


def test_load_bundle_missing_file_is_schema_error(ui_data_dir):
    os.remove(os.path.join(ui_data_dir, "backtest.parquet"))
    with pytest.raises(ui_io.SchemaError):
        ui_io.load_bundle(ui_data_dir)


def test_run_refresh_success_and_failure(tmp_path):
    class R:
        def __init__(s, code, err=b""): s.returncode, s.stderr, s.stdout = code, err, b""
    ok, tail = ui_io.run_refresh(str(tmp_path), runner=lambda *a, **k: R(0))
    assert ok
    ok, tail = ui_io.run_refresh(str(tmp_path),
                                 runner=lambda *a, **k: R(1, b"boom\nlast line"))
    assert not ok and "last line" in tail
