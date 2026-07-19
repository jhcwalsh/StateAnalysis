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
    seen = {}
    def fake_runner(cmd, **kwargs):
        seen["cmd"], seen["kwargs"] = cmd, kwargs
        return R(0)
    ok, tail = ui_io.run_refresh(str(tmp_path), runner=fake_runner)
    assert ok
    cmd = seen["cmd"]
    assert cmd[1:6] == ["-m", "jupyter", "nbconvert", "--to", "notebook"]
    assert "--execute" in cmd
    assert any(str(c).endswith(ui_io.NOTEBOOK) for c in cmd)
    out_i = cmd.index("--output-dir")
    assert cmd[out_i + 1].endswith("last_run")
    assert seen["kwargs"]["cwd"] == str(tmp_path)
    ok, tail = ui_io.run_refresh(str(tmp_path),
                                 runner=lambda *a, **k: R(1, b"boom\nlast line"))
    assert not ok and "last line" in tail
