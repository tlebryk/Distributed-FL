import os
import json
import pytest
import tempfile
from pathlib import Path

import utils


def test_load_safetensors_from_bytes(monkeypatch):
    # patch the load_file function imported in utils
    called = {}

    def fake_load_file(path):
        called["path"] = path
        # verify that the file actually exists at call time
        assert os.path.exists(path)
        return {"fake_tensor": 123}

    monkeypatch.setattr(utils, "load_file", fake_load_file)

    raw = b"dummy data"
    out = utils.load_safetensors_from_bytes(raw)
    assert out == {"fake_tensor": 123}

    # the temporary file should have .safetensors suffix and then been deleted
    tmp_path = called["path"]
    assert tmp_path.endswith(".safetensors")
    assert not os.path.exists(tmp_path)


def test_find_latest_adapter_version_empty(tmp_path, caplog):
    # no central/latest, no v* directories
    caplog.set_level("INFO")
    got = utils.find_latest_adapter_version(str(tmp_path))
    assert got == 0
    assert "No adapter versions found" in caplog.text


def test_find_latest_adapter_version_scan(tmp_path):
    # create several version dirs
    for name in ("v1", "v10", "v3", "vinvalid"):
        (tmp_path / name).mkdir()
    # should pick 10
    assert utils.find_latest_adapter_version(str(tmp_path)) == 10


def test_find_latest_adapter_version_with_valid_symlink(tmp_path):
    # create version directory and metadata
    vdir = tmp_path / "v5"
    vdir.mkdir()
    metadata = {"version": 5}
    with open(vdir / "metadata.json", "w") as f:
        json.dump(metadata, f)

    # central/latest → v5
    central = tmp_path / "central"
    central.mkdir()
    (central / "latest").symlink_to(vdir, target_is_directory=True)

    assert utils.find_latest_adapter_version(str(tmp_path)) == 5


def test_find_latest_adapter_version_broken_symlink(tmp_path):
    # create some version dirs
    for name in ("v2", "v4"):
        (tmp_path / name).mkdir()

    # central/latest → nonexistent
    central = tmp_path / "central"
    central.mkdir()
    (central / "latest").symlink_to(
        tmp_path / "does_not_exist", target_is_directory=True
    )

    # falls back to scanning, picks 4
    assert utils.find_latest_adapter_version(str(tmp_path)) == 4


def test_find_latest_adapter_version_bad_metadata(tmp_path):
    # v7 with invalid JSON metadata
    v7 = tmp_path / "v7"
    v7.mkdir()
    with open(v7 / "metadata.json", "w") as f:
        f.write("not a json")

    # also v3 so scanning max is 7
    (tmp_path / "v3").mkdir()

    central = tmp_path / "central"
    central.mkdir()
    (central / "latest").symlink_to(v7, target_is_directory=True)

    # invalid JSON → exception → fallback → 7
    assert utils.find_latest_adapter_version(str(tmp_path)) == 7


def test_find_latest_adapter_version_listdir_error(tmp_path, monkeypatch, caplog):
    # force os.listdir to raise
    monkeypatch.setattr(
        utils.os, "listdir", lambda p: (_ for _ in ()).throw(Exception("oops"))
    )
    caplog.set_level("INFO")
    assert utils.find_latest_adapter_version(str(tmp_path)) == 0
    assert "Error finding latest adapter version" in caplog.text
