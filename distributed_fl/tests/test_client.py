import io
import os
import shutil
import tempfile

import pytest
import torch

import client
from client import FederatedClient, serialize_state_dict


def test_serialize_state_dict_roundtrip():
    # Create a dummy state dict
    orig = {"a": torch.tensor([1, 2, 3]), "b": torch.tensor(5)}
    data = serialize_state_dict(orig)
    # Load it back
    buffer = io.BytesIO(data)
    loaded = torch.load(buffer)
    # Tensors equal
    assert set(loaded.keys()) == set(orig.keys())
    for k in orig:
        assert torch.equal(loaded[k], orig[k])


@pytest.mark.parametrize(
    "path, expected",
    [
        ("/no/versions/here", 0),
        ("v1", 1),
        ("/foo/v2/bar", 2),
        ("foo/v10/v3", 3),  # last wins
        ("foo/v42_extra", 0),  # fullmatch only
        ("foo/v007", 7),
    ],
)
def test_get_version(path, expected):
    assert FederatedClient.get_version(path) == expected


def test__get_latest_version_empty(tmp_path):
    # no directory: returns 0
    assert (
        FederatedClient("id")._get_latest_version(str(tmp_path / "does_not_exist")) == 0
    )
    # empty dir
    d = tmp_path / "empty"
    d.mkdir()
    assert FederatedClient("id")._get_latest_version(str(d)) == 0


def test__get_latest_version_with_versions(tmp_path):
    d = tmp_path / "versions"
    d.mkdir()
    # create some entries
    for name in ["v1", "v2", "v10", "notav", "v3_extra"]:
        (d / name).mkdir(exist_ok=True)
    # only full vN directories count
    assert FederatedClient("id")._get_latest_version(str(d)) == 10


@pytest.mark.parametrize(
    "dirname, expected_round",
    [
        ("adapter_foo_r42", 42),
        ("adapter_bar_r7", 7),
        ("adapter__r100", 100),
        ("notadapter_foo_r5", 0),
        ("adapter_foo_no_r", 0),
        ("something_else", 0),
    ],
)
def test__get_round_from_path(dirname, expected_round):
    client = FederatedClient("id")
    # monkey‐patch out gRPC init to avoid side effects
    client.initialize_connection = lambda: None
    assert client._get_round_from_path(dirname) == expected_round


def test_update_server_address(monkeypatch):
    # stub out initialize_connection so no real gRPC
    monkeypatch.setattr(
        FederatedClient,
        "initialize_connection",
        lambda self: setattr(self, "_init_called", True),
    )
    c = FederatedClient(client_id="id", server_address="host:1", fallback_addresses=[])
    # initial init called
    assert hasattr(c, "_init_called")
    delattr(c, "_init_called")

    # updating to a new address
    result = c.update_server_address("host:2")
    assert result is True
    assert c.server_address == "host:2"
    assert getattr(c, "_init_called", False)

    # updating to same address -> no change
    delattr(c, "_init_called")
    result_same = c.update_server_address("host:2")
    assert result_same is False
    assert not hasattr(c, "_init_called")

    # empty or None
    assert c.update_server_address("") is False
    assert c.update_server_address(None) is False


import sys
import time
import os
import shutil
from types import SimpleNamespace

import pytest
import torch

import client
from client import FederatedClient, parse_args, serialize_state_dict
import model_update_pb2


class FakeHeartbeatResponse:
    def __init__(self, alive):
        self.alive = alive


class FakeStub:
    def __init__(
        self,
        hb_responses=None,
        connect_response=None,
        agg_response=None,
        submit_response=None,
    ):
        # lists or single objects
        self._hbs = hb_responses or []
        self._connect = connect_response
        self._agg = agg_response
        self._submit = submit_response
        self._hb_calls = 0

    def SendHeartbeat(self, *args, **kwargs):
        if self._hb_calls >= len(self._hbs):
            raise RuntimeError("No more fake heartbeats")
        resp = self._hbs[self._hb_calls]
        self._hb_calls += 1
        if isinstance(resp, Exception):
            raise resp
        return resp

    def ConnectClient(self, *args, **kwargs):
        return self._connect

    def GetAggregatedModel(self, *args, **kwargs):
        return self._agg

    def SubmitUpdate(self, *args, **kwargs):
        return self._submit


@pytest.fixture(autouse=True)
def no_sleep(monkeypatch):
    # speed up retries
    monkeypatch.setattr(time, "sleep", lambda s: None)


def make_client(monkeypatch):
    # prevent real gRPC and agent init
    monkeypatch.setattr(FederatedClient, "initialize_connection", lambda self: None)
    fake_agent = SimpleNamespace(
        model=SimpleNamespace(load_state_dict=lambda *a, **k: None), tokenizer=None
    )
    monkeypatch.setattr(
        FederatedClient,
        "initialize_agent",
        lambda self: setattr(self, "agent", fake_agent),
    )
    return FederatedClient("test_id")


def test_try_fallback_servers_no_fallback(monkeypatch):
    c = make_client(monkeypatch)
    c.fallback_addresses = []
    assert not c.try_fallback_servers()


def test_try_fallback_servers_success(monkeypatch):
    c = make_client(monkeypatch)
    # two fallbacks, first same as original is skipped
    c.server_address = "A"
    c.fallback_addresses = ["A", "B", "C"]
    # fake stub returns alive on first try
    c.stub = FakeStub(hb_responses=[FakeHeartbeatResponse(True)])
    # record updates
    updated = []

    def fake_update(addr):
        updated.append(addr)
        c.server_address = addr
        return True

    c.update_server_address = fake_update
    assert c.try_fallback_servers() is True
    assert updated == ["B"]
    assert c.server_address == "B"


def test_try_fallback_servers_all_fail(monkeypatch):
    c = make_client(monkeypatch)
    c.server_address = "X"
    c.fallback_addresses = ["X", "Y"]
    # both fallback heartbeats raise
    c.stub = FakeStub(hb_responses=[RuntimeError(), RuntimeError()])
    # track restore
    calls = []

    def fake_update(addr):
        calls.append(addr)
        c.server_address = addr
        return True

    c.update_server_address = fake_update
    res = c.try_fallback_servers()
    assert not res
    # first update to Y, then restore to original X
    assert calls == ["Y", "X"]
    assert c.server_address == "X"


def test_check_connection_success(monkeypatch):
    c = make_client(monkeypatch)
    c.stub = FakeStub(hb_responses=[FakeHeartbeatResponse(True)])
    assert c.check_connection() is True


def test_check_connection_fallback(monkeypatch):
    c = make_client(monkeypatch)
    # heartbeat raises, but fallback returns True immediately
    c.stub = FakeStub(hb_responses=[RuntimeError()])
    monkeypatch.setattr(c, "try_fallback_servers", lambda: True)
    assert c.check_connection() is True


def test_check_connection_failure(monkeypatch):
    c = make_client(monkeypatch)
    c.stub = FakeStub(hb_responses=[RuntimeError(), RuntimeError(), RuntimeError()])
    monkeypatch.setattr(c, "try_fallback_servers", lambda: False)
    assert c.check_connection(max_retries=2) is False


def test_connect_to_server_no_update(monkeypatch):
    c = make_client(monkeypatch)
    # stub ConnectClient returns no update
    fake_version = SimpleNamespace(
        update_available=False, latest_version=9, model_state=b""
    )
    c.stub = FakeStub(connect_response=fake_version)
    # pretend connection always OK
    monkeypatch.setattr(c, "check_connection", lambda *a, **k: True)
    assert c.connect_to_server() is True
    assert c.current_version == 0  # unchanged


def test_get_latest_model_no_state(monkeypatch):
    c = make_client(monkeypatch)
    # stub GetAggregatedModel returns no state
    fake_agg = SimpleNamespace(model_state=b"", version=2)
    c.stub = FakeStub(agg_response=fake_agg)
    assert c.get_latest_model() is False


def test_get_latest_model_with_state(monkeypatch, tmp_path):
    c = make_client(monkeypatch)
    fake_bytes = b"abc"
    fake_agg = SimpleNamespace(model_state=fake_bytes, version=7)
    c.stub = FakeStub(agg_response=fake_agg)
    # stub loads and saves
    monkeypatch.setattr(
        client, "load_safetensors_from_bytes", lambda b: {"x": torch.tensor([1])}
    )
    c.agent.model.load_state_dict = lambda st, strict: None
    # intercept save_adapter_to_disk
    monkeypatch.setattr(c, "save_adapter_to_disk", lambda d, v: d)
    monkeypatch.setattr(shutil, "copy2", lambda *a, **k: None)
    assert c.get_latest_model() is True
    assert c.current_version == 7


def test_train_and_submit_success(monkeypatch):
    c = make_client(monkeypatch)
    # stub SubmitUpdate success
    fake_ack = SimpleNamespace(success=True, message="ok")
    c.stub = FakeStub(submit_response=fake_ack)
    # stub internals
    monkeypatch.setattr(c, "check_connection", lambda: True)
    monkeypatch.setattr(c, "connect_to_server", lambda *a, **k: True)
    monkeypatch.setattr(
        c, "get_adapter_update", lambda agent, code_path: (b"upd", 4.56)
    )
    assert c.train_and_submit() is True


def test_train_and_submit_outdated(monkeypatch):
    c = make_client(monkeypatch)
    # stub SubmitUpdate failure due to outdated
    fake_ack = SimpleNamespace(success=False, message="outdated model, please refresh")
    c.stub = FakeStub(submit_response=fake_ack)
    monkeypatch.setattr(c, "check_connection", lambda: True)
    monkeypatch.setattr(c, "connect_to_server", lambda *a, **k: True)
    monkeypatch.setattr(c, "get_adapter_update", lambda agent, code_path: (b"", 0.0))
    called = {"got": False}
    monkeypatch.setattr(
        c, "get_latest_model", lambda *a, **k: called.__setitem__("got", True)
    )
    res = c.train_and_submit()
    assert res is False
    assert called["got"]


def test_shutdown_closes_channel(monkeypatch):
    c = make_client(monkeypatch)
    closed = {"flag": False}
    c.channel = SimpleNamespace(close=lambda: closed.__setitem__("flag", True))
    c.running = True
    c.shutdown()
    assert not c.running
    assert closed["flag"]


def test_parse_args_defaults(tmp_path, monkeypatch):
    monkeypatch.setenv("PATH_TO_ADAPTERS", str(tmp_path))
    monkeypatch.setattr(sys, "argv", ["prog"])
    args = parse_args()
    assert args.client_id == "client_1"
    assert args.server_address == "localhost:50051"
    assert args.fallback_addresses is None
    assert args.interval == 15
    assert args.code_path == "./data"


def test_parse_args_custom(tmp_path, monkeypatch):
    monkeypatch.setenv("PATH_TO_ADAPTERS", str(tmp_path))
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "prog",
            "--client_id",
            "foo",
            "--server_address",
            "host:1234",
            "--fallback_addresses",
            "a:1",
            "b:2",
            "--interval",
            "42",
            "--code_path",
            "/tmp/code",
        ],
    )
    args = parse_args()
    assert args.client_id == "foo"
    assert args.server_address == "host:1234"
    assert args.fallback_addresses == ["a:1", "b:2"]
    assert args.interval == 42
    assert args.code_path == "/tmp/code"
