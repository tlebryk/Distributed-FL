import os
import time
import socket
import tempfile
from types import SimpleNamespace

import pytest

import server
from server import (
    normalize_address,
    cleanup_disconnected_clients,
    register_leader_in_zk,
    FederatedLearningServiceServicer,
    DecodedModelUpdate,
)
import model_update_pb2


def test_normalize_address_empty_or_invalid():
    # empty string
    assert normalize_address("") == ""
    # not in host:port form
    assert normalize_address("justastring") == "justastring"
    # wrong format with multiple colons
    assert normalize_address("a:b:c") == "a:b:c"


def test_normalize_address_local_hostname(monkeypatch):
    # pretend our hostname is "myhost"
    monkeypatch.setattr(socket, "gethostname", lambda: "myhost")
    # if gethostbyname returns loopback, should map to localhost
    monkeypatch.setattr(socket, "gethostbyname", lambda name: "127.0.0.1")
    addr = normalize_address("myhost:9999")
    assert addr == "localhost:9999"


def test_normalize_address_resolve_to_local_ip(monkeypatch):
    # machine hostname is "foo"
    monkeypatch.setattr(socket, "gethostname", lambda: "foo")
    # remote lookup returns a non-loopback IP that matches one of our addresses
    monkeypatch.setattr(socket, "gethostbyname", lambda name: "192.168.1.5")
    # pretend getaddrinfo for "foo" returns that IP among others
    fake_info = [(None, None, None, None, ("192.168.1.5", 0))]
    monkeypatch.setattr(socket, "getaddrinfo", lambda hostname, _: fake_info)
    addr = normalize_address("foo:1234")
    assert addr == "localhost:1234"


def test_cleanup_disconnected_clients():
    servicer = SimpleNamespace(
        connected_clients={},
        notification_queues={},
    )
    # two clients, one just seen, one stale
    now = time.time()
    servicer.connected_clients["active"] = {"last_seen": now}
    servicer.notification_queues["active"] = "queue1"
    servicer.connected_clients["stale"] = {"last_seen": now - 301}
    servicer.notification_queues["stale"] = "queue2"

    # run cleanup
    cleanup_disconnected_clients(servicer)
    # stale should be removed
    assert "stale" not in servicer.connected_clients
    assert "stale" not in servicer.notification_queues
    # active remains
    assert "active" in servicer.connected_clients
    assert "active" in servicer.notification_queues


class DummyZK:
    def __init__(self):
        self._nodes = {}

    def ensure_path(self, path):
        # no-op
        pass

    def exists(self, path):
        return path in self._nodes

    def create(self, path, data):
        self._nodes[path] = data

    def set(self, path, data):
        self._nodes[path] = data


def test_register_leader_in_zk_first_and_update(monkeypatch):
    dummy = SimpleNamespace(zk=DummyZK())
    # first time, no node exists
    ok = register_leader_in_zk(dummy, port=5555)
    assert ok is True
    assert dummy.zk._nodes["/myapp/leader/current"] == b"localhost:5555"

    # change port and register again should update existing node
    ok2 = register_leader_in_zk(dummy, port=7777)
    assert ok2 is True
    assert dummy.zk._nodes["/myapp/leader/current"] == b"localhost:7777"


def make_request(
    update_bytes=b"\x00\x01",
    client_id="cid",
    version=0,
    timestamp=None,
    pylint_score=0.5,
):
    if timestamp is None:
        timestamp = int(time.time())
    return SimpleNamespace(
        client_id=client_id,
        version=version,
        update=update_bytes,
        timestamp=timestamp,
        pylint_score=pylint_score,
    )


def test_submit_update_success(monkeypatch):
    # create a servicer in 'test' mode to skip heavy evaluation
    servicer = FederatedLearningServiceServicer(mode="test")
    # stub out load_safetensors_from_bytes to return an empty dict
    monkeypatch.setattr(server, "load_safetensors_from_bytes", lambda raw: {})
    # stub ZKManager so zk is None → weight=0.5
    servicer.zk_manager.zk = None

    req = make_request()
    # call with context=None (unused on success)
    ack = servicer.SubmitUpdate(req, context=None)
    assert isinstance(ack, model_update_pb2.Acknowledgement)
    assert ack.success
    assert "received" in ack.message.lower()
    # request should be queued
    assert len(servicer.update_requests) == 1
    dm = servicer.update_requests[0]
    assert isinstance(dm, DecodedModelUpdate)
    assert dm.client_id == "cid"
    assert dm.update == {}
    assert dm.weight == 0.5


def test_submit_update_failure(monkeypatch):
    servicer = FederatedLearningServiceServicer(mode="test")

    # cause the load to raise
    def bad_load(raw):
        raise RuntimeError("broken")

    monkeypatch.setattr(server, "load_safetensors_from_bytes", bad_load)
    req = make_request()
    ack = servicer.SubmitUpdate(req, context=None)
    assert not ack.success
    assert "broken" in ack.message


def test_get_aggregated_model_no_state():
    servicer = FederatedLearningServiceServicer(mode="test")
    # no global_adapter_state by default
    req = SimpleNamespace(client_id="any")
    agg = servicer.GetAggregatedModel(req, context=None)
    assert agg.version == servicer.version == 0
    assert agg.model_state == b""


def test_get_aggregated_model_with_state(tmp_path, monkeypatch):
    # point adapters path at tmp_path
    monkeypatch.setattr(server, "PATH_TO_ADAPTERS", str(tmp_path))
    servicer = FederatedLearningServiceServicer(mode="test")
    servicer.version = 1
    servicer.global_adapter_state = {"foo": 123}

    # write a dummy safetensors file
    model_dir = tmp_path / "server" / "central" / "v1"
    model_dir.mkdir(parents=True)
    data = b"hello-model"
    with open(model_dir / "adapter_model.safetensors", "wb") as f:
        f.write(data)

    req = SimpleNamespace(client_id="c")
    agg = servicer.GetAggregatedModel(req, context=None)
    assert agg.version == 1
    assert agg.model_state == data


def test_connect_client_no_update():
    servicer = FederatedLearningServiceServicer(mode="test")
    servicer.version = 0
    servicer.global_adapter_state = None

    req = SimpleNamespace(client_id="cli", current_version=0, ready_for_training=True)
    resp = servicer.ConnectClient(req, context=None)
    assert not resp.update_available
    assert resp.latest_version == 0
    assert resp.model_state == b""


def test_connect_client_with_update(tmp_path, monkeypatch):
    monkeypatch.setattr(server, "PATH_TO_ADAPTERS", str(tmp_path))
    servicer = FederatedLearningServiceServicer(mode="test")
    servicer.version = 2
    servicer.global_adapter_state = {"x": 1}

    # create v2 model file
    central_v2 = tmp_path / "server" / "central" / "v2"
    central_v2.mkdir(parents=True)
    data = b"model-bytes"
    with open(central_v2 / "adapter_model.safetensors", "wb") as f:
        f.write(data)

    req = SimpleNamespace(client_id="cli", current_version=1, ready_for_training=False)
    resp = servicer.ConnectClient(req, context=None)
    assert resp.update_available
    assert resp.latest_version == 2
    assert resp.model_state == data
    # ensure the client was tracked
    assert "cli" in servicer.connected_clients
    assert servicer.connected_clients["cli"]["version"] == 1


def test_notify_and_subscribe(monkeypatch):
    servicer = FederatedLearningServiceServicer(mode="test")
    servicer.version = 42
    servicer.global_adapter_state = {"k": "v"}

    req = SimpleNamespace(client_id="sub1", current_version=0)

    class DummyCtx:
        def __init__(self):
            self._active = False

        def is_active(self):
            return self._active

    ctx = DummyCtx()
    gen = servicer.SubscribeToUpdates(req, ctx)

    # initial notification
    note = next(gen)
    assert note.new_version == 42
    assert note.update_type == "FULL"

    # queue should now exist but be empty
    q = servicer.notification_queues["sub1"]
    assert q.qsize() == 0

    # trigger a notification
    servicer._notify_clients_of_update()
    assert q.qsize() == 1
    note2 = q.get_nowait()
    assert note2.new_version == 42
