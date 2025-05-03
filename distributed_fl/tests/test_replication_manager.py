import os
import tempfile
import threading
import time
import socket
import grpc
import pytest
from unittest.mock import patch, MagicMock

import replica_manager
from replica_manager import ReplicaManager, normalize_address

# --- Helpers and Fakes ---


class FakeKazooClient:
    def __init__(self):
        self.paths = set()
        self.data_store = {}

    def ensure_path(self, path):
        self.paths.add(path)

    def exists(self, path):
        return path in self.data_store or path in self.paths

    def create(self, path, value=b"", makepath=False, ephemeral=False, sequence=False):
        self.data_store[path] = value
        return path

    def set(self, path, value):
        self.data_store[path] = value

    def get(self, path):
        return self.data_store[path], None

    def get_children(self, path):
        prefix = path.rstrip("/") + "/"
        return [p[len(prefix) :] for p in self.data_store if p.startswith(prefix)]


class FakeZKManager:
    def __init__(self, hosts=None):
        self.zk = FakeKazooClient()


class FakeChannel:
    def __init__(self):
        self.closed = False

    def close(self):
        self.closed = True


class VersionInfo:
    def __init__(self, latest_version=0, update_available=False, model_state=b""):
        self.latest_version = latest_version
        self.update_available = update_available
        self.model_state = model_state


class HeartbeatResponse:
    def __init__(self, alive=True, current_version=0):
        self.alive = alive
        self.current_version = current_version


class AggregatedModel:
    def __init__(self, model_state=b"", version=0):
        self.model_state = model_state
        self.version = version


class FakeStub:
    def __init__(self, channel):
        self.channel = channel
        self.connected = False

    def ConnectClient(self, request, timeout=None):
        self.connected = True
        # Return no update by default
        return VersionInfo(
            latest_version=request.current_version, update_available=False
        )

    def SendHeartbeat(self, request, timeout=None):
        return HeartbeatResponse(alive=True, current_version=request.timestamp)

    def GetAggregatedModel(self, request):
        return AggregatedModel(
            model_state=b"state", version=request.current_version + 1
        )


# Fixture to patch ZKManager and gRPC channel/stub
@pytest.fixture(autouse=True)
def patch_dependencies(monkeypatch):
    # Patch ZKManager
    monkeypatch.setattr(replica_manager, "ZKManager", FakeZKManager)
    # Patch grpc channel
    monkeypatch.setattr(
        replica_manager.grpc, "insecure_channel", lambda addr: FakeChannel()
    )
    # Patch stub creation
    monkeypatch.setattr(
        replica_manager.model_update_pb2_grpc,
        "FederatedLearningServiceStub",
        lambda channel: FakeStub(channel),
    )
    # Patch safetensors save and utils load
    monkeypatch.setattr(
        replica_manager,
        "save_file",
        lambda d, path: open(path, "wb").write(b"data"),
    )
    monkeypatch.setattr(replica_manager, "load_safetensors_from_bytes", lambda b: {})


# --- Tests for normalize_address ---


def test_normalize_address_empty():
    assert normalize_address("") == ""


@pytest.mark.parametrize("address", ["noport", "too:many:parts"])
def test_normalize_address_invalid_format(address):
    assert normalize_address(address) == address


def test_normalize_address_loopback(monkeypatch):
    monkeypatch.setattr(socket, "gethostname", lambda: "myhost")
    monkeypatch.setattr(socket, "gethostbyname", lambda h: "127.0.0.1")
    assert normalize_address("somehost:1234") == "localhost:1234"


def test_normalize_address_local_interface(monkeypatch):
    monkeypatch.setattr(socket, "gethostname", lambda: "myhost")
    # Simulate resolution to non-loopback and match interface
    monkeypatch.setattr(socket, "gethostbyname", lambda h: "192.168.0.5")
    monkeypatch.setattr(
        socket,
        "getaddrinfo",
        lambda h, *args: [(None, None, None, None, ("192.168.0.5", 0))],
    )
    assert normalize_address("otherhost:5678") == "localhost:5678"


def test_normalize_address_unresolvable(monkeypatch):
    monkeypatch.setattr(socket, "gethostname", lambda: "myhost")

    class GAIError(Exception):
        pass

    monkeypatch.setattr(
        socket, "gethostbyname", lambda h: (_ for _ in ()).throw(socket.gaierror())
    )
    # Should return original when resolution fails
    assert normalize_address("unknown:9999") == "unknown:9999"


# --- Tests for ReplicaManager initialization and registration ---


def test_register_replica(tmp_path):
    adapters_path = tmp_path
    mgr = ReplicaManager(
        replica_id="r1",
        leader_address="leader:111",
        server_port=111,
        adapters_path=str(adapters_path),
    )
    zk = mgr.zk_manager.zk
    # Ensure replicas path created and replica node registered
    assert "/myapp/replicas" in zk.paths
    assert f"/myapp/replicas/111" in zk.data_store
    assert zk.data_store[f"/myapp/replicas/111"] == b"r1"
    # Initial version should be 0
    assert mgr.current_version == 0


# --- Tests for leader election ---


def test_elect_leader(tmp_path):
    mgr = ReplicaManager(
        replica_id="r2",
        leader_address="leader:222",
        server_port=222,
        adapters_path=str(tmp_path),
    )
    zk = mgr.zk_manager.zk
    # Simulate two replicas registered
    zk.data_store["/myapp/replicas/222"] = b"r2"
    zk.data_store["/myapp/replicas/333"] = b"r3"
    # 222 is lower than 333
    assert mgr._elect_leader() is True
    # Now test non-leader
    mgr2 = ReplicaManager(
        replica_id="r3",
        leader_address="leader:333",
        server_port=333,
        adapters_path=str(tmp_path),
    )
    zk2 = mgr2.zk_manager.zk
    zk2.data_store["/myapp/replicas/222"] = b"r2"
    zk2.data_store["/myapp/replicas/333"] = b"r3"
    assert mgr2._elect_leader() is False


# --- Tests for leader registration ---


def test_register_as_leader(tmp_path):
    mgr = ReplicaManager(
        replica_id="r4",
        leader_address="leader:444",
        server_port=444,
        adapters_path=str(tmp_path),
    )
    result = mgr._register_as_leader()
    zk = mgr.zk_manager.zk
    assert result is True
    # Leader path should exist
    assert "/myapp/leader/current" in zk.data_store
    assert zk.data_store["/myapp/leader/current"] == b"localhost:444"


# --- Tests for model updates ---


def test_update_local_model(tmp_path):
    adapters_path = tmp_path
    # Create adapter config
    config_path = tmp_path / "adapter_config.json"
    config_path.write_text('{"config": true}')

    mgr = ReplicaManager(
        replica_id="r5",
        leader_address="leader:555",
        server_port=555,
        adapters_path=str(adapters_path),
    )
    # Update model
    success = mgr._update_local_model(b"somebytes", version=1)
    assert success is True
    # Check version updated
    assert mgr.current_version == 1
    version_dir = adapters_path / "server" / "central" / "v1"
    assert version_dir.exists()
    assert (version_dir / "adapter_model.safetensors").exists()
    assert (version_dir / "adapter_config.json").exists()


def test_get_latest_model_calls_update(tmp_path):
    adapters_path = tmp_path
    mgr = ReplicaManager(
        replica_id="r6",
        leader_address="leader:666",
        server_port=666,
        adapters_path=str(adapters_path),
    )
    # Patch stub.GetAggregatedModel
    called = {"update": False}

    def fake_update(model_state, version):
        called["update"] = True
        return True

    mgr._update_local_model = fake_update
    assert mgr._get_latest_model() is True
    assert called["update"] is True


# --- Tests for connection to leader ---


def test_connect_to_leader_success(tmp_path):
    mgr = ReplicaManager(
        replica_id="r7",
        leader_address="leader:777",
        server_port=777,
        adapters_path=str(tmp_path),
    )
    assert mgr.connect_to_leader() is True


@patch.object(FakeStub, "ConnectClient", side_effect=Exception("fail"))
def test_connect_to_leader_failure(mock_connect, tmp_path):
    mgr = ReplicaManager(
        replica_id="r8",
        leader_address="leader:888",
        server_port=888,
        adapters_path=str(tmp_path),
    )
    assert mgr.connect_to_leader() is False


# --- Tests for heartbeat ---


def test_check_leader_heartbeat_success(tmp_path):
    mgr = ReplicaManager(
        replica_id="r9",
        leader_address="leader:999",
        server_port=999,
        adapters_path=str(tmp_path),
    )
    # Set current_version low
    mgr.current_version = 0
    called = {"got": False}

    def fake_get():
        called["got"] = True
        return True

    mgr._get_latest_model = fake_get

    # Simulate heartbeat with higher version
    class HBR(HeartbeatResponse):
        pass

    mgr.stub.SendHeartbeat = lambda req, timeout=None: HBR(
        alive=True, current_version=5
    )
    result = mgr.check_leader_heartbeat()
    assert result is True
    assert mgr.missed_heartbeats == 0
    assert mgr.leader_is_alive is True
    assert called["got"] is True


def test_check_leader_heartbeat_missed(tmp_path):
    mgr = ReplicaManager(
        replica_id="r10",
        leader_address="leader:1010",
        server_port=1010,
        adapters_path=str(tmp_path),
    )
    # Simulate failures
    mgr.stub.SendHeartbeat = lambda req, timeout=None: (_ for _ in ()).throw(
        Exception("down")
    )
    for i in range(mgr.max_missed_heartbeats):
        res = mgr.check_leader_heartbeat()
        assert res is False
    # After threshold
    assert mgr.missed_heartbeats == mgr.max_missed_heartbeats
    assert mgr.leader_is_alive is False


# --- Tests for leader discovery ---


def test_discover_leader_changes(tmp_path, monkeypatch):
    old_addr = "old:1"
    new_addr = "new:2"
    mgr = ReplicaManager(
        replica_id="r11",
        leader_address=old_addr,
        server_port=1111,
        adapters_path=str(tmp_path),
    )
    zk = mgr.zk_manager.zk
    # Write new leader in ZK
    zk.ensure_path("/myapp/leader/current")
    zk.data_store["/myapp/leader/current"] = new_addr.encode()
    # Patch normalize_address
    monkeypatch.setattr(replica_manager, "normalize_address", lambda x: x)
    # Patch channel and stub
    test_channel = FakeChannel()
    created = {"new_stub": None}

    def new_channel(addr):
        return test_channel

    def make_stub(ch):
        created["new_stub"] = FakeStub(ch)
        return created["new_stub"]

    monkeypatch.setattr(replica_manager.grpc, "insecure_channel", new_channel)
    monkeypatch.setattr(
        replica_manager.model_update_pb2_grpc,
        "FederatedLearningServiceStub",
        make_stub,
    )
    # Call discover
    result = mgr._discover_leader()
    assert result is True
    # Channel should be updated
    assert mgr.leader_address == new_addr
    # Old channel closed
    assert isinstance(mgr.channel, FakeChannel)


# --- Tests for shutdown ---


def test_shutdown(tmp_path):
    mgr = ReplicaManager(
        replica_id="r12",
        leader_address="leader:1212",
        server_port=1212,
        adapters_path=str(tmp_path),
    )
    # Assign fake channel
    ch = FakeChannel()
    mgr.channel = ch
    mgr.running = True
    mgr.shutdown()
    assert mgr.running is False
    assert ch.closed is True
