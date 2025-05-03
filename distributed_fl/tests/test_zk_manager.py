import pytest
from unittest.mock import patch
import uuid

import threading

# Assuming the class is defined in zk_manager.py
from zk_manager import ZKManager


# Fake ZooKeeper client for testing
class FakeKazooClient:
    def __init__(self, hosts=None):
        self.hosts = hosts
        self.started = False
        self.listeners = []
        self.paths = set()
        self.data_store = {}
        self.next_seq = 1
        self.data_watch_callbacks = {}

    def start(self):
        self.started = True

    def add_listener(self, listener):
        self.listeners.append(listener)

    def ensure_path(self, path):
        # Just record the path
        self.paths.add(path)

    def exists(self, path):
        return path in self.data_store or path in self.paths

    def create(self, path, value=b"", makepath=False, ephemeral=False, sequence=False):
        # Handle election sequential nodes
        if sequence:
            seq = f"{self.next_seq:010d}"
            self.next_seq += 1
            full_path = f"{path}{seq}"
        else:
            full_path = path
        # Simulate path creation
        self.data_store[full_path] = value
        # Ensure parent path
        parent = "/".join(full_path.rstrip("/").split("/")[:-1]) or "/"
        self.paths.add(parent)
        return full_path

    def set(self, path, data):
        if path not in self.data_store:
            raise Exception("Path does not exist")
        self.data_store[path] = data

    def get(self, path):
        if path not in self.data_store:
            raise Exception("Path does not exist")
        return self.data_store[path], None

    def get_children(self, path):
        # Return node names under the given path
        prefix = path.rstrip("/") + "/"
        children = []
        for full in self.data_store:
            if full.startswith(prefix):
                name = full[len(prefix) :]
                if "/" not in name:
                    children.append(name)
        return children

    def DataWatch(self, path):
        # Return a decorator to register a watch callback
        def decorator(func):
            self.data_watch_callbacks[path] = func
            return func

        return decorator

    def stop(self):
        self.started = False

    def close(self):
        pass


# Fixture to patch the KazooClient used in ZKManager
@pytest.fixture(autouse=True)
def patch_kazoo(monkeypatch):
    monkeypatch.setattr("zk_manager.KazooClient", FakeKazooClient)
    yield


def test_update_and_get_client_weight_default(tmp_path):
    mgr = ZKManager()
    client_id = "client1"
    # Initially no weight, get should create default and return 0.5
    weight = mgr.get_client_weight(client_id)
    assert weight == 0.5
    # Now manually update weight
    assert mgr.update_client_weight(client_id, 2.5)
    # Get updated weight
    weight2 = mgr.get_client_weight(client_id)
    assert weight2 == pytest.approx(2.5)


def test_update_and_get_past_accuracy(tmp_path):
    mgr = ZKManager()
    # Initially no accuracy, get should create default 0 and return 0
    acc = mgr.get_past_accuracy()
    assert acc == 0
    # Update accuracy
    assert mgr.update_past_accuracy(0.9)
    acc2 = mgr.get_past_accuracy()
    assert acc2 == pytest.approx(0.9)


def test_leader_data_methods():
    mgr = ZKManager()
    # Not leader yet, create_leader_data should fail
    assert not mgr.create_leader_data({"foo": "bar"})
    # Manually set as leader
    mgr.is_leader = True
    # Create data
    assert mgr.create_leader_data({"foo": "bar"})
    data = mgr.get_leader_data()
    assert data == {"foo": "bar"}


def test_stop_client():
    mgr = ZKManager()
    assert mgr.zk is not None
    mgr.stop(timeout=1)
    assert mgr.zk is None
