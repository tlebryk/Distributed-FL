# zk_manager.py
from kazoo.client import KazooClient
import logging


class ZKManager:
    def __init__(self, hosts="127.0.0.1:2181"):
        # Initialize ZooKeeper client
        self.zk = None
        try:
            self.zk = KazooClient(hosts=hosts)
            self.zk.start()
        except Exception as e:
            logging.error(f"ZooKeeper connection error: {e}")

    def update_client_weight(self, client_id, weight):

        # 3. Ensure the parent path exists
        self.zk.ensure_path("/myapp/clients")

        path = f"/myapp/clients/{client_id}"
        data = str(weight).encode("utf-8")

        if self.zk.exists(path):
            self.zk.set(path, data)
        else:
            self.zk.create(path, data, makepath=True)

        print(f"Set {path} = {weight}")

    def get_client_weight(self, client_id):

        path = f"/myapp/clients/{client_id}"
        weight = 0.5
        if self.zk.exists(path):
            raw, stat = self.zk.get(path)
            weight = float(raw.decode("utf-8"))
            print(f"Weight for {client_id}: {weight}")
        else:
            print(f"No entry for {client_id}")
            self.update_client_weight(client_id, weight)

        return weight

    def update_past_accuracy(self, accuracy):
        path = f"/myapp/human_eval/accuracy"
        data = str(accuracy).encode("utf-8")

        if self.zk.exists(path):
            self.zk.set(path, data)
        else:
            self.zk.create(path, data, makepath=True)
        print(f"Set {path} = {accuracy}")

    def get_past_accuracy(self):
        path = f"/myapp/human_eval/accuracy"
        accuracy = 0
        if self.zk.exists(path):
            raw, stat = self.zk.get(path)
            accuracy = float(raw.decode("utf-8"))
            print(f"accuracy for: {accuracy}")
        else:
            print(f"No entry for human eval accuracy")
            self.update_past_accuracy(accuracy)
        return accuracy
