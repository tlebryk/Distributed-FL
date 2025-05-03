from kazoo.client import KazooClient, KazooState
from kazoo.exceptions import NodeExistsError, NoNodeError
import time
import logging
import threading
import uuid


class ZKManager:
    def __init__(self, hosts="127.0.0.1:2181", election_path="/myapp/election"):
        # Initialize ZooKeeper client
        self.zk = None
        self.hosts = hosts
        self.election_path = election_path
        self.node_id = None  # Election node ID
        self.election_path_full = None  # Full path of this instance's election node
        self.is_leader = False  # Flag indicating if this instance is the leader
        self.leader_id = None  # Current leader's ID
        self.leader_watch_event = threading.Event()  # Event to signal leader changes
        self.leadership_callbacks = []  # Callbacks for leadership changes

        # Initialize connection
        self._initialize_connection()

    def _initialize_connection(self):
        """Initialize the ZooKeeper connection."""
        try:
            logging.info(f"Connecting to ZooKeeper at {self.hosts}")
            self.zk = KazooClient(hosts=self.hosts)
            self.zk.start()

            # Add listener for connection state changes
            self.zk.add_listener(self._connection_listener)

            # Ensure election path exists
            self.zk.ensure_path(self.election_path)

            logging.info("Connected to ZooKeeper successfully")
        except Exception as e:
            logging.error(f"ZooKeeper connection error: {e}")
            self.zk = None

    def _connection_listener(self, state):
        """Handle ZooKeeper connection state changes."""
        if state == KazooState.LOST:
            # Connection is lost
            logging.warning("ZooKeeper connection lost")
            self.is_leader = False
        elif state == KazooState.SUSPENDED:
            # Connection is suspended
            logging.warning("ZooKeeper connection suspended")
        elif state == KazooState.CONNECTED:
            # Connection (re)established
            logging.info("ZooKeeper connection (re)established")
            # Re-register in the election if previously registered
            if self.node_id:
                self._rejoin_election()

    def _rejoin_election(self):
        """Rejoin the leader election after reconnection."""
        try:
            if self.election_path_full and self.zk.exists(self.election_path_full):
                # Our node still exists, we're still in the election
                logging.info(f"Election node {self.election_path_full} still exists")
                # Check if we are now the leader
                self._check_leadership()
            else:
                # Our node disappeared, need to rejoin election
                logging.info("Rejoining leader election")
                self.node_id = None
                self.election_path_full = None
                self.participate_election()
        except Exception as e:
            logging.error(f"Error rejoining election: {e}")

    def update_client_weight(self, client_id, weight):
        """Update a client's weight in ZooKeeper."""
        if not self.zk:
            logging.error("ZooKeeper not connected")
            return False

        # Ensure the parent path exists
        self.zk.ensure_path("/myapp/clients")

        path = f"/myapp/clients/{client_id}"
        data = str(weight).encode("utf-8")

        if self.zk.exists(path):
            self.zk.set(path, data)
        else:
            self.zk.create(path, data, makepath=True)

        logging.info(f"Set {path} = {weight}")
        return True

    def get_client_weight(self, client_id):
        """Get a client's weight from ZooKeeper."""
        if not self.zk:
            logging.error("ZooKeeper not connected")
            return 0.5

        path = f"/myapp/clients/{client_id}"
        weight = 0.5
        if self.zk.exists(path):
            raw, stat = self.zk.get(path)
            weight = float(raw.decode("utf-8"))
            logging.info(f"Weight for {client_id}: {weight}")
        else:
            logging.info(f"No entry for {client_id}")
            self.update_client_weight(client_id, weight)

        return weight

    def update_past_accuracy(self, accuracy):
        """Update the past accuracy value in ZooKeeper."""
        if not self.zk:
            logging.error("ZooKeeper not connected")
            return False

        path = f"/myapp/human_eval/accuracy"
        data = str(accuracy).encode("utf-8")

        if self.zk.exists(path):
            self.zk.set(path, data)
        else:
            self.zk.create(path, data, makepath=True)
        logging.info(f"Set {path} = {accuracy}")
        return True

    def get_past_accuracy(self):
        """Get the past accuracy value from ZooKeeper."""
        if not self.zk:
            logging.error("ZooKeeper not connected")
            return 0

        path = f"/myapp/human_eval/accuracy"
        accuracy = 0
        if self.zk.exists(path):
            raw, stat = self.zk.get(path)
            accuracy = float(raw.decode("utf-8"))
            logging.info(f"accuracy for: {accuracy}")
        else:
            logging.info(f"No entry for human eval accuracy")
            self.update_past_accuracy(accuracy)
        return accuracy

    # --- Leader Election Methods ---

    def participate_election(self, server_id=None):
        """
        Participate in leader election by creating an ephemeral sequential node.

        Args:
            server_id: Unique identifier for this server instance

        Returns:
            bool: True if successfully joined election, False otherwise
        """
        if not self.zk:
            logging.error("Cannot participate in election: ZooKeeper not connected")
            return False

        try:
            # Generate a unique ID if not provided
            if not server_id:
                server_id = f"server-{uuid.uuid4()}"

            # Create sequential ephemeral node
            self.node_id = server_id
            node_path = f"{self.election_path}/candidate-"
            self.election_path_full = self.zk.create(
                node_path,
                value=server_id.encode("utf-8"),
                ephemeral=True,
                sequence=True,
            )

            logging.info(f"Created election node: {self.election_path_full}")

            # Check if we are the leader
            self._check_leadership()

            # Watch for changes in leader
            self._watch_predecessor()

            return True

        except Exception as e:
            logging.error(f"Error participating in election: {e}")
            return False

    def _get_sorted_candidates(self):
        """
        Get a sorted list of all candidates in the election.

        Returns:
            list: Sorted list of candidate node names
        """
        if not self.zk:
            return []

        try:
            children = self.zk.get_children(self.election_path)
            # Filter for only candidate nodes
            candidates = [c for c in children if c.startswith("candidate-")]
            # Sort them (ZooKeeper sequence nodes are lexicographically sortable)
            candidates.sort()
            return candidates
        except Exception as e:
            logging.error(f"Error getting candidates: {e}")
            return []

    def _check_leadership(self):
        """Check if this node is now the leader based on ZooKeeper node ordering."""
        if not self.zk or not self.election_path_full:
            return False

        try:
            # Get sorted candidates
            candidates = self._get_sorted_candidates()
            if not candidates:
                logging.warning("No candidates found in election")
                return False

            # Get our sequence number
            my_node = self.election_path_full.split("/")[-1]

            # Leader is the lowest sequence number
            leader_node = candidates[0]

            # If we are the leader and weren't before, notify
            was_leader = self.is_leader
            self.is_leader = my_node == leader_node

            if self.is_leader:
                leader_path = f"{self.election_path}/{leader_node}"
                leader_data, _ = self.zk.get(leader_path)
                self.leader_id = leader_data.decode("utf-8")

                if not was_leader:
                    logging.info(f"This node is now the leader: {self.leader_id}")
                    # Trigger leadership callbacks
                    self._trigger_leadership_callbacks(True)
            else:
                # If we're not the leader, get the leader's ID
                leader_path = f"{self.election_path}/{leader_node}"
                leader_data, _ = self.zk.get(leader_path)
                self.leader_id = leader_data.decode("utf-8")

                if was_leader:
                    logging.info(
                        f"This node is no longer the leader. New leader: {self.leader_id}"
                    )
                    # Trigger leadership callbacks
                    self._trigger_leadership_callbacks(False)
                else:
                    logging.info(f"Current leader is: {self.leader_id}")

            return self.is_leader

        except Exception as e:
            logging.error(f"Error checking leadership: {e}")
            return False

    def _watch_predecessor(self):
        """Watch the predecessor node for changes."""
        if not self.zk or not self.election_path_full:
            return

        try:
            # Get sorted candidates
            candidates = self._get_sorted_candidates()
            if not candidates:
                return

            # Get our sequence number
            my_node = self.election_path_full.split("/")[-1]

            # Find our position
            my_index = candidates.index(my_node)

            # If we're the leader (index 0), no need to watch anyone
            if my_index == 0:
                return

            # Watch the node before us in the sequence
            predecessor = candidates[my_index - 1]
            predecessor_path = f"{self.election_path}/{predecessor}"

            logging.info(f"Watching predecessor: {predecessor_path}")

            # Set up a watch on the predecessor
            @self.zk.DataWatch(predecessor_path)
            def watch_predecessor(data, stat, event):
                if event and event.type == "DELETED":
                    logging.info(f"Predecessor node {predecessor_path} deleted")
                    # Check if we are now the leader
                    self._check_leadership()
                    # Set up watch on new predecessor
                    self._watch_predecessor()

                    # Signal the event for leader changes
                    self.leader_watch_event.set()

                return True

        except Exception as e:
            logging.error(f"Error setting up predecessor watch: {e}")

    def get_current_leader(self):
        """
        Get the current leader ID.

        Returns:
            str: ID of the current leader, or None if no leader exists
        """
        if not self.zk:
            return None

        try:
            candidates = self._get_sorted_candidates()
            if not candidates:
                return None

            # Leader is the lowest sequence number
            leader_node = candidates[0]
            leader_path = f"{self.election_path}/{leader_node}"

            # Get leader data
            leader_data, _ = self.zk.get(leader_path)

            return leader_data.decode("utf-8")

        except Exception as e:
            logging.error(f"Error getting current leader: {e}")
            return None

    def add_leadership_callback(self, callback):
        """
        Add a callback function to be called when leadership changes.

        Args:
            callback: Function to call when leadership changes.
                     Will be called with a boolean argument indicating if this node is now the leader.
        """
        self.leadership_callbacks.append(callback)

    def _trigger_leadership_callbacks(self, is_leader):
        """
        Trigger all registered leadership callbacks.

        Args:
            is_leader: Boolean indicating if this node is now the leader
        """
        for callback in self.leadership_callbacks:
            try:
                callback(is_leader)
            except Exception as e:
                logging.error(f"Error in leadership callback: {e}")

    def wait_for_leadership_change(self, timeout=None):
        """
        Wait for a leadership change notification.

        Args:
            timeout: Maximum time to wait in seconds, or None to wait indefinitely

        Returns:
            bool: True if leadership changed, False if timeout occurred
        """
        # Clear any previous events
        self.leader_watch_event.clear()

        # Wait for the event
        return self.leader_watch_event.wait(timeout)

    def create_leader_data(self, data):
        """
        Create or update a node with leader-specific data.

        Args:
            data: Dictionary of data to store

        Returns:
            bool: True if successful, False otherwise
        """
        if not self.zk:
            return False

        if not self.is_leader:
            logging.warning("Cannot create leader data: not the leader")
            return False

        try:
            # Convert data to JSON string
            import json

            data_bytes = json.dumps(data).encode("utf-8")

            # Create or update leader data node
            leader_data_path = "/myapp/leader_data"

            if self.zk.exists(leader_data_path):
                self.zk.set(leader_data_path, data_bytes)
            else:
                self.zk.create(leader_data_path, data_bytes, makepath=True)

            return True

        except Exception as e:
            logging.error(f"Error creating leader data: {e}")
            return False

    def get_leader_data(self):
        """
        Get the current leader's data.

        Returns:
            dict: Leader data, or None if no data exists
        """
        if not self.zk:
            return None

        try:
            leader_data_path = "/myapp/leader_data"

            if not self.zk.exists(leader_data_path):
                return None

            data, _ = self.zk.get(leader_data_path)

            # Parse JSON data
            import json

            return json.loads(data.decode("utf-8"))

        except Exception as e:
            logging.error(f"Error getting leader data: {e}")
            return None

    def stop(self, timeout=5):
        """
        Properly stop the ZooKeeper client with a timeout.

        Args:
            timeout: Maximum time in seconds to wait for ZK to disconnect
        """
        if self.zk is None:
            logging.debug("ZooKeeper already stopped or not started")
            return

        try:
            logging.info("Stopping ZooKeeper client...")

            # Start a background thread to stop ZK with a timeout
            # This prevents hanging if ZK stop() blocks
            def stop_zk_with_timeout():
                try:
                    self.zk.stop()
                    self.zk.close()
                    logging.info("ZooKeeper client stopped successfully")
                except Exception as e:
                    logging.error(f"Error stopping ZooKeeper client: {e}")

            stop_thread = threading.Thread(target=stop_zk_with_timeout)
            stop_thread.daemon = (
                True  # Make it a daemon thread so it won't block shutdown
            )
            stop_thread.start()

            # Wait with timeout
            stop_thread.join(timeout)

            if stop_thread.is_alive():
                logging.warning(
                    f"ZooKeeper client did not stop within {timeout} seconds, continuing shutdown"
                )
                # Let the daemon thread continue in the background
                # System exit will force it to terminate
        except Exception as e:
            logging.error(f"Error during ZooKeeper shutdown: {e}")
        finally:
            # Clear references to allow garbage collection
            self.zk = None
