
# Distributed Federated Learning System

This project implements a distributed federated learning system using gRPC for communication between servers and clients.

## Getting Started

### Prerequisites

- Python 3.12
- [uv](https://github.com/astral-sh/uv) - Fast Python package installer and resolver

### Setup

1. Clone the repository
2. Install dependencies using uv:

```bash
uv sync
```

### Running commands in a distributed setting

1. Set up 3 zookeeper nodes

On three different machines:
```bash
docker run -d --name zk<n> \
  -p 2181:2181 -p 2888:2888 -p 3888:3888 \
  -e ZOO_MY_ID=<n> \
  -e ZOO_SERVERS="server.1=<ip1>>:2888:3888 server.2=<ip2>:2888:3888 server.3=<ip3>:2888:3888" \
  zookeeper:latest
```


2. Set up leader
```bash
python server.py --mode leader --server-port 50051 --zk-hosts "<ip1>:2181,<ip2>:2181,<ip3>:2181"
```

3. Set up replicas

On two different machines:
```bash
python server.py --mode replica --leader-address <leaderip>:50051 --server-port 50051 \
  --zk-hosts "<ip1>:2181,<ip2>:2181,<ip3>:2181" --auto-takeover
python server.py --mode replica --leader-address localhost:50051 --server-port 50051 \
  --zk-hosts "<ip1>:2181,<ip2>:2181,<ip3>:2181" --auto-takeover
```

Note that zookeeper handles replica discovery so replicas don't need to know about each other on initialization.

4. Set up the client 

```bash
uv run python distributed_fl/client.py -- server_address <leaderip>":50051 --fallback_addresses <replicaip>:50051 <replicaip2>:50051 
```

5. Set up the client inference server 
```bash
uv run python distributed_fl/inference_server.py
```

6. Set up the UI 
```bash
cd UI
npm run dev
```

You can open the UI at `http://localhost:8080/`.

## Components

- *Server (distributed_fl/server.py)*
The server component manages the global model and coordinates the federated learning process. It receives model updates from clients. Aggregates updates to improve the global model. Distributes the updated global model back to clients

- *Client: training (distributed_fl/client.py)*
Trains on local data. Sends model updates to the server Receives the latest global model from the server after a training round has finished. 

- *Client: inference (distributed_fl/inference_server.py):*
Spins up an inference server for the front end to hit. Loads the latest adapter on restart. 

## Hitting the inference server

Make sure you're running the client and server (see above).
Sample request: 
```
curl -X POST http://127.0.0.1:5000/generate   -H "Content-Type: application/json"   -d '{"prompt": "Write a Python function to sort a list:"}'
```

## Testing:
From the root directory: 
 
```bash 
make local_test
```

