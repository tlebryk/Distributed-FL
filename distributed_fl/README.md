
# Distributed Federated Learning System

This project implements a distributed federated learning system using gRPC for communication between servers and clients.

## Getting Started

### Prerequisites

- Python 3.x
- [uv](https://github.com/astral-sh/uv) - Fast Python package installer and resolver

### Setup

1. Clone the repository
2. Install dependencies using uv:

```bash
uv sync
```

### Run the server
```bash
make zookeeper_build
```

```bash
make zookeeper_run
```

```bash
make local_server
```
### Run a client

```bash
make local_client
```

## Components

- Server (distributed_fl/server.py)
The server component manages the global model and coordinates the federated learning process. It:

Receives model updates from clients
Aggregates updates to improve the global model
Distributes the updated global model back to clients

- Client (distributed_fl/client.py)

Spins up an inference server for the front end to hit. 
Trains on local data
Sends model updates to the server
Receives the latest global model from the server after a training round has finished. 


## Hitting the inference server

Make sure you're running the client and server (see above).
Sample request: 
```
curl -X POST http://127.0.0.1:5000/generate   -H "Content-Type: application/json"   -d '{"prompt": "Write a Python function to sort a list:"}'
```