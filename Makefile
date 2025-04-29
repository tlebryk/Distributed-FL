.PHONY: grpc_update build_server build_client run_server docker_run_client_a docker_run_client_b

grpc_update:
	python -m grpc_tools.protoc -I. --python_out=distributed_fl --grpc_python_out=distributed_fl distributed_fl/model_update.proto

build_server:
	docker build -f distributed_fl/Dockerfile.server -t server . --progress=plain

build_client:
	docker build -f distributed_fl/Dockerfile.client -t client .

run_server:
	docker run -p 50051:50051 \
	-v ./distributed_fl:/app/distributed_fl \
	-v ~/.cache/huggingface/:/root/.cache/huggingface \
	server

run_client_a:
	docker run -p 50052:50052 client

run_client_b:
	docker run -p 50053:50053 client

client_entrypoint: 
	docker run --rm -it -p 50052:50052 client bash


server_entrypoint: 
	docker run --rm -it -p 50051:50051 \
	-v ./distributed_fl:/app/distributed_fl \
	-v ~/.cache/huggingface/:/root/.cache/huggingface \
	server bash 

local_test:
	cd distributed_fl && uv run python -m pytest

local_server:
	uv run python distributed_fl/server.py

local_client:
	uv run python distributed_fl/client.py

zookeeper_build:
	docker build -t my-zk:latest -f distributed_fl/Dockerfile.zookeeper .

zookeeper_run:
	docker run -d --rm --name zk -p 2181:2181 my-zk:latest