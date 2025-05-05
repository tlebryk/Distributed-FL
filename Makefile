.PHONY: grpc_update build_server run_server server_entrypoint local_test local_server local_client local_inference zookeeper_build zookeeper_run 

grpc_update:
	uv run python -m grpc_tools.protoc -I=distributed_fl --python_out=distributed_fl --grpc_python_out=distributed_fl distributed_fl/model_update.proto

build_server:
	docker build -f distributed_fl/Dockerfile.server -t server . --progress=plain

run_server:
	docker run -p 50051:50051 \
	-v ./distributed_fl:/app/distributed_fl \
	-v ~/.cache/huggingface/:/root/.cache/huggingface \
	server

server_entrypoint: 
	docker run --rm -it -p 50051:50051 \
	-v ./distributed_fl:/app/distributed_fl \
	-v ~/.cache/huggingface/:/root/.cache/huggingface \
	server bash 

local_test:
	cd distributed_fl && uv run python -m pytest --cov=./

local_server:
	uv run python distributed_fl/server.py

local_client:
	uv run python distributed_fl/client.py

local_inference:
	uv run python distributed_fl/inference_server.py

zookeeper_build:
	docker build -t my-zk:latest -f distributed_fl/Dockerfile.zookeeper .

zookeeper_run:
	docker run -d --rm --name zk_local -p 2181:2181 my-zk:latest
