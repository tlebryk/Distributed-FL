.PHONY: grpc_update build_server build_client run_server docker_run_client_a docker_run_client_b

grpc_update:
	python -m grpc_tools.protoc -I. --python_out=distributed_fl --grpc_python_out=distributed_fl distributed_fl/model_update.proto

build_server:
	docker build -f Dockerfile.server -t server distributed_fl
build_client:
	docker build -f Dockerfile.client -t client distributed_fl

run_server:
	docker run -p 50051:50051 server

docker_run_client_a:
	docker run -p 50052:50052 client

docker_run_client_b:
	docker run -p 50053:50053 client

local_test:
	cd distributed_fl && python -m pytest