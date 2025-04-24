#!/bin/bash
set -e

echo "Starting Federated Learning Server..."

# Activate virtual environment if it exists
if [ -f "/app/.venv/bin/activate" ]; then
    echo "Activating virtual environment..."
    source /app/.venv/bin/activate
fi

# Run any pre-flight checks or setup tasks here
# e.g., ensure model proto files exist, clean up temp files, etc.

# Start the gRPC server
exec "$@"
