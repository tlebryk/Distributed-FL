#!/bin/bash
set -e

# Activate the virtual environment, if it exists
if [ -f "/app/.venv/bin/activate" ]; then
    echo "Activating virtual environment..."
    source /app/.venv/bin/activate
fi

# Optional: run migrations or other pre-start tasks here
# echo "Running pre-start tasks..."

# Start the FastAPI application
exec "$@"
