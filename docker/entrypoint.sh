#!/bin/bash
# Entrypoint script for SeamlessExpressive container
# Supports both CLI mode and API server mode

set -e

# Default mode is CLI (keeps container running)
MODE=${SEAMLESS_MODE:-cli}

echo "Starting SeamlessExpressive in $MODE mode..."

if [ "$MODE" = "api" ]; then
    echo "Starting API server on port 8000..."
    
    # Install FastAPI dependencies if not already installed
    if ! python -c "import fastapi" 2>/dev/null; then
        echo "Installing FastAPI dependencies..."
        pip install --no-cache-dir fastapi uvicorn python-multipart
    fi
    
    # Check PyTorch version
    echo "Checking PyTorch version..."
    python -c "import torch; print(f'PyTorch version: {torch.__version__}')"
    
    # Try alternate API start method using uvicorn
    cd /app
    echo "PYTHONPATH is: $PYTHONPATH"
    echo "Starting API server using uvicorn..."
    
    # Use uvicorn to start the API server
    python -m uvicorn seamless_communication.api_server:app --host 0.0.0.0 --port 8000 --reload
else
    echo "Running in CLI mode (container will stay running)..."
    # Keep container running for CLI commands
    tail -f /dev/null
fi