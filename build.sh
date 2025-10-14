#!/bin/bash
set -e

echo "Stopping all drishti pipelines..."
docker ps -q --filter "name=drishti-" | xargs -r docker kill 2>/dev/null || true

echo "Building Docker image..."
docker build -t ds_python:latest .
