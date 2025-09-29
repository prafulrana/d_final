#!/bin/bash

# Run PyServiceMaker container with auto-cleanup
echo "Running PyServiceMaker container..."
docker run --rm \
    --name pyservicemaker-test \
    --gpus all \
    --network host \
    pyservicemaker-hello:latest

echo "Container finished and removed."