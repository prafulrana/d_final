#!/bin/bash

# Build from app directory
echo "Building PyServiceMaker hello world image..."
cd app
docker build -t pyservicemaker-hello:latest .
cd ..

# Build complete - image ready for use