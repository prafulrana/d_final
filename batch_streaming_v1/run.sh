#!/bin/bash
docker run --rm \
  --gpus all \
  --network host \
  batch_streaming:latest