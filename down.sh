#!/bin/bash

echo "Stopping all drishti pipelines..."
docker ps -q --filter "name=drishti-" | xargs -r docker kill 2>/dev/null || true
sleep 2

echo "All pipelines stopped"
docker ps --filter "name=drishti-"
