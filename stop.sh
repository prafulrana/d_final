#!/bin/bash

echo "Stopping all containers..."
docker stop ds-s0 ds-s1 ds-s2 publisher 2>/dev/null
docker rm ds-s0 ds-s1 ds-s2 publisher 2>/dev/null

echo "All containers stopped and removed."
