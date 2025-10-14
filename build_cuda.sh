#!/bin/bash
# Build CUDA kernel for segmentation overlay

nvcc -shared -Xcompiler -fPIC \
    -o segmentation_overlay.so \
    segmentation_overlay.cu \
    -I/usr/local/cuda/include \
    -L/usr/local/cuda/lib64 \
    -lcudart

echo "Built segmentation_overlay.so"
