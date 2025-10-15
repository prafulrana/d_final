#!/bin/bash
# Build complete C++ DeepStream application

set -e

echo "Building complete C++ DeepStream application..."

# Compile CUDA kernel
echo "1. Compiling CUDA kernel..."
nvcc -c segmentation_overlay_direct.cu -o segmentation_overlay_direct.o \
    --compiler-options '-fPIC' \
    -arch=sm_75 \
    -Wno-deprecated-gpu-targets

# Compile C++ probe
echo "2. Compiling C++ probe..."
g++ -c segmentation_probe_complete.cpp -o segmentation_probe_complete.o \
    -fPIC \
    -I/opt/nvidia/deepstream/deepstream-8.0/sources/includes \
    -I/usr/local/cuda/include \
    $(pkg-config --cflags gstreamer-1.0)

# Compile main application
echo "3. Compiling main application..."
g++ -c main.cpp -o main.o \
    -fPIC \
    -I/opt/nvidia/deepstream/deepstream-8.0/sources/includes \
    -I/usr/local/cuda/include \
    $(pkg-config --cflags gstreamer-1.0 glib-2.0)

# Link everything into executable
echo "4. Linking executable..."
g++ -o deepstream_app \
    main.o segmentation_probe_complete.o segmentation_overlay_direct.o \
    -L/usr/local/cuda/lib64 \
    -L/opt/nvidia/deepstream/deepstream-8.0/lib \
    -lcudart \
    -lnvdsgst_meta \
    -lnvds_meta \
    $(pkg-config --libs gstreamer-1.0 glib-2.0)

echo "Build complete! Executable: ./deepstream_app"
echo ""
echo "Usage: ./deepstream_app <rtsp_in> <rtsp_out> <pgie_config>"
