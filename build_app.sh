#!/bin/bash
# Build C++ DeepStream YOLO detection application (no segmentation)

set -e

echo "Building C++ DeepStream YOLO application..."

# Compile main application
echo "1. Compiling main application..."
g++ -c main.cpp -o main.o \
    -fPIC \
    -I/opt/nvidia/deepstream/deepstream-8.0/sources/includes \
    $(pkg-config --cflags gstreamer-1.0 glib-2.0)

# Link executable
echo "2. Linking executable..."
g++ -o deepstream_app \
    main.o \
    -L/opt/nvidia/deepstream/deepstream-8.0/lib \
    -lnvdsgst_meta \
    -lnvds_meta \
    $(pkg-config --libs gstreamer-1.0 glib-2.0)

echo "Build complete! Executable: ./deepstream_app"
echo ""
echo "Usage: ./deepstream_app <rtsp_in> <rtsp_out> <pgie_config>"
