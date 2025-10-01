FROM nvcr.io/nvidia/deepstream:8.0-triton-multiarch

# Build minimal C RTSP server (GStreamer + RTSP Server)
RUN apt-get update && apt-get install -y \
    build-essential \
    pkg-config \
    libgstreamer1.0-dev \
    libgstreamer-plugins-base1.0-dev \
    libgstrtspserver-1.0-dev \
    gstreamer1.0-plugins-good \
    gstreamer1.0-plugins-bad \
    gstreamer1.0-plugins-ugly \
    gstreamer1.0-libav \
    gstreamer1.0-tools \
    libx264-164 \
    && rm -rf /var/lib/apt/lists/*

# Copy application files
COPY pgie.txt /opt/nvidia/deepstream/deepstream-8.0/pgie.txt
COPY pipeline.txt /opt/nvidia/deepstream/deepstream-8.0/pipeline.txt
COPY src/rtsp_server.c /opt/nvidia/deepstream/deepstream-8.0/rtsp_server.c

WORKDIR /opt/nvidia/deepstream/deepstream-8.0

# Compile C RTSP server
RUN gcc -O2 -pipe -Wall -Wextra -o rtsp_server rtsp_server.c \
    $(pkg-config --cflags --libs gstreamer-1.0 gstreamer-rtsp-server-1.0 glib-2.0)

# Start RTSP server (NVENC + UDP-wrapped RTSP). Override via env.
CMD ["/opt/nvidia/deepstream/deepstream-8.0/rtsp_server"]
