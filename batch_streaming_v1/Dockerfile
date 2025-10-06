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
    libopenh264-7 \
    gstreamer1.0-tools \
    libx264-164 \
    libvpx9 \
    libmp3lame0 \
    libflac12t64 \
    libmpg123-0 \
    libdvdnav4 \
    libdvdread8 \
    libdca0 \
    mjpegtools \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Clear GStreamer cache and ensure encoders are discoverable
RUN rm -rf /root/.cache/gstreamer-1.0 || true
RUN gst-inspect-1.0 x264enc || gst-inspect-1.0 avenc_h264 || true

# Copy application files
COPY pgie.txt /opt/nvidia/deepstream/deepstream-8.0/pgie.txt
COPY pipeline.txt /opt/nvidia/deepstream/deepstream-8.0/pipeline.txt
COPY src /opt/nvidia/deepstream/deepstream-8.0/src
# Optionally run DeepStream helper if present in base image too
RUN if [ -x /opt/nvidia/deepstream/deepstream/user_additional_install.sh ]; then \
      bash /opt/nvidia/deepstream/deepstream/user_additional_install.sh || true; \
    fi

WORKDIR /opt/nvidia/deepstream/deepstream-8.0
ENV CTRL_PORT=8080

# Compile C RTSP server (multi-file, simple layering)
RUN gcc -O2 -pipe -Wall -Wextra -Isrc -o rtsp_server \
    src/main.c src/app.c src/config.c src/branch.c src/control.c \
    $(pkg-config --cflags --libs gstreamer-1.0 gstreamer-rtsp-server-1.0 glib-2.0)

# Start RTSP server (NVENC + UDP-wrapped RTSP). Override via env.
CMD ["/opt/nvidia/deepstream/deepstream-8.0/rtsp_server"]

HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
  CMD curl -fsS http://127.0.0.1:${CTRL_PORT}/status || exit 1
