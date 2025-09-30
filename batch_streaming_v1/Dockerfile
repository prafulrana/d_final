FROM nvcr.io/nvidia/deepstream:8.0-triton-multiarch

# Install Python dependencies
RUN apt-get update && apt-get install -y \
    python3-gi \
    python3-gst-1.0 \
    gir1.2-gst-rtsp-server-1.0 \
    && rm -rf /var/lib/apt/lists/*

# Copy application files
COPY pgie.txt /opt/nvidia/deepstream/deepstream-8.0/pgie.txt
COPY pipeline.txt /opt/nvidia/deepstream/deepstream-8.0/pipeline.txt
COPY rtsp_server.py /opt/nvidia/deepstream/deepstream-8.0/rtsp_server.py
RUN chmod +x /opt/nvidia/deepstream/deepstream-8.0/rtsp_server.py

WORKDIR /opt/nvidia/deepstream/deepstream-8.0

# Start RTSP server with pipeline (unbuffered output)
CMD ["python3", "-u", "rtsp_server.py", "pipeline.txt"]