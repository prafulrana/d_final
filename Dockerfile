FROM nvcr.io/nvidia/deepstream:8.0-triton-multiarch

# Add DeepStream libraries to LD_LIBRARY_PATH
ENV LD_LIBRARY_PATH=/opt/nvidia/deepstream/deepstream-8.0/lib:$LD_LIBRARY_PATH

# Enable GStreamer debug logging
ENV GST_DEBUG=2
ENV GST_DEBUG_NO_COLOR=1

WORKDIR /app

# Copy C++ source files
COPY segmentation_overlay_direct.cu /app/
COPY segmentation_probe_complete.cpp /app/
COPY main.cpp /app/

# Copy and run build script
COPY build_app.sh /app/
RUN chmod +x /app/build_app.sh && /app/build_app.sh

# Create models directory
RUN mkdir -p /models && chmod 777 /models

CMD ["/app/deepstream_app", \
     "rtsp://34.14.140.30:8554/in_s5", \
     "rtsp://34.14.140.30:8554/s5", \
     "/config/pgie_peoplesegnet.txt"]
