FROM nvcr.io/nvidia/deepstream:8.0-triton-multiarch

# Run NVIDIA's install script to properly set up PyServiceMaker
RUN /opt/nvidia/deepstream/deepstream-8.0/install.sh

# Install missing dependencies that NVIDIA forgot to include
RUN pip install pyyaml
RUN apt-get update && apt-get install -y libgstreamer-plugins-bad1.0-dev gir1.2-gst-rtsp-server-1.0

# GStreamer already available in DeepStream image

# Copy our hello world script and configs
COPY hello_multi.py /opt/nvidia/deepstream/deepstream-8.0/
COPY pgie_b2.txt /opt/nvidia/deepstream/deepstream-8.0/
COPY pgie_b8.txt /opt/nvidia/deepstream/deepstream-8.0/

# Default command to run our hello world
CMD ["python3", "hello_multi.py"]