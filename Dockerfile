FROM nvcr.io/nvidia/deepstream:8.0-triton-multiarch

# Install Python dependencies and pyds bindings
RUN apt-get update && apt-get install -y \
    python3-flask \
    python3-gi \
    && rm -rf /var/lib/apt/lists/*

# Install DeepStream Python bindings by building from source
RUN cd /opt/nvidia/deepstream/deepstream-8.0 && \
    ./user_deepstream_python_apps_install.sh --build-bindings

# Copy main application and dependencies
COPY config.py /app/config.py
COPY pipeline.py /app/pipeline.py
COPY main.py /app/main.py
COPY libnvdsinfer_custom_impl_Yolo.so /app/libnvdsinfer_custom_impl_Yolo.so
COPY config_tracker_NvDCF_perf.yml /app/config_tracker_NvDCF_perf.yml

RUN chmod +x /app/main.py

WORKDIR /app

# RTSP server port (individual outputs per stream on 9600)
EXPOSE 9600

# HTTP control API
EXPOSE 5555

# Set Python path to include pyds
ENV PYTHONPATH=/opt/nvidia/deepstream/deepstream/sources/deepstream_python_apps/pyds/lib/python3.12/site-packages:$PYTHONPATH

CMD ["python3", "-u", "/app/main.py", "34.47.221.242"]
