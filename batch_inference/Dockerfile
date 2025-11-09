FROM nvcr.io/nvidia/deepstream:8.0-triton-multiarch

# Install Python dependencies
RUN apt-get update && apt-get install -y \
    python3-flask \
    python3-gi \
    && rm -rf /var/lib/apt/lists/*

# Copy application and YOLO custom library
COPY batch_app.py /app/batch_app.py
COPY libnvdsinfer_custom_impl_Yolo.so /app/libnvdsinfer_custom_impl_Yolo.so
RUN chmod +x /app/batch_app.py

WORKDIR /app

# Expose Flask API port
EXPOSE 5555

# RTSP ports (8554-8589 for 36 streams)
EXPOSE 8554-8589

CMD ["python3", "-u", "/app/batch_app.py"]
