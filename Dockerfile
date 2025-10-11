FROM nvcr.io/nvidia/deepstream:8.0-triton-multiarch

# Run official Python installation script
RUN cd /opt/nvidia/deepstream/deepstream-8.0 && \
    ./user_deepstream_python_apps_install.sh --build-bindings

# Enable GStreamer debug logging
ENV GST_DEBUG=2
ENV GST_DEBUG_NO_COLOR=1

WORKDIR /app
COPY app.py /app/

# Create models directory and copy ONNX model for TensorRT engine caching
RUN mkdir -p /models && chmod 777 /models && \
    cp /opt/nvidia/deepstream/deepstream-8.0/samples/models/Primary_Detector/resnet18_trafficcamnet_pruned.onnx /models/

# Copy and modify pgie config to use /models for both ONNX and engine
RUN cp /opt/nvidia/deepstream/deepstream-8.0/sources/apps/sample_apps/deepstream-test1/dstest1_pgie_config.txt /app/pgie_config.txt && \
    sed -i 's|onnx-file=.*|onnx-file=/models/resnet18_trafficcamnet_pruned.onnx|g' /app/pgie_config.txt && \
    sed -i 's|model-engine-file=.*|model-engine-file=/models/resnet18_trafficcamnet_pruned.onnx_b1_gpu0_fp16.engine|g' /app/pgie_config.txt && \
    sed -i 's|labelfile-path=.*|labelfile-path=/opt/nvidia/deepstream/deepstream-8.0/samples/models/Primary_Detector/labels.txt|g' /app/pgie_config.txt && \
    sed -i 's|int8-calib-file=.*|int8-calib-file=/opt/nvidia/deepstream/deepstream-8.0/samples/models/Primary_Detector/cal_trt.bin|g' /app/pgie_config.txt

CMD ["python3", "-u", "/app/app.py", \
     "-i", "rtsp://34.100.230.7:8554/in_s0", \
     "-o", "rtsp://34.100.230.7:8554/s0", \
     "-c", "/app/pgie_config.txt"]
