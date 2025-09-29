FROM nvcr.io/nvidia/deepstream:8.0-triton-multiarch

# Run NVIDIA's install script to properly set up PyServiceMaker
RUN /opt/nvidia/deepstream/deepstream-8.0/install.sh

# Install missing PyYAML dependency that NVIDIA forgot to include
RUN pip install pyyaml

# Copy our hello world script
COPY hello_from_psyservicemaker.py /opt/nvidia/deepstream/deepstream-8.0/

# Default command to run our hello world
CMD ["python3", "hello_from_psyservicemaker.py"]