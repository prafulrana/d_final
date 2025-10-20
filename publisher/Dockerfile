FROM nvcr.io/nvidia/deepstream:8.0-triton-multiarch

# Copy video file and loop script
COPY bowling_bottom_right.mp4 /bowling_bottom_right.mp4
COPY loop_stream.sh /loop_stream.sh
RUN chmod +x /loop_stream.sh

CMD ["/bin/bash", "/loop_stream.sh"]
