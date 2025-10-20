#!/bin/bash
echo "Starting continuous stream to in_s2..."
while true; do
  gst-launch-1.0 -e \
    filesrc location=/opt/nvidia/deepstream/deepstream-8.0/samples/streams/sample_1080p_h264.mp4 ! \
    qtdemux ! h264parse config-interval=-1 ! \
    rtspclientsink location=rtsp://34.47.221.242:8554/in_s2 protocols=tcp
done
