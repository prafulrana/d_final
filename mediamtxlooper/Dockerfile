FROM bluenviron/mediamtx:latest-ffmpeg

COPY sample_1080p_h264.mp4 /
COPY mediamtx.yml /
WORKDIR /

CMD ["sh", "-c", "mediamtx & sleep 2 && ffmpeg -re -stream_loop -1 -i /sample_1080p_h264.mp4 -c copy -f rtsp rtsp://localhost:8554/iphone-feed"]