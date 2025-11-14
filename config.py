"""Configuration constants for DeepStream pipeline"""

# Pipeline configuration
GPU_ID = 0
MAX_NUM_SOURCES = 36
MUXER_OUTPUT_WIDTH = 720
MUXER_OUTPUT_HEIGHT = 1280
MUXER_BATCH_TIMEOUT_USEC = 33000
PGIE_CONFIG_FILE = "/config/bowling_yolo12n_batch.txt"

# Network configuration
RTSP_SERVER_PORT = 9600
RTSP_UDPSINK_BASE_PORT = 5001
HTTP_API_PORT = 5555

# Stream restart delay
SOURCE_RESTART_DELAY_SEC = 0.5

# Encoder settings
ENCODER_BITRATE = 4000000
ENCODER_IDR_INTERVAL = 30  # IDR frame every 30 frames (~1 sec at 30fps)
ENCODER_IFRAME_INTERVAL = 0  # Force IDR at start

# RTSP settings
RTSP_UDP_BUFFER_SIZE = 524288
RTSP_CAPS_STRING = "application/x-rtp, media=video, clock-rate=90000, encoding-name=H264, payload=96"
