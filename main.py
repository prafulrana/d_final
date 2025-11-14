#!/usr/bin/env python3

"""Flask HTTP API for DeepStream pipeline control"""

import sys
import threading
from flask import Flask, request, jsonify

import pipeline
from config import *

# Relay configuration
relay_host = None
pipeline_thread = None

# Flask app
app = Flask(__name__)


def get_source_uri(source_id):
    """Generate RTSP input URI from relay host and source ID"""
    return f"rtsp://{relay_host}:8554/in_s{source_id}"


@app.route('/stream/restart', methods=['POST'])
def http_restart_stream():
    """Restart a stream by source ID"""
    global pipeline_thread

    data = request.json
    source_id = data.get('id')

    if source_id is None:
        return jsonify({"status": "error", "message": "id required"}), 400

    if source_id < 0 or source_id >= MAX_NUM_SOURCES:
        return jsonify({"status": "error", "message": f"id must be 0-{MAX_NUM_SOURCES-1}"}), 400

    # Create pipeline on first source (0->1 transition)
    if pipeline.pipeline is None:
        print(f"\n{'='*70}")
        print("Creating pipeline for first source...")
        print(f"{'='*70}\n")

        # Pass relay host to pipeline for URI generation
        if not pipeline.create_pipeline(relay_host):
            return jsonify({"status": "error", "message": "Failed to create pipeline"}), 500

        # Start GLib loop in background thread
        pipeline_thread = threading.Thread(target=pipeline.run_loop, daemon=True)
        pipeline_thread.start()

    # Restart the requested source
    if pipeline.restart_source(source_id):
        return jsonify({"status": "ok", "id": source_id})
    else:
        # Check if subprocess died - if so, recreate and retry
        if pipeline.pipeline is None:
            print(f"\n{'='*70}")
            print("Subprocess died - recreating pipeline...")
            print(f"{'='*70}\n")

            if not pipeline.create_pipeline(relay_host):
                return jsonify({"status": "error", "message": "Failed to recreate pipeline"}), 500

            # Retry restart
            if pipeline.restart_source(source_id):
                return jsonify({"status": "ok", "id": source_id})

        return jsonify({
            "status": "error",
            "message": f"Source {source_id} cannot be started. The RTSP stream is not available or not publishing.",
            "suggestion": "Ensure the stream is publishing to rtsp://{relay_host}:8554/in_s{source_id} before adding it"
        }), 400


@app.route('/stream/status', methods=['GET'])
def http_status():
    """Get status of all streams"""
    if pipeline.pipeline is None:
        return jsonify({
            "pipeline": "not_created",
            "active_streams": [],
            "count": 0,
            "relay_host": relay_host
        })

    status = pipeline.get_status()
    status["pipeline"] = "running"
    status["relay_host"] = relay_host
    return jsonify(status)


@app.route('/stream/hard_reset', methods=['POST'])
def http_hard_reset():
    """Kill and recreate pipeline subprocess (recovery from fatal errors)"""
    global pipeline_thread

    print(f"\n{'='*70}")
    print("HARD RESET requested - killing and recreating pipeline subprocess")
    print(f"{'='*70}\n")

    # Kill existing subprocess
    if pipeline.pipeline is not None:
        pipeline.destroy_pipeline()
        if pipeline_thread:
            pipeline_thread.join(timeout=2)
            pipeline_thread = None

    # Recreate pipeline
    if not pipeline.create_pipeline(relay_host):
        return jsonify({
            "status": "error",
            "message": "Failed to recreate pipeline after hard reset"
        }), 500

    # Start GLib loop in background thread
    pipeline_thread = threading.Thread(target=pipeline.run_loop, daemon=True)
    pipeline_thread.start()

    print(f"\n{'='*70}")
    print("✓ Pipeline subprocess recreated successfully")
    print(f"{'='*70}\n")

    return jsonify({
        "status": "ok",
        "message": "Pipeline subprocess recreated. All streams cleared. Use /stream/restart to add streams."
    })


def main(args):
    """Main entry point"""
    global relay_host

    if len(args) != 2:
        sys.stderr.write(f"usage: {args[0]} <relay_host>\n")
        sys.stderr.write(f"example: {args[0]} 34.47.221.242\n")
        sys.exit(1)

    relay_host = args[1]

    print(f"\n{'='*70}")
    print("DeepStream Multi-Stream Control API")
    print(f"{'='*70}")
    print(f"HTTP API: http://localhost:{HTTP_API_PORT}")
    print(f"  POST /stream/restart {{\"id\": N}} - Start/restart stream N")
    print(f"  GET  /stream/status - Get status of all streams")
    print(f"  POST /stream/hard_reset - Kill and recreate pipeline (fatal error recovery)")
    print(f"\nRelay host: {relay_host}")
    print(f"Input pattern: rtsp://{relay_host}:8554/in_s{{N}}")
    print(f"Output pattern: rtsp://localhost:{RTSP_SERVER_PORT}/x{{N}}")
    print(f"Max streams: {MAX_NUM_SOURCES}")
    print(f"\nPipeline will be created on first /stream/restart call")
    print(f"{'='*70}\n")

    # Start Flask (blocking)
    app.run(host='0.0.0.0', port=HTTP_API_PORT, threaded=True)


if __name__ == '__main__':
    sys.exit(main(sys.argv))
