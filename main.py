#!/usr/bin/env python3

"""Flask HTTP API for DeepStream pipeline control"""

import sys
import threading
from flask import Flask, request, jsonify

import pipeline
from config import *

# Store URIs from command line
source_uris = {}
pipeline_thread = None

# Flask app
app = Flask(__name__)


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

    if source_id not in source_uris:
        return jsonify({"status": "error", "message": f"source {source_id} not configured"}), 400

    # Create pipeline on first source (0->1 transition)
    if pipeline.pipeline is None:
        print(f"\n{'='*70}")
        print("Creating pipeline for first source...")
        print(f"{'='*70}\n")

        # Pass all URIs to pipeline
        uris = [source_uris[i] for i in sorted(source_uris.keys())]
        if not pipeline.create_pipeline(uris):
            return jsonify({"status": "error", "message": "Failed to create pipeline"}), 500

        # Start GLib loop in background thread
        pipeline_thread = threading.Thread(target=pipeline.run_loop, daemon=True)
        pipeline_thread.start()

    # Restart the requested source
    if pipeline.restart_source(source_id):
        return jsonify({"status": "ok", "id": source_id})
    else:
        return jsonify({"status": "error", "message": "restart failed"}), 500


@app.route('/stream/status', methods=['GET'])
def http_status():
    """Get status of all streams"""
    if pipeline.pipeline is None:
        return jsonify({
            "pipeline": "not_created",
            "active_streams": [],
            "count": 0,
            "configured_sources": list(source_uris.keys())
        })

    status = pipeline.get_status()
    status["pipeline"] = "running"
    status["configured_sources"] = list(source_uris.keys())
    return jsonify(status)


def main(args):
    """Main entry point"""
    if len(args) < 2:
        sys.stderr.write(f"usage: {args[0]} <uri1> [uri2] [uri3] ...\n")
        sys.exit(1)

    # Store URIs from command line
    for i in range(len(args) - 1):
        uri = args[i + 1]
        source_uris[i] = uri
        print(f"Configured source {i}: {uri}")

    print(f"\n{'='*70}")
    print("DeepStream Multi-Stream Control API")
    print(f"{'='*70}")
    print(f"HTTP API: http://localhost:{HTTP_API_PORT}")
    print(f"  POST /stream/restart {{\"id\": N}} - Start/restart stream N")
    print(f"  GET  /stream/status - Get status of all streams")
    print(f"\nConfigured sources: {len(source_uris)}")
    print(f"Pipeline will be created on first /stream/restart call")
    print(f"{'='*70}\n")

    # Start Flask (blocking)
    app.run(host='0.0.0.0', port=HTTP_API_PORT, threaded=True)


if __name__ == '__main__':
    sys.exit(main(sys.argv))
