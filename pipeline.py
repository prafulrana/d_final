"""DeepStream pipeline manager - thin wrapper around subprocess"""

import multiprocessing
import time
from config import *

# Subprocess handle
_subprocess = None
_command_queue = None
_response_queue = None
_relay_host = None

# Expose these for main.py compatibility
pipeline = None  # Will be set to "subprocess" when subprocess is running


def launch_subprocess(relay_host):
    """Launch pipeline in subprocess"""
    global _subprocess, _command_queue, _response_queue, _relay_host, pipeline

    print("Launching pipeline subprocess...")

    _relay_host = relay_host
    _command_queue = multiprocessing.Queue()
    _response_queue = multiprocessing.Queue()

    # Import here to avoid loading in main process
    from pipeline_subprocess import subprocess_main

    _subprocess = multiprocessing.Process(
        target=subprocess_main,
        args=(relay_host, _command_queue, _response_queue),
        daemon=False
    )
    _subprocess.start()

    # Wait for init
    try:
        response = _response_queue.get(timeout=30)
        if response.get("event") == "INIT_SUCCESS":
            print("✓ Pipeline subprocess initialized")
            pipeline = "subprocess"  # Mark as running
            return True
        else:
            print("✗ Pipeline subprocess init failed")
            return False
    except:
        print("✗ Pipeline subprocess init timeout")
        return False


def kill_subprocess():
    """Kill pipeline subprocess"""
    global _subprocess, _command_queue, _response_queue, pipeline

    if _subprocess is None:
        return

    print("Killing pipeline subprocess...")

    # Try graceful shutdown first
    try:
        _command_queue.put({"cmd": "SHUTDOWN"})
        response = _response_queue.get(timeout=2)
    except:
        pass

    # Wait for exit
    _subprocess.join(timeout=5)

    if _subprocess.is_alive():
        print("⚠ Subprocess did not exit, terminating...")
        _subprocess.terminate()
        _subprocess.join(timeout=2)

    if _subprocess.is_alive():
        print("⚠ Subprocess still alive, killing...")
        _subprocess.kill()

    _subprocess = None
    _command_queue = None
    _response_queue = None
    pipeline = None

    print("✓ Pipeline subprocess killed")


def create_pipeline(relay_host):
    """Create pipeline (compatibility wrapper)"""
    return launch_subprocess(relay_host)


def destroy_pipeline():
    """Destroy pipeline (compatibility wrapper)"""
    kill_subprocess()


def restart_source(source_id):
    """Restart a source via subprocess"""
    global _command_queue, _response_queue, _subprocess, pipeline

    if pipeline is None:
        print("Pipeline subprocess not running")
        return False

    # Check if subprocess died
    if _subprocess and not _subprocess.is_alive():
        print("⚠ Pipeline subprocess died, resetting state")
        pipeline = None
        return False

    try:
        _command_queue.put({"cmd": "START_SOURCE", "id": source_id})
        response = _response_queue.get(timeout=10)
        return response.get("status") == "ok"
    except Exception as e:
        print(f"Error communicating with subprocess: {e}")
        # Subprocess may have died, reset state
        if _subprocess and not _subprocess.is_alive():
            print("⚠ Subprocess died during communication")
            pipeline = None
        return False


def get_status():
    """Get status via subprocess"""
    global _command_queue, _response_queue, _subprocess, _relay_host, pipeline

    if pipeline is None:
        return {
            "active_streams": [],
            "count": 0,
            "uris": {},
            "rtsp_server": f"rtsp://localhost:{RTSP_SERVER_PORT}",
            "rtsp_paths": {}
        }

    # Check if subprocess died
    if _subprocess and not _subprocess.is_alive():
        print("⚠ Pipeline subprocess died, resetting state")
        pipeline = None
        return {
            "active_streams": [],
            "count": 0,
            "uris": {},
            "rtsp_server": f"rtsp://localhost:{RTSP_SERVER_PORT}",
            "rtsp_paths": {}
        }

    try:
        _command_queue.put({"cmd": "GET_STATUS"})
        response = _response_queue.get(timeout=2)
        return response
    except Exception as e:
        print(f"Error getting status: {e}")
        # Subprocess may have died, reset state
        if _subprocess and not _subprocess.is_alive():
            print("⚠ Subprocess died during status check")
            pipeline = None
        return {
            "active_streams": [],
            "count": 0,
            "uris": {},
            "rtsp_server": f"rtsp://localhost:{RTSP_SERVER_PORT}",
            "rtsp_paths": {}
        }


def run_loop():
    """No-op - subprocess manages its own loop"""
    pass
