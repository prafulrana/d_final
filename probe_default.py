import gi
gi.require_version('Gst', '1.0')
from gi.repository import Gst

def osd_sink_pad_buffer_probe(pad, info, u_data):
    """Default probe - no custom processing"""
    return Gst.PadProbeReturn.OK
