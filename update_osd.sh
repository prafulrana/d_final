#!/bin/bash
# Update OSD settings across s0_rtsp.py and file_s3_s4.txt from config/config_osd.txt
# Run this after editing config/config_osd.txt to sync settings everywhere

set -e

OSD_CONFIG="config/config_osd.txt"
PYTHON_SCRIPT="s0_rtsp.py"
S3S4_CONFIG="config/file_s3_s4.txt"

if [[ ! -f "$OSD_CONFIG" ]]; then
    echo "Error: $OSD_CONFIG not found"
    exit 1
fi

echo "Reading OSD settings from $OSD_CONFIG..."

# Parse config/config_osd.txt
display_text=$(grep "^display-text=" "$OSD_CONFIG" | cut -d'=' -f2)
display_bbox=$(grep "^display-bbox=" "$OSD_CONFIG" | cut -d'=' -f2)
gpu_id=$(grep "^gpu-id=" "$OSD_CONFIG" | cut -d'=' -f2)
border_width=$(grep "^border-width=" "$OSD_CONFIG" | cut -d'=' -f2)
text_size=$(grep "^text-size=" "$OSD_CONFIG" | cut -d'=' -f2)
text_color=$(grep "^text-color=" "$OSD_CONFIG" | cut -d'=' -f2)
text_bg_color=$(grep "^text-bg-color=" "$OSD_CONFIG" | cut -d'=' -f2)
font=$(grep "^font=" "$OSD_CONFIG" | cut -d'=' -f2)
show_clock=$(grep "^show-clock=" "$OSD_CONFIG" | cut -d'=' -f2)
nvbuf_memory_type=$(grep "^nvbuf-memory-type=" "$OSD_CONFIG" | cut -d'=' -f2)
process_mode=$(grep "^process-mode=" "$OSD_CONFIG" | cut -d'=' -f2)

echo "Updating $S3S4_CONFIG [osd] section..."

# Update s3/s4 config file [osd] section
sed -i '/^\[osd\]/,/^\[/ {
    /^enable=/! {
        /^\[osd\]/a\
enable=1\
display-text='"$display_text"'\
display-bbox='"$display_bbox"'\
gpu-id='"$gpu_id"'\
border-width='"$border_width"'\
text-size='"$text_size"'\
text-color='"$text_color"'\
text-bg-color='"$text_bg_color"'\
font='"$font"'\
show-clock='"$show_clock"'\
nvbuf-memory-type='"$nvbuf_memory_type"'\
process-mode='"$process_mode"'
        /^display-text=/d
        /^display-bbox=/d
        /^gpu-id=/d
        /^border-width=/d
        /^text-size=/d
        /^text-color=/d
        /^text-bg-color=/d
        /^font=/d
        /^show-clock=/d
        /^nvbuf-memory-type=/d
        /^process-mode=/d
    }
}' "$S3S4_CONFIG"

echo "Updating $PYTHON_SCRIPT OSD properties..."

# Update Python script OSD section
python3 << EOF
import re

with open('$PYTHON_SCRIPT', 'r') as f:
    content = f.read()

# Find OSD section and replace properties
osd_pattern = r'(# OSD.*?nvosd = Gst\.ElementFactory\.make\("nvdsosd", "onscreendisplay"\)\n)(.*?)(# nvvidconv after OSD)'

osd_props = f"""    nvosd.set_property('display-text', $display_text)
    nvosd.set_property('display-bbox', $display_bbox)
    nvosd.set_property('process-mode', $process_mode)
    nvosd.set_property('gpu-id', $gpu_id)

    """

content = re.sub(osd_pattern, r'\1' + osd_props + r'\3', content, flags=re.DOTALL)

with open('$PYTHON_SCRIPT', 'w') as f:
    f.write(content)
EOF

echo "✓ OSD settings synchronized!"
echo ""
echo "Settings applied:"
echo "  display-text: $display_text"
echo "  display-bbox: $display_bbox"
echo "  border-width: $border_width"
echo "  text-size: $text_size"
echo "  font: $font"
echo ""
echo "Run ./start.sh to apply changes"
