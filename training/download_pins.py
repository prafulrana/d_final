#!/usr/bin/env python3
"""
Download Roboflow Bowling Pin Detection Dataset
Dataset: https://universe.roboflow.com/lsc-kik8c/bowling-pin-detection
"""

from roboflow import Roboflow
import os

# Initialize Roboflow
rf = Roboflow(api_key="VTC3lDY3driIqWesEiRM")
project = rf.workspace("lsc-kik8c").project("bowling-pin-detection")
dataset = project.version(4).download("yolov8")

print(f"\n✓ Dataset downloaded to: {dataset.location}")
