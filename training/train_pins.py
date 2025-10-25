#!/usr/bin/env python3
"""
Bowling Pin Detection Model Training Script
Trains YOLOv8n for s3 stream (pin detection)
Uses same proven workflow as s2 (blur ball detection)
"""

import argparse
import sys
from datetime import datetime

# Training configuration (EXACT s2 match from bowling_640_blur_v3/args.yaml)
DEFAULT_CONFIG = {
    'model': 'yolov8n.pt',
    'data': '/data/Bowling-Pin-Detection--4/data.yaml',
    'epochs': 20,  # s2 used 20, not 100
    'batch': 16,
    'imgsz': 640,  # Same as s2 blur ball detection
    'device': 0,
    'workers': 4,
    'patience': 10,  # s2 used 10, not 20
    'optimizer': 'auto',
    'lr0': 0.01,
    'lrf': 0.01,
    'momentum': 0.937,
    'weight_decay': 0.0005,
    'mosaic': 1.0,
    'close_mosaic': 10,
}

def parse_args():
    parser = argparse.ArgumentParser(
        description='Train YOLOv8 bowling pin detection model for s3',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument('--imgsz', type=int, default=DEFAULT_CONFIG['imgsz'],
                       help='Training image size')
    parser.add_argument('--batch', type=int, default=DEFAULT_CONFIG['batch'],
                       help='Batch size')
    parser.add_argument('--epochs', type=int, default=DEFAULT_CONFIG['epochs'],
                       help='Number of epochs')
    parser.add_argument('--name', type=str, default=None,
                       help='Experiment name')

    return parser.parse_args()

def main():
    args = parse_args()

    # Auto-generate name
    if args.name is None:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        args.name = f'pins_{args.imgsz}_b{args.batch}_{timestamp}'

    print(f"\n{'='*60}")
    print(f"BOWLING PIN DETECTION - TRAINING (s3)")
    print(f"{'='*60}")
    print(f"Model:       {DEFAULT_CONFIG['model']}")
    print(f"Dataset:     {DEFAULT_CONFIG['data']}")
    print(f"Classes:     bowling-ball, bowling-pins, sweep board")
    print(f"Resolution:  {args.imgsz}x{args.imgsz}")
    print(f"Batch size:  {args.batch}")
    print(f"Epochs:      {args.epochs}")
    print(f"Device:      GPU {DEFAULT_CONFIG['device']}")
    print(f"Experiment:  {args.name}")
    print(f"{'='*60}\n")

    # Import here to avoid loading if just showing help
    from ultralytics import YOLO

    # Load model
    model = YOLO(DEFAULT_CONFIG['model'])

    # Train
    results = model.train(
        data=DEFAULT_CONFIG['data'],
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=DEFAULT_CONFIG['device'],
        workers=DEFAULT_CONFIG['workers'],
        patience=DEFAULT_CONFIG['patience'],
        project='/data/pins_training',
        name=args.name,
        lr0=DEFAULT_CONFIG['lr0'],
        lrf=DEFAULT_CONFIG['lrf'],
        momentum=DEFAULT_CONFIG['momentum'],
        weight_decay=DEFAULT_CONFIG['weight_decay'],
        mosaic=DEFAULT_CONFIG['mosaic'],
        close_mosaic=DEFAULT_CONFIG['close_mosaic'],
    )

    print(f"\n{'='*60}")
    print(f"✓ Training completed!")
    print(f"Best weights: /data/pins_training/{args.name}/weights/best.pt")
    print(f"{'='*60}\n")

    return 0

if __name__ == '__main__':
    sys.exit(main())
