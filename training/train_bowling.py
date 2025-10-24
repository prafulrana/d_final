#!/usr/bin/env python3
"""
Bowling Ball Detection Model Training Script
Trains YOLOv8 models with consistent, reproducible parameters
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path
from datetime import datetime

# Default hyperparameters (based on working 640 model @ batch=16)
DEFAULT_CONFIG = {
    'model': 'yolov8n.pt',
    'data': '/data/bowling-ball-1/data.yaml',
    'epochs': 100,
    'batch': 16,
    'imgsz': 640,
    'device': 0,
    'workers': 4,
    'patience': 10,
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
        description='Train YOLOv8 bowling ball detection model',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Model parameters
    parser.add_argument('--model', default=DEFAULT_CONFIG['model'],
                       help='Base model to train from')
    parser.add_argument('--imgsz', type=int, default=DEFAULT_CONFIG['imgsz'],
                       help='Training image size (640 or 1280 recommended)')
    parser.add_argument('--batch', type=int, default=DEFAULT_CONFIG['batch'],
                       help='Batch size (16 recommended for best quality)')

    # Training parameters
    parser.add_argument('--epochs', type=int, default=DEFAULT_CONFIG['epochs'],
                       help='Number of training epochs')
    parser.add_argument('--patience', type=int, default=DEFAULT_CONFIG['patience'],
                       help='Early stopping patience')
    parser.add_argument('--device', type=int, default=DEFAULT_CONFIG['device'],
                       help='GPU device ID')
    parser.add_argument('--workers', type=int, default=DEFAULT_CONFIG['workers'],
                       help='Number of dataloader workers')

    # Output
    parser.add_argument('--name', type=str, default=None,
                       help='Experiment name (auto-generated if not provided)')
    parser.add_argument('--export-onnx', action='store_true', default=True,
                       help='Export to ONNX after training')

    return parser.parse_args()

def run_training(args):
    """Execute YOLO training with specified parameters"""

    # Auto-generate experiment name if not provided
    if args.name is None:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        args.name = f'bowling_{args.imgsz}_b{args.batch}_{timestamp}'

    print(f"\n{'='*60}")
    print(f"BOWLING BALL DETECTION - TRAINING")
    print(f"{'='*60}")
    print(f"Model:       {args.model}")
    print(f"Resolution:  {args.imgsz}x{args.imgsz}")
    print(f"Batch size:  {args.batch}")
    print(f"Epochs:      {args.epochs}")
    print(f"Device:      GPU {args.device}")
    print(f"Experiment:  {args.name}")
    print(f"{'='*60}\n")

    # Build training command
    cmd = [
        'yolo', 'train',
        f'model={args.model}',
        f'data={DEFAULT_CONFIG["data"]}',
        f'epochs={args.epochs}',
        f'imgsz={args.imgsz}',
        f'batch={args.batch}',
        f'device={args.device}',
        f'workers={args.workers}',
        f'patience={args.patience}',
        f'project=/data/bowling',
        f'name={args.name}',
        f'lr0={DEFAULT_CONFIG["lr0"]}',
        f'lrf={DEFAULT_CONFIG["lrf"]}',
        f'momentum={DEFAULT_CONFIG["momentum"]}',
        f'weight_decay={DEFAULT_CONFIG["weight_decay"]}',
        f'mosaic={DEFAULT_CONFIG["mosaic"]}',
        f'close_mosaic={DEFAULT_CONFIG["close_mosaic"]}',
    ]

    print(f"Command: {' '.join(cmd)}\n")

    # Run training
    try:
        subprocess.run(cmd, check=True)
        print(f"\n{'='*60}")
        print(f"✓ Training completed successfully!")
        print(f"{'='*60}\n")
        return True
    except subprocess.CalledProcessError as e:
        print(f"\n{'='*60}")
        print(f"✗ Training failed with error code {e.returncode}")
        print(f"{'='*60}\n")
        return False

def export_onnx(args):
    """Export trained model to ONNX format"""

    weights_path = Path(f'/data/bowling/{args.name}/weights/best.pt')

    if not weights_path.exists():
        print(f"✗ Weights not found at {weights_path}")
        return False

    print(f"\n{'='*60}")
    print(f"EXPORTING TO ONNX")
    print(f"{'='*60}")
    print(f"Weights:     {weights_path}")
    print(f"Output size: {args.imgsz}x{args.imgsz}")
    print(f"{'='*60}\n")

    # Export using DeepStream-Yolo script
    cmd = [
        'python3',
        './DeepStream-Yolo/utils/export_yoloV8.py',
        '-w', str(weights_path),
        '--dynamic',
        '-s', str(args.imgsz)
    ]

    print(f"Command: {' '.join(cmd)}\n")

    try:
        # Use /data when running in docker, or current directory otherwise
        export_cwd = '/data' if os.path.exists('/data/DeepStream-Yolo') else os.path.dirname(__file__)
        subprocess.run(cmd, check=True, cwd=export_cwd)
        onnx_path = weights_path.with_suffix('.pt.onnx')
        print(f"\n{'='*60}")
        print(f"✓ ONNX export completed!")
        print(f"Output: {onnx_path}")
        print(f"{'='*60}\n")

        # Show suggested deployment command
        print(f"To deploy to DeepStream:")
        print(f"  cp {onnx_path} /root/d_final/models/bowling_{args.imgsz}.onnx")
        print(f"  rm /root/d_final/models/bowling_{args.imgsz}_b1_gpu0_fp16.engine")
        print(f"  docker restart ds-s3\n")

        return True
    except subprocess.CalledProcessError as e:
        print(f"\n{'='*60}")
        print(f"✗ ONNX export failed with error code {e.returncode}")
        print(f"{'='*60}\n")
        return False

def main():
    args = parse_args()

    # Run training
    success = run_training(args)

    if not success:
        sys.exit(1)

    # Export to ONNX if requested
    if args.export_onnx:
        export_success = export_onnx(args)
        if not export_success:
            sys.exit(1)

    print("\n✓ All tasks completed successfully!\n")

if __name__ == '__main__':
    main()
