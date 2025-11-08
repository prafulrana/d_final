#!/usr/bin/env python3
"""
Export EdgeTAM model to ONNX format for DeepStream deployment.
Single-model export with automatic grid prompts for bowling pins and balls.

Usage:
    python export_edgetam.py --output_dir /root/d_final/models
"""

import argparse
import torch
import onnx
from transformers import Sam2Processor, EdgeTamModel
import numpy as np


class EdgeTAMWrapper(torch.nn.Module):
    """Wrapper for EdgeTAM with fixed grid prompts for automatic segmentation."""

    def __init__(self, model, grid_size=16):
        super().__init__()
        self.vision_encoder = model.vision_encoder
        self.prompt_encoder = model.prompt_encoder
        self.mask_decoder = model.mask_decoder
        self.grid_size = grid_size

    def forward(self, pixel_values):
        """
        Forward pass with automatic grid prompts.

        Args:
            pixel_values: Input image tensor (1, 3, H, W)

        Returns:
            masks: Binary segmentation masks (1, grid_size^2, H, W)
            scores: IoU quality scores (1, grid_size^2)
        """
        batch_size, _, height, width = pixel_values.shape

        # Encode image
        vision_outputs = self.vision_encoder(pixel_values)
        image_embeddings = vision_outputs.last_hidden_state
        high_res_features = vision_outputs.fpn_hidden_states
        image_positional_embeddings = vision_outputs.fpn_position_encoding[0]

        # Generate grid of point prompts
        grid_points_x = torch.linspace(width * 0.1, width * 0.9, self.grid_size, device=pixel_values.device)
        grid_points_y = torch.linspace(height * 0.1, height * 0.9, self.grid_size, device=pixel_values.device)
        grid_x, grid_y = torch.meshgrid(grid_points_x, grid_points_y, indexing='xy')

        # Flatten to (1, grid_size^2, 2)
        coords = torch.stack([grid_x.flatten(), grid_y.flatten()], dim=-1).unsqueeze(0)

        # Input shape for coordinate normalization
        input_shape = torch.tensor([[height, width]], dtype=torch.long, device=pixel_values.device)

        # All foreground points (label=1)
        input_labels = torch.ones((batch_size, coords.shape[1]), dtype=torch.long, device=pixel_values.device)

        # Encode prompts (input_points is a tuple of coords and shape)
        sparse_embeddings, dense_embeddings = self.prompt_encoder(
            input_points=(coords, input_shape),
            input_labels=input_labels,
            input_boxes=None,
            input_masks=None
        )

        # Decode masks
        low_res_masks, iou_predictions, sam_tokens_out, object_score_logits = self.mask_decoder(
            image_embeddings=image_embeddings,
            image_positional_embeddings=image_positional_embeddings,
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            multimask_output=False,
            high_resolution_features=high_res_features
        )

        # Upsample masks to original size
        masks = torch.nn.functional.interpolate(
            low_res_masks,
            size=(height, width),
            mode='bilinear',
            align_corners=False
        )

        # Apply sigmoid to get probabilities
        masks = torch.sigmoid(masks)

        return masks.squeeze(2), iou_predictions.squeeze(1)


def main():
    parser = argparse.ArgumentParser(description='Export EdgeTAM to ONNX')
    parser.add_argument('--model_id', type=str, default='yonigozlan/EdgeTAM-hf',
                        help='HuggingFace model ID')
    parser.add_argument('--output_dir', type=str, default='/root/d_final/models',
                        help='Output directory for ONNX file')
    parser.add_argument('--image_size', type=int, default=1024,
                        help='Input image size')
    parser.add_argument('--grid_size', type=int, default=16,
                        help='Grid size for automatic prompts (16 = 256 points)')
    args = parser.parse_args()

    print("=" * 60)
    print("EdgeTAM ONNX Export Tool")
    print("Single-model export with automatic grid prompts")
    print("=" * 60)

    # Load model
    print(f"\nLoading model from {args.model_id}...")
    model = EdgeTamModel.from_pretrained(args.model_id)
    model.eval()
    print("✓ Model loaded")

    # Wrap model with grid prompts
    print(f"\nWrapping model with {args.grid_size}x{args.grid_size} grid prompts...")
    wrapped_model = EdgeTAMWrapper(model, grid_size=args.grid_size)
    wrapped_model.eval()

    # Create dummy input
    dummy_input = torch.randn(1, 3, args.image_size, args.image_size)

    # Test forward pass
    print("\nTesting forward pass...")
    with torch.no_grad():
        masks, scores = wrapped_model(dummy_input)
    print(f"✓ Forward pass successful")
    print(f"  Masks shape: {masks.shape}")
    print(f"  Scores shape: {scores.shape}")

    # Export to ONNX
    output_path = f"{args.output_dir}/edgetam_full.onnx"
    print(f"\nExporting to {output_path}...")

    torch.onnx.export(
        wrapped_model,
        dummy_input,
        output_path,
        export_params=True,
        opset_version=18,
        do_constant_folding=True,
        input_names=['pixel_values'],
        output_names=['masks', 'iou_scores'],
        dynamic_axes={
            'pixel_values': {0: 'batch', 2: 'height', 3: 'width'},
            'masks': {0: 'batch', 1: 'num_masks'},
            'iou_scores': {0: 'batch', 1: 'num_masks'}
        }
    )

    # Verify export
    print("\nVerifying ONNX model...")
    onnx_model = onnx.load(output_path)
    onnx.checker.check_model(onnx_model)
    print("✓ ONNX model verified")

    print("\n" + "=" * 60)
    print("Export complete!")
    print("=" * 60)
    print(f"\nModel: {output_path}")
    print(f"Grid size: {args.grid_size}x{args.grid_size} = {args.grid_size**2} masks")
    print("\nNext steps:")
    print("1. Convert to TensorRT: trtexec --onnx=edgetam_full.onnx --saveEngine=edgetam_full.engine --fp16")
    print("2. Create DeepStream config file")
    print("3. Implement custom parser for mask outputs")
    print("4. Integrate with live_stream_edgetam.c")


if __name__ == "__main__":
    main()
