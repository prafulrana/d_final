// Simplified zero-copy GPU segmentation overlay
// Takes GPU pointers directly from Python, no metadata navigation needed

#include <stdio.h>
#include <cuda_runtime.h>

extern "C" {

__global__ void apply_segmentation_overlay_direct(
    unsigned char* frame_data,      // RGBA frame [H, W, 4]
    const float* seg_data,          // Segmentation data [H, W, C] or [C, H, W]
    int frame_width,
    int frame_height,
    int seg_channels,
    int seg_height,
    int seg_width,
    float alpha
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= frame_width || y >= frame_height) return;

    // Map frame coordinates to segmentation mask coordinates
    int seg_x = (x * seg_width) / frame_width;
    int seg_y = (y * seg_height) / frame_height;

    int class_id;

    if (seg_channels == 1) {
        // Single channel - argmax already computed, pixel value IS the class ID
        // HWC layout: [H, W, C] -> index = y * width + x
        int idx = seg_y * seg_width + seg_x;
        float val = seg_data[idx];
        class_id = (int)(val + 0.5f);  // Round to nearest int
    } else {
        // Multi-channel - compute argmax across channels
        // HWC layout: [H, W, C] -> index = (y * width + x) * channels + c
        int max_class = 0;
        int base_idx = (seg_y * seg_width + seg_x) * seg_channels;
        float max_val = seg_data[base_idx];  // Channel 0

        for (int c = 1; c < seg_channels; c++) {
            float val = seg_data[base_idx + c];
            if (val > max_val) {
                max_val = val;
                max_class = c;
            }
        }
        class_id = max_class;
    }

    // Check if this pixel is person class (class 1)
    if (class_id == 1) {
        int pixel_idx = (y * frame_width + x) * 4;

        // Apply green overlay with alpha blending
        unsigned char orig_r = frame_data[pixel_idx + 0];
        unsigned char orig_g = frame_data[pixel_idx + 1];
        unsigned char orig_b = frame_data[pixel_idx + 2];

        frame_data[pixel_idx + 0] = (unsigned char)(orig_r * (1.0f - alpha) + 0 * alpha);
        frame_data[pixel_idx + 1] = (unsigned char)(orig_g * (1.0f - alpha) + 255 * alpha);
        frame_data[pixel_idx + 2] = (unsigned char)(orig_b * (1.0f - alpha) + 0 * alpha);
    }
}

// Kernel to convert int32 class map to float on GPU
__global__ void convert_int_to_float(const int* int_data, float* float_data, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float_data[idx] = (float)int_data[idx];
    }
}

// Host function callable from Python via ctypes
void launch_segmentation_overlay_direct(
    unsigned char* d_frame_data,
    const float* d_seg_logits,
    int frame_width,
    int frame_height,
    int seg_channels,
    int seg_height,
    int seg_width,
    float alpha
) {
    dim3 block(16, 16);
    dim3 grid((frame_width + block.x - 1) / block.x,
              (frame_height + block.y - 1) / block.y);

    apply_segmentation_overlay_direct<<<grid, block>>>(
        d_frame_data, d_seg_logits,
        frame_width, frame_height,
        seg_channels, seg_height, seg_width,
        alpha
    );

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("[CUDA ERROR] Kernel launch failed: %s\n", cudaGetErrorString(err));
    }

    err = cudaStreamSynchronize(0);
    if (err != cudaSuccess) {
        printf("[CUDA ERROR] Kernel execution failed: %s\n", cudaGetErrorString(err));
    }
}

// Convert int class map to float on GPU (zero CPU overhead)
void convert_classmap_gpu(const int* d_int_data, float* d_float_data, int size) {
    int threads = 256;
    int blocks = (size + threads - 1) / threads;
    convert_int_to_float<<<blocks, threads>>>(d_int_data, d_float_data, size);
    cudaStreamSynchronize(0);
}

} // extern "C"
