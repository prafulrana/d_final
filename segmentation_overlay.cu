// CUDA kernel to apply green translucent overlay for segmentation masks
// Operates on RGBA format in NVMM buffers

extern "C" {

__global__ void apply_segmentation_overlay(
    unsigned char* frame_data,
    const unsigned char* seg_mask,
    int frame_width,
    int frame_height,
    int seg_width,
    int seg_height,
    float alpha
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= frame_width || y >= frame_height) return;

    // Map frame coordinates to segmentation mask coordinates
    int seg_x = (x * seg_width) / frame_width;
    int seg_y = (y * seg_height) / frame_height;
    int seg_idx = seg_y * seg_width + seg_x;

    // Check if this pixel is a person (class 1)
    if (seg_mask[seg_idx] == 1) {
        // RGBA format: frame_data is [R, G, B, A, R, G, B, A, ...]
        int pixel_idx = (y * frame_width + x) * 4;

        // Apply green overlay with alpha blending
        // Green color: R=0, G=255, B=0
        unsigned char orig_r = frame_data[pixel_idx + 0];
        unsigned char orig_g = frame_data[pixel_idx + 1];
        unsigned char orig_b = frame_data[pixel_idx + 2];

        // Alpha blend: new = original * (1-alpha) + overlay * alpha
        frame_data[pixel_idx + 0] = (unsigned char)(orig_r * (1.0f - alpha) + 0 * alpha);
        frame_data[pixel_idx + 1] = (unsigned char)(orig_g * (1.0f - alpha) + 255 * alpha);
        frame_data[pixel_idx + 2] = (unsigned char)(orig_b * (1.0f - alpha) + 0 * alpha);
        // Alpha channel unchanged: frame_data[pixel_idx + 3]
    }
}

// Host function to launch kernel
void launch_segmentation_overlay(
    unsigned char* d_frame_data,
    const unsigned char* d_seg_mask,
    int frame_width,
    int frame_height,
    int seg_width,
    int seg_height,
    float alpha,
    cudaStream_t stream
) {
    dim3 block(16, 16);
    dim3 grid((frame_width + block.x - 1) / block.x,
              (frame_height + block.y - 1) / block.y);

    apply_segmentation_overlay<<<grid, block, 0, stream>>>(
        d_frame_data, d_seg_mask,
        frame_width, frame_height,
        seg_width, seg_height,
        alpha
    );
}

} // extern "C"
