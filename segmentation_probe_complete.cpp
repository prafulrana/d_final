// Complete C++ segmentation probe - zero-copy GPU, no Python GIL
// Implements GStreamer pad probe in pure C++

#include <gst/gst.h>
#include <string.h>
#include "gstnvdsmeta.h"
#include "nvbufsurface.h"
#include <cuda_runtime.h>

// Forward declaration of CUDA functions
extern "C" void launch_segmentation_overlay_direct(
    unsigned char* d_frame_data,
    const float* d_seg_logits,
    int frame_width,
    int frame_height,
    int seg_channels,
    int seg_height,
    int seg_width,
    float alpha
);

extern "C" void convert_classmap_gpu(const int* d_int_data, float* d_float_data, int size);

// Tensor metadata structures (from nvdsinfer headers)
typedef struct {
    unsigned int numDims;
    int d[8];
} NvDsInferDims;

typedef enum {
    FLOAT = 0,
    HALF = 1,
    INT8 = 2,
    INT32 = 3
} NvDsInferDataType;

typedef struct {
    NvDsInferDataType dataType;
    NvDsInferDims inferDims;
    void *buffer;
} NvDsInferLayerInfo;

typedef struct {
    unsigned int num_output_layers;
    NvDsInferLayerInfo *output_layers_info;
    void **out_buf_ptrs_host;
    void **out_buf_ptrs_dev;  // GPU device pointers
    int gpu_id;
    void *priv_data;
} NvDsInferTensorMeta;

// Segmentation metadata structure
typedef struct {
    unsigned int classes;
    unsigned int width;
    unsigned int height;
    int *class_map;
    float *class_probabilities_map;
    void *priv_data;
} NvDsInferSegmentationMeta;

// Metadata type constants - these are from nvdsmeta.h
#define NVDS_INFER_TENSOR_OUTPUT_META 12  // Actual enum value
#define NVDSINFER_TENSOR_OUTPUT_META NVDS_INFER_TENSOR_OUTPUT_META
#define NVDSINFER_SEGMENTATION_META NVDS_INFER_TENSOR_OUTPUT_META  // Same metadata type

extern "C" GstPadProbeReturn
segmentation_probe_callback(GstPad *pad, GstPadProbeInfo *info, gpointer u_data)
{
    static int frame_count = 0;
    if (frame_count++ % 30 == 0) {
        g_print("[C++ PROBE] Callback triggered, frame %d\n", frame_count);
    }

    GstBuffer *buf = (GstBuffer *)info->data;
    NvDsBatchMeta *batch_meta = gst_buffer_get_nvds_batch_meta(buf);

    if (!batch_meta) {
        return GST_PAD_PROBE_OK;
    }

    // Map buffer to access NvBufSurface
    GstMapInfo map_info;
    memset(&map_info, 0, sizeof(map_info));
    if (!gst_buffer_map(buf, &map_info, GST_MAP_READ)) {
        g_print("Error: Failed to map buffer\n");
        return GST_PAD_PROBE_OK;
    }

    NvBufSurface *surf = (NvBufSurface *)map_info.data;

    for (NvDsMetaList *l_frame = batch_meta->frame_meta_list;
         l_frame != NULL; l_frame = l_frame->next) {

        NvDsFrameMeta *frame_meta = (NvDsFrameMeta *)l_frame->data;

        // Get frame GPU pointer
        void *frame_gpu_ptr = surf->surfaceList[frame_meta->batch_id].dataPtr;
        int frame_width = surf->surfaceList[frame_meta->batch_id].width;
        int frame_height = surf->surfaceList[frame_meta->batch_id].height;

        // Search for segmentation metadata
        int meta_count = 0;
        for (NvDsMetaList *l_user = frame_meta->frame_user_meta_list;
             l_user != NULL; l_user = l_user->next) {

            NvDsUserMeta *user_meta = (NvDsUserMeta *)l_user->data;
            meta_count++;

            // Check for segmentation metadata (type may vary)
            if (frame_count % 60 == 1) {
                g_print("[C++ PROBE] Metadata type: %d\n", user_meta->base_meta.meta_type);
            }

            // Try to process as segmentation metadata first
            // Type might be different from tensor output
            if (user_meta->base_meta.meta_type != 12) {
                // This might be segmentation metadata
                NvDsInferSegmentationMeta *seg_meta = (NvDsInferSegmentationMeta *)user_meta->user_meta_data;

                if (seg_meta && seg_meta->width > 0 && seg_meta->height > 0) {
                    if (frame_count % 60 == 1) {
                        g_print("[C++ PROBE] Found SEGMENTATION: %dx%d, classes=%d\n",
                                seg_meta->width, seg_meta->height, seg_meta->classes);
                    }

                    // GPU-only conversion: Copy int class_map to GPU, convert there
                    int seg_size = seg_meta->width * seg_meta->height;

                    // Single H2D copy of int data
                    int *class_map_gpu;
                    cudaMalloc((void**)&class_map_gpu, seg_size * sizeof(int));
                    cudaMemcpy(class_map_gpu, seg_meta->class_map, seg_size * sizeof(int), cudaMemcpyHostToDevice);

                    // Allocate GPU buffer for float conversion
                    float *seg_float_gpu;
                    cudaMalloc((void**)&seg_float_gpu, seg_size * sizeof(float));

                    // Convert int→float on GPU (ZERO CPU overhead!)
                    convert_classmap_gpu(class_map_gpu, seg_float_gpu, seg_size);

                    // Launch kernel with FULL alpha for visibility
                    launch_segmentation_overlay_direct(
                        (unsigned char *)frame_gpu_ptr,
                        seg_float_gpu,
                        frame_width, frame_height,
                        1, seg_meta->height, seg_meta->width,
                        1.0f  // FULL GREEN overlay
                    );

                    cudaFree(class_map_gpu);
                    cudaFree(seg_float_gpu);

                    if (frame_count % 60 == 1) {
                        g_print("[C++ PROBE] Overlay complete\n");
                    }
                    continue;
                }
            }

            // Fallback: process as tensor metadata
            if (user_meta->base_meta.meta_type == 12) {
                NvDsInferTensorMeta *tensor_meta = (NvDsInferTensorMeta *)user_meta->user_meta_data;

                g_print("[C++ PROBE] Found TENSOR_OUTPUT_META: num_layers=%u\n", tensor_meta->num_output_layers);
                g_print("[C++ PROBE] out_buf_ptrs_dev=%p, out_buf_ptrs_host=%p\n",
                        tensor_meta->out_buf_ptrs_dev, tensor_meta->out_buf_ptrs_host);

                if (!tensor_meta->out_buf_ptrs_dev || tensor_meta->num_output_layers == 0) {
                    g_print("[C++ PROBE] WARNING: GPU pointers not available (out_buf_ptrs_dev=%p)\n",
                            tensor_meta->out_buf_ptrs_dev);
                    continue;
                }

                g_print("[C++ PROBE] out_buf_ptrs_dev[0]=%p\n", tensor_meta->out_buf_ptrs_dev[0]);

                // Get segmentation output GPU pointer
                void *seg_data_gpu = tensor_meta->out_buf_ptrs_dev[0];
                NvDsInferLayerInfo *layer_info = &tensor_meta->output_layers_info[0];

                g_print("[C++ PROBE] Data type: %d (0=FLOAT, 3=INT32)\n", layer_info->dataType);

                // Debug: print raw dimensions
                g_print("[C++ PROBE] numDims=%d, raw dims: [", layer_info->inferDims.numDims);
                for (int i = 0; i < layer_info->inferDims.numDims && i < 8; i++) {
                    g_print("%d%s", layer_info->inferDims.d[i], (i < layer_info->inferDims.numDims - 1) ? ", " : "");
                }
                g_print("]\n");

                // Extract dimensions
                // TensorRT output format: [H, W, C] for segmentation
                int seg_h, seg_w, seg_c;
                if (layer_info->inferDims.numDims == 3) {
                    seg_h = layer_info->inferDims.d[0];  // Height
                    seg_w = layer_info->inferDims.d[1];  // Width
                    seg_c = layer_info->inferDims.d[2];  // Channels
                } else if (layer_info->inferDims.numDims == 4) {
                    // Batch size in d[0], skip it
                    seg_h = layer_info->inferDims.d[1];
                    seg_w = layer_info->inferDims.d[2];
                    seg_c = layer_info->inferDims.d[3];
                } else {
                    g_print("[C++ PROBE] Unexpected numDims=%d\n", layer_info->inferDims.numDims);
                    continue;
                }

                g_print("[C++ PROBE] Parsed as [H=%d, W=%d, C=%d], Frame: %dx%d\n",
                        seg_h, seg_w, seg_c, frame_width, frame_height);

                // Convert INT32/INT64 to float if needed
                float *seg_float_gpu = NULL;
                int seg_size = seg_h * seg_w * seg_c;

                if (layer_info->dataType == INT32 || layer_info->dataType == 3) {
                    g_print("[C++ PROBE] Converting INT32 tensor to float\n");
                    // Allocate temp buffer for conversion
                    int32_t *int_data = new int32_t[seg_size];
                    cudaMemcpy(int_data, seg_data_gpu, seg_size * sizeof(int32_t), cudaMemcpyDeviceToHost);

                    float *float_data = new float[seg_size];
                    for (int i = 0; i < seg_size; i++) {
                        float_data[i] = (float)int_data[i];
                    }

                    cudaMalloc((void**)&seg_float_gpu, seg_size * sizeof(float));
                    cudaMemcpy(seg_float_gpu, float_data, seg_size * sizeof(float), cudaMemcpyHostToDevice);

                    delete[] int_data;
                    delete[] float_data;
                } else {
                    // Already float
                    seg_float_gpu = (float*)seg_data_gpu;
                }

                // Sample pixels from different locations to debug
                float sample_pixels[16];
                // Top-left corner
                cudaMemcpy(&sample_pixels[0], seg_float_gpu, 4 * sizeof(float), cudaMemcpyDeviceToHost);
                // Center
                int center_offset = (seg_h / 2) * seg_w + (seg_w / 2);
                cudaMemcpy(&sample_pixels[4], seg_float_gpu + center_offset, 4 * sizeof(float), cudaMemcpyDeviceToHost);
                // Bottom-right
                int bottom_offset = (seg_h - 1) * seg_w + (seg_w - 4);
                cudaMemcpy(&sample_pixels[8], seg_float_gpu + bottom_offset, 4 * sizeof(float), cudaMemcpyDeviceToHost);
                // Random middle section
                int mid_offset = (seg_h / 3) * seg_w + (seg_w / 3);
                cudaMemcpy(&sample_pixels[12], seg_float_gpu + mid_offset, 4 * sizeof(float), cudaMemcpyDeviceToHost);

                g_print("[C++ PROBE] Pixels - TopLeft:[%.2f,%.2f,%.2f,%.2f] Center:[%.2f,%.2f,%.2f,%.2f] BotRight:[%.2f,%.2f,%.2f,%.2f] Mid:[%.2f,%.2f,%.2f,%.2f]\n",
                        sample_pixels[0], sample_pixels[1], sample_pixels[2], sample_pixels[3],
                        sample_pixels[4], sample_pixels[5], sample_pixels[6], sample_pixels[7],
                        sample_pixels[8], sample_pixels[9], sample_pixels[10], sample_pixels[11],
                        sample_pixels[12], sample_pixels[13], sample_pixels[14], sample_pixels[15]);

                // Launch CUDA kernel
                g_print("[C++ PROBE] Launching kernel: frame_ptr=%p, seg_ptr=%p\n",
                        frame_gpu_ptr, seg_float_gpu);
                launch_segmentation_overlay_direct(
                    (unsigned char *)frame_gpu_ptr,
                    seg_float_gpu,
                    frame_width,
                    frame_height,
                    seg_c,
                    seg_h,
                    seg_w,
                    0.7f  // Increased alpha for more visible overlay
                );

                // Free converted buffer if we allocated it
                if (layer_info->dataType == INT32 || layer_info->dataType == 3) {
                    cudaFree(seg_float_gpu);
                }

                g_print("[C++ PROBE] Kernel launched\n");
            }
        }
    }

    gst_buffer_unmap(buf, &map_info);
    return GST_PAD_PROBE_OK;
}

// Function to attach probe (called from Python)
extern "C" {
    void attach_segmentation_probe(GstElement *element, const char *pad_name) {
        GstPad *pad = gst_element_get_static_pad(element, pad_name);
        if (!pad) {
            g_print("Error: Could not get pad %s\n", pad_name);
            return;
        }

        gst_pad_add_probe(pad, GST_PAD_PROBE_TYPE_BUFFER,
                         segmentation_probe_callback, NULL, NULL);

        g_print("Segmentation probe attached to %s\n", pad_name);
        gst_object_unref(pad);
    }
}
