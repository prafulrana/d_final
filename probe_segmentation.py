import gi
gi.require_version('Gst', '1.0')
from gi.repository import Gst
import pyds
import ctypes
import numpy as np

# Load CUDA library
cuda_lib = ctypes.CDLL('/app/segmentation_overlay.so')
cuda_lib.launch_segmentation_overlay.argtypes = [
    ctypes.c_void_p,  # d_frame_data
    ctypes.c_void_p,  # d_seg_mask
    ctypes.c_int,     # frame_width
    ctypes.c_int,     # frame_height
    ctypes.c_int,     # seg_width
    ctypes.c_int,     # seg_height
    ctypes.c_float,   # alpha
    ctypes.c_void_p   # stream
]

def osd_sink_pad_buffer_probe(pad, info, u_data):
    """Apply green segmentation overlay using CUDA"""
    gst_buffer = info.get_buffer()
    if not gst_buffer:
        return Gst.PadProbeReturn.OK

    batch_meta = pyds.gst_buffer_get_nvds_batch_meta(hash(gst_buffer))
    l_frame = batch_meta.frame_meta_list

    while l_frame is not None:
        try:
            frame_meta = pyds.NvDsFrameMeta.cast(l_frame.data)
        except StopIteration:
            break

        # Find segmentation metadata
        l_user = frame_meta.frame_user_meta_list
        while l_user is not None:
            try:
                user_meta = pyds.NvDsUserMeta.cast(l_user.data)
            except StopIteration:
                break

            if user_meta.base_meta.meta_type == pyds.NvDsMetaType.NVDSINFER_SEGMENTATION_META:
                seg_meta = pyds.NvDsInferSegmentationMeta.cast(user_meta.user_meta_data)

                # Get surface buffer
                n_frame = pyds.get_nvds_buf_surface(hash(gst_buffer), frame_meta.batch_id)
                if n_frame is not None:
                    # Get GPU pointers
                    frame_ptr = n_frame.__array_interface__['data'][0]

                    # Get segmentation mask data
                    masks = pyds.get_segmentation_masks(seg_meta)
                    mask_array = np.array(masks, copy=True, order='C')

                    # Convert int32 to uint8 for CUDA kernel
                    mask_array = mask_array.astype(np.uint8)

                    # Allocate GPU memory for mask and copy
                    import pycuda.driver as cuda
                    import pycuda.autoinit

                    mask_gpu = cuda.mem_alloc(mask_array.nbytes)
                    cuda.memcpy_htod(mask_gpu, mask_array)

                    # Launch CUDA kernel
                    cuda_lib.launch_segmentation_overlay(
                        frame_ptr,
                        int(mask_gpu),
                        frame_meta.source_frame_width,
                        frame_meta.source_frame_height,
                        seg_meta.width,
                        seg_meta.height,
                        ctypes.c_float(0.5),  # 50% alpha
                        None  # default stream
                    )

                    # Free GPU memory
                    mask_gpu.free()

            try:
                l_user = l_user.next
            except StopIteration:
                break

        try:
            l_frame = l_frame.next
        except StopIteration:
            break

    return Gst.PadProbeReturn.OK
