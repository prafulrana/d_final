import gi
gi.require_version('Gst', '1.0')
from gi.repository import Gst
import pyds
import numpy as np
import ctypes

# COCO class names
COCO_CLASSES = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
    "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
    "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack",
    "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball",
    "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket",
    "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
    "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair",
    "couch", "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse",
    "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator",
    "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"
]

def osd_sink_pad_buffer_probe(pad, info, u_data):
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

        frame_width = frame_meta.source_frame_width
        frame_height = frame_meta.source_frame_height

        # Access tensor metadata
        l_user = frame_meta.frame_user_meta_list
        while l_user is not None:
            try:
                user_meta = pyds.NvDsUserMeta.cast(l_user.data)
            except StopIteration:
                break

            if user_meta.base_meta.meta_type == pyds.NvDsMetaType.NVDSINFER_TENSOR_OUTPUT_META:
                tensor_meta = pyds.NvDsInferTensorMeta.cast(user_meta.user_meta_data)

                # Find [N, 6] tensor
                for i in range(tensor_meta.num_output_layers):
                    layer = pyds.get_nvds_LayerInfo(tensor_meta, i)
                    dims = layer.inferDims

                    if dims.numDims == 2 and dims.d[1] == 6:
                        N = dims.d[0]
                        # Get base pointer array, then offset by layer index
                        ptr_array = pyds.get_ptr(tensor_meta.out_buf_ptrs_host)
                        ptr = ctypes.cast(ptr_array, ctypes.POINTER(ctypes.c_void_p))[i]
                        if not ptr:
                            continue

                        # Parse [N,6]: [x1, y1, x2, y2, score, class]
                        # Create flat array then reshape
                        flat_data = np.ctypeslib.as_array(
                            ctypes.cast(ptr, ctypes.POINTER(ctypes.c_float)),
                            shape=(N * 6,)
                        )
                        detections = flat_data.reshape(N, 6)

                        conf_threshold = 0.25

                        # With maintain-aspect-ratio=0, DeepStream does simple resize
                        # Model outputs coordinates in 640x640 space, scale to frame dims
                        net_w, net_h = 640, 640
                        scale_x = frame_width / net_w
                        scale_y = frame_height / net_h

                        for j in range(N):
                            x1, y1, x2, y2, score, cls = detections[j, :]
                            if score < conf_threshold:
                                continue

                            # Scale from 640x640 network space to frame space
                            x1 = x1 * scale_x
                            y1 = y1 * scale_y
                            x2 = x2 * scale_x
                            y2 = y2 * scale_y

                            # Clamp coordinates
                            x1 = max(0, min(frame_width - 1, x1))
                            y1 = max(0, min(frame_height - 1, y1))
                            x2 = max(0, min(frame_width - 1, x2))
                            y2 = max(0, min(frame_height - 1, y2))

                            # Create bounding box
                            obj_meta = pyds.nvds_acquire_obj_meta_from_pool(batch_meta)
                            obj_meta.class_id = int(cls)
                            obj_meta.confidence = float(score)
                            obj_meta.rect_params.left = float(x1)
                            obj_meta.rect_params.top = float(y1)
                            obj_meta.rect_params.width = float(x2 - x1)
                            obj_meta.rect_params.height = float(y2 - y1)
                            obj_meta.rect_params.border_width = 2
                            obj_meta.rect_params.border_color.set(0.0, 1.0, 0.0, 1.0)

                            # Text params with class name
                            class_name = COCO_CLASSES[int(cls)] if int(cls) < len(COCO_CLASSES) else f"ID{int(cls)}"
                            obj_meta.text_params.display_text = f"{class_name} {score:.2f}"
                            obj_meta.text_params.x_offset = int(x1)
                            obj_meta.text_params.y_offset = max(0, int(y1) - 10)
                            obj_meta.text_params.font_params.font_name = "Serif"
                            obj_meta.text_params.font_params.font_size = 10
                            obj_meta.text_params.font_params.font_color.set(1.0, 1.0, 1.0, 1.0)
                            obj_meta.text_params.set_bg_clr = 1
                            obj_meta.text_params.text_bg_clr.set(0.0, 1.0, 0.0, 1.0)

                            pyds.nvds_add_obj_meta_to_frame(frame_meta, obj_meta, None)

                        break

            try:
                l_user = l_user.next
            except StopIteration:
                break

        try:
            l_frame = l_frame.next
        except StopIteration:
            break

    return Gst.PadProbeReturn.OK
