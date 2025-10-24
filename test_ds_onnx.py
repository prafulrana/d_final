from ultralytics import YOLO

print("=== Testing DeepStream-Yolo Exported ONNX ===")
onnx_model = YOLO("/data/training/bowling/bowling_640_blur_v3/weights/best.pt.onnx", task='detect')
results = onnx_model.predict("/data/asd3.jpg", imgsz=640, conf=0.1, verbose=True, save=True, project="/data", name="ds_onnx_test")

for r in results:
    print(f"\nDeepStream-ONNX Detections: {len(r.boxes)}")
    for i, box in enumerate(r.boxes):
        conf = float(box.conf[0])
        cls = int(box.cls[0])
        xyxy = box.xyxy[0].cpu().numpy()
        print(f"  [{i}] conf={conf:.4f} cls={cls} bbox=[{xyxy[0]:.1f}, {xyxy[1]:.1f}, {xyxy[2]:.1f}, {xyxy[3]:.1f}]")
