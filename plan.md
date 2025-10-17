# YOLO Detection Pipeline Plan

## Project Goal
Add s4 YOLO object detection pipeline to detect people (COCO dataset), then switch to custom bowling pin detection model from Roboflow.

## Milestones

### Milestone 1: YOLO COCO People Detection (Baseline)
**Goal:** Get YOLOv8 running on s4 with COCO classes (person detection)

**Tasks:**
- [x] Create plan.md
- [ ] Download YOLOv8n ONNX model (COCO 80 classes)
- [ ] Create DeepStream config for YOLOv8 detection (pgie_yolov8_coco.txt)
- [ ] Update up.sh to run s4 YOLO pipeline (stop s5 segmentation)
- [ ] Test people detection with bounding boxes
- [ ] Verify performance (<5% CPU, real-time inference)

**Success Criteria:**
- s4 pipeline detects people with bounding boxes
- Real-time performance (30fps)
- View at http://34.14.140.30:8889/s4/

---

### Milestone 2: Roboflow Model Export
**Goal:** Export bowling pin detection model from Roboflow as ONNX

**Tasks:**
- [ ] Research Roboflow export options (Ultralytics, ONNX formats)
- [ ] Export model from https://universe.roboflow.com/lsc-kik8c/bowling-pin-detection
- [ ] Verify ONNX model structure (input/output shapes, class count)
- [ ] Download to models/ directory
- [ ] Add to Git LFS

**Success Criteria:**
- ONNX file downloaded and verified
- Model metadata understood (1 class: bowling_pin)

---

### Milestone 3: ONNX Conversion & Optimization (if needed)
**Goal:** Convert/optimize bowling pin model for DeepStream 8.0 compatibility

**Tasks:**
- [ ] Test Roboflow ONNX directly in DS8
- [ ] If incompatible: Use DS8 container to convert with onnx-simplifier
- [ ] Verify TensorRT engine builds successfully
- [ ] Check inference output format matches DeepStream expectations

**Tools:**
- `onnx-simplifier` (inside DS8 container)
- `onnxruntime` for validation
- TensorRT engine builder (built-in to DeepStream nvinfer)

**Success Criteria:**
- ONNX model loads in DeepStream nvinfer
- TensorRT engine builds without errors
- Output format compatible (bounding boxes)

---

### Milestone 4: Bowling Pin Detection Config
**Goal:** Configure DeepStream for bowling pin detection

**Tasks:**
- [ ] Create pgie_bowling_pin.txt config
- [ ] Set correct num-detected-classes=1
- [ ] Configure label file (bowling_pin)
- [ ] Set network-type=0 (detector)
- [ ] Configure NMS threshold, confidence threshold
- [ ] Test bowling pin detection on sample video

**Success Criteria:**
- Bowling pins detected with bounding boxes
- Proper class label displayed
- No false positives on people/other objects

---

### Milestone 5: Pipeline Switching & Documentation
**Goal:** Finalize s4 YOLO pipeline and update docs

**Tasks:**
- [ ] Update up.sh to easily switch between segmentation (s5) and detection (s4)
- [ ] Update AGENTS.md with YOLO pipeline details
- [ ] Update STRUCTURE.md with new model files
- [ ] Update STANDARDS.md with YOLO config patterns
- [ ] Commit all changes to master/main
- [ ] Push model to LFS

**Success Criteria:**
- s4: Bowling pin detection
- s5: People segmentation (existing)
- Documentation complete and accurate

---

## Technical Notes

### YOLOv8 ONNX Requirements
- **Input:** 640x640 (or 416x416), RGB, NCHW format
- **Output:** Detection format (x, y, w, h, confidence, class_probs)
- **Preprocessing:** 1/255.0 normalization, no mean subtraction
- **Post-processing:** NMS (Non-Maximum Suppression) in DeepStream or custom parser

### Roboflow Export Options
1. **Ultralytics YOLOv8 Format** → Export as ONNX
2. **Direct ONNX Export** (if available)
3. **PyTorch → ONNX Conversion** (fallback)

### DeepStream YOLO Config Pattern
```ini
[property]
network-type=0                    # Detector
num-detected-classes=1            # Bowling pin only
onnx-file=/models/bowling_pin.onnx
network-mode=2                    # FP16
cluster-mode=2                    # DBSCAN clustering
nms-iou-threshold=0.45
pre-cluster-threshold=0.25        # Confidence threshold
```

### Git LFS Tracking
Already configured for `.onnx` files - just add and commit.

---

## Current Status
**Active:** Milestone 1 (YOLOv8 COCO baseline)
