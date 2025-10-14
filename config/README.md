# DeepStream Inference Configurations

This directory contains various nvinfer configurations for different models and use cases.

## Available Configurations

### 1. pgie_resnet_traffic.txt (Current - In Use)
- **Model**: ResNet18 TrafficCamNet (ONNX)
- **Path**: `/models/resnet18_trafficcamnet_pruned.onnx`
- **Classes**: 4 (Vehicle, Person, RoadSign, TwoWheeler)
- **Input**: 960x544
- **Backend**: TensorRT FP16
- **Network Type**: Detector
- **Use Case**: Traffic camera object detection
- **Performance**: Fast, optimized for traffic scenes

### 2. pgie_peoplenet_int8.txt
- **Model**: ResNet34 PeopleNet (ONNX)
- **Path**: `/opt/nvidia/deepstream/deepstream/samples/models/peoplenet/resnet34_peoplenet_int8.onnx`
- **Classes**: 3 (Person, Bag, Face)
- **Input**: 544x960
- **Backend**: TensorRT INT8 (highest performance)
- **Network Type**: Detector
- **Use Case**: People detection with face/bag detection
- **Performance**: Fastest, INT8 quantized

### 3. pgie_city_segmentation.txt
- **Model**: CitySegFormer (ONNX)
- **Path**: `/opt/nvidia/deepstream/deepstream/samples/models/citysemsegformer/citysemsegformer.onnx`
- **Classes**: 19 (Road, Sidewalk, Building, etc.)
- **Backend**: TensorRT FP16
- **Network Type**: Semantic Segmentation
- **Use Case**: Urban scene segmentation
- **Performance**: Moderate, pixel-level segmentation

### 4. pgie_people_segmentation.txt
- **Model**: PeopleSegNet (Triton Inference Server)
- **Backend**: Triton
- **Network Type**: Semantic Segmentation
- **Use Case**: People semantic segmentation
- **Note**: Requires Triton Inference Server running

### 5. pgie_resnet_detector.txt
- **Model**: ResNet18 Primary Detector (ONNX)
- **Path**: `../../models/Primary_Detector/resnet18_trafficcamnet_pruned.onnx`
- **Classes**: 4 (Vehicle, Bicycle, Person, Roadsign)
- **Input**: 960x544
- **Backend**: TensorRT FP16
- **Network Type**: Detector
- **Use Case**: General object detection

### 6. pgie_default_test1.txt
- **Model**: ResNet18 TrafficCamNet (ONNX)
- **Path**: Relative to samples directory
- **Classes**: 4
- **Network Type**: Detector
- **Use Case**: Default test configuration

## Usage

To use a specific config, update `up.sh`:

```bash
docker run ... \
  python3 /app/app.py \
  -i rtsp://... \
  -o rtsp://... \
  -c /config/pgie_resnet_traffic.txt
```

Mount the config directory:
```bash
-v "$(pwd)/config":/config
```

## Available Models in Container

From `/opt/nvidia/deepstream/deepstream-8.0/samples/models/`:

1. **Primary_Detector**: ResNet18 TrafficCamNet (ONNX)
2. **SONYC_Audio_Classifier**: Audio classification
3. **Secondary_VehicleMake**: Vehicle make classification
4. **Secondary_VehicleTypes**: Vehicle type classification

## Model Path Patterns

Configs use relative paths. When copying to repo:
- Replace `../../models/` with absolute paths
- Or mount models directory at expected location
- Update `onnx-file`, `model-engine-file`, `labelfile-path`, `int8-calib-file`

## Performance Notes

- **TrafficCamNet**: Optimized for traffic, 960x544 input, FP16
- **PeopleNet**: INT8 quantized, fastest for people detection
- **Segmentation**: Higher compute, pixel-level masks
- **Batch Size**: Default 1 (single stream)
- **Network Mode**: 0=FP32, 1=INT8, 2=FP16
- **Engine Caching**: Enabled via `/models` volume mount

## Quick Comparison

| Config | Type | Precision | Speed | Use Case |
|--------|------|-----------|-------|----------|
| pgie_resnet_traffic | Detector | FP16 | Fast | Traffic scenes |
| pgie_peoplenet_int8 | Detector | INT8 | Fastest | People + face/bag |
| pgie_city_segmentation | Segmentation | FP16 | Moderate | Urban scenes |
| pgie_resnet_detector | Detector | FP16 | Fast | General objects |

## Testing Different Models

1. Copy model files to `/models` if needed
2. Update paths in config files to absolute paths
3. Rebuild Docker image if config embedded
4. Or mount `/config` and pass `-c /config/pgie_xxx.txt`
