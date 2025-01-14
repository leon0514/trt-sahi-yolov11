# TRT-SAHI-YOLOv11

## 项目简介

`TRT-SAHI-YOLOv11` 是一个基于 **SAHI** 图像切割和 **TensorRT** 推理引擎的目标检测系统。该项目结合了高效的图像预处理与加速推理技术，旨在提供快速、精准的目标检测能力。通过切割大图像成多个小块进行推理，并应用非极大值抑制（NMS）来优化检测结果，最终实现对物体的精确识别。

## 功能特性

1. **SAHI 图像切割**  
   利用 CUDA 实现 SAHI的功能将输入图像切割成多个小块，支持重叠切割，以提高目标检测的准确性，特别是在边缘和密集物体区域。

2. **TensorRT 推理**  
   使用 `TensorRT` 进行深度学习模型推理加速。
   目前支持`TensorRT8` 和 `TensorRT10` API


## 注意事项
1. 模型需要是动态batch的
2. 如果模型切割后的数量大于batch的最大数量会导致无法推理

## 使用
```C++
cv::Mat image = cv::imread("inference/persons.jpg");
// cv::Mat image = cv::imread("6.jpg");
auto yolo = yolov11::load("yolov8n.transd.engine");
if (yolo == nullptr) return;
auto objs = yolo->forwardAuto(tensor::cvimg(image));
printf("objs size : %d\n", objs.size());
// OUTPUT
/*
------------------------------------------------------
TensorRT-Engine 🌱 is Dynamic Shape model
Inputs: 1
	0.images : {-1 x 3 x 640 x 640} [float32]
Outputs: 1
	0.output0 : {-1 x 8400 x 84} [float32]
------------------------------------------------------
------------------------------------------------------
CUDA SAHI CROP IMAGE ✂️ 
Slice width                : 784
Slice Height               : 1068
Overlap width  ratio       : 0.800000
Overlap height ratio       : 0.800000
Number of horizontal cuts  : 6
Number of vertical cuts    : 1
------------------------------------------------------
objs size : 39
Save result to Yolo-result.jpg, 39 objects
*/
```

## 对比
<div align="center">
   <img src="https://github.com/leon0514/trt-sahi-yolov11/blob/main/workspace/result/sliced.jpg?raw=true" width="45%"/>
   <img src="https://github.com/leon0514/trt-sahi-yolov11/blob/main/workspace/result/no_sliced.jpg?raw=true" width="45%"/>
</div>

## TensoRT8 API支持
在`Makefile`中通过 `TRT_VERSION`来控制编译哪个版本的tensorrt封装文件

## TODO

- [x] **NMS 实现**：完成 所有子图的 NMS 处理逻辑，去除冗余框。
- [x] **Tensorrt8支持**：目前是使用的tensorrt10的API
- [ ] **更多模型支持**：添加对其他 YOLO 模型版本的支持。目前支持YOLOv11/yolov8

