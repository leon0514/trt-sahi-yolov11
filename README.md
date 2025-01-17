# TRT-SAHI-YOLOv11

## 项目简介

**TRT-SAHI-YOLOv11** 是一个基于 **SAHI** 图像切割和 **TensorRT** 推理引擎的目标检测系统。该项目结合了高效的图像预处理与加速推理技术，旨在提供快速、精准的目标检测能力。通过切割大图像成多个小块进行推理，并应用非极大值抑制（NMS）来优化检测结果，最终实现对物体的精确识别。

## 功能特性

1. **SAHI 图像切割**  
   利用 CUDA 实现 **SAHI** 的功能将输入图像切割成多个小块，支持重叠切割，以提高目标检测的准确性，特别是在边缘和密集物体区域。

2. **TensorRT 推理**  
   使用 **TensorRT** 进行深度学习模型推理加速。
   目前支持 **TensorRT8** 和 **TensorRT10** API


## 注意事项
1. 模型需要是动态batch的
2. 如果模型切割后的数量大于batch的最大数量会导致无法推理
3. **TensorRT 10**在执行推理的时候需要指定输入和输出的名称，名称可以在netron中查看
   ```C++
   #ifdef TRT10
   if (!trt_->forward(std::unordered_map<std::string, const void *>{
            { "images", input_buffer_.gpu() }, 
            { "output0", bbox_predict_.gpu() }
      }, stream_))
   {
      printf("Failed to tensorRT forward.");
      return {};
   }
   #else
   std::vector<void *> bindings{input_buffer_.gpu(), bbox_output_device};
   if (!trt_->forward(bindings, stream)) 
   {
      printf("Failed to tensorRT forward.");
      return {};
   }
   #endif
   ```

## 使用
```C++
cv::Mat image = cv::imread("inference/persons.jpg");
auto yolo = yolo::load("helmetv5.engine", yolo::YoloType::YOLOV5);
if (yolo == nullptr) return;
auto objs = yolo->forward(tensor::cvimg(image));
printf("objs size : %d\n", objs.size());
```

## 对比
<div align="center">
   <img src="https://github.com/leon0514/trt-sahi-yolov11/blob/main/assert/sliced.jpg?raw=true" width="45%"/>
   <img src="https://github.com/leon0514/trt-sahi-yolov11/blob/main/assert/no_sliced.jpg?raw=true" width="45%"/>
</div>

## TensoRT8 API支持
在Makefile中通过 **TRT_VERSION** 来控制编译哪个版本的 **TensorRT** 封装文件

## 优化文字显示
目标检测模型识别到多个目标时，在图上显示文字可能会有重叠，导致类别置信度显示被遮挡。
优化了目标文字显示，尽可能改善遮挡情况    
详细说明见 [目标检测可视化文字重叠](https://www.jianshu.com/p/a6e289df4b90)
<div align="center">
   <img src="https://github.com/leon0514/trt-sahi-yolov11/blob/main/assert/sliced_text.jpg?raw=true" width="100%"/>
</div>

## 添加Python支持
使用pybind11封装程序
### 使用
```python
import trtsahiyolo
from trtsahiyolo import YoloType as YoloType
import cv2

instance = trtsahiyolo.TrtSahiYolo("phone.engine", YoloType.YOLOV5, 0)

frame = cv2.imread("test.jpg")

result = instance.autoSliceForward(frame)

print(result)
```

## TODO
- [x] **NMS 实现**：完成所有子图的 NMS 处理逻辑，去除冗余框。已完成
- [x] **TensorRT8支持**：完成使用 **TensorRT8** 和 **TensorRT10** API
- [x] **Python支持**：使用 **Pybind11** 封装，使用 **Pyton** 调用
- [ ] **更多模型支持**：添加对其他 YOLO 模型版本的支持。目前支持 **YOLOv11/YOLOv8/YOLOv5**

