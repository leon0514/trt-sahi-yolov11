#ifndef YOLOV11_HPP__
#define YOLOV11_HPP__
#include <vector>
#include "common/memory.hpp"
#include "common/image.hpp"

namespace yolov11
{

struct Box 
{
    float left, top, right, bottom, confidence;
    int class_label;

    Box() = default;
    Box(float left, float top, float right, float bottom, float confidence, int class_label)
        : left(left),
            top(top),
            right(right),
            bottom(bottom),
            confidence(confidence),
            class_label(class_label) {}
    void dump()
    {
        std::cout << "left : " << left << ", top : " << top << " , right : " << right << ", bottom : " << bottom << std::endl;
    }
};

using BoxArray = std::vector<Box>;


class Infer {
public:
    virtual BoxArray forward(const tensor::Image &image, int slice_width, int slice_height, float overlap_width_ratio, int overlap_height_ratio, void *stream = nullptr) = 0;
    virtual BoxArray forward(const tensor::Image &image, void *stream = nullptr) = 0;
    virtual BoxArray forwards(void *stream = nullptr) = 0;
};

std::shared_ptr<Infer> load(const std::string &engine_file, int gpu_id = 0, float confidence_threshold=0.5f, float nms_threshold=0.45f);

}



#endif