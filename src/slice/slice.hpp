#ifndef SLICE_HPP__
#define SLICE_HPP__

#include "opencv2/opencv.hpp"
#include "common/image.hpp"
#include "common/memory.hpp"
#include <vector>

namespace slice
{

int calculateNumCuts(int dimension, int subDimension, float overlapRatio);

class SliceImage{
public:
    tensor::Memory<unsigned char> input_image_;
    tensor::Memory<unsigned char> output_images_;

    tensor::Memory<int> slice_start_point_;

    int slice_num_h_;
    int slice_num_v_;

    int slice_width_;
    int slice_height_;

    // std::vector<int> slice_position_;

public:
    void slice(
        const tensor::Image& image, 
        const int slice_width,
        const int slice_height, 
        const float overlap_width_ratio,
        const float overlap_height_ratio,
        void* stream=nullptr);
    
    void autoSlice(
        const tensor::Image& image, 
        void* stream=nullptr);
};



}


#endif