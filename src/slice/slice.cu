#include "slice/slice.hpp"
#include "common/check.hpp"
#include <cmath>

static __global__ void slice_kernel(
  const uchar3* __restrict__ image,
  uchar3* __restrict__ outs,
  const int width,
  const int height,
  const int slice_width,
  const int slice_height,
  const int slice_num_h,
  const int slice_num_v,
  const int* __restrict__ slice_start_point)
{
    const int slice_idx = blockIdx.z;

    const int start_x = slice_start_point[slice_idx * 2];
    const int start_y = slice_start_point[slice_idx * 2 + 1];

    // 当前像素在切片内的相对位置
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;

    if(x >= slice_width || y >= slice_height) 
        return;

    const int dx = start_x + x;
    const int dy = start_y + y;

    if(dx >= width || dy >= height) 
        return;

    // 读取像素
    const int src_index = dy * width + dx;
    const uchar3 pixel = image[src_index];

    // 写入切片
    const int dst_index = slice_idx * slice_width * slice_height + y * slice_width + x;
    outs[dst_index] = pixel;
}

static void slice_plane(const uint8_t* image,
    uint8_t* outs,
    int* slice_start_point,
    const int width,
    const int height,
    const int slice_width,
    const int slice_height,
    const int slice_num_h,
    const int slice_num_v,
    void* stream=nullptr)
{
    int slice_total = slice_num_h * slice_num_v;
    cudaStream_t stream_ = (cudaStream_t)stream;
    dim3 block(32, 32);
    dim3 grid(
        (slice_width + block.x - 1) / block.x,
        (slice_height + block.y - 1) / block.y,
        slice_total
    );
    slice_kernel<<<grid, block, 0, stream_>>>(
        reinterpret_cast<const uchar3*>(image),
        reinterpret_cast<uchar3*>(outs),
        width, height, 
        slice_width, slice_height, 
        slice_num_h, slice_num_v, 
        slice_start_point
    );
}


namespace slice
{

int calculateNumCuts(int dimension, int subDimension, float overlapRatio) {
    float step = subDimension * (1 - overlapRatio);
    if(step == 0)
    {
        return 1;
    }
    float cuts = static_cast<float>(dimension - subDimension) / step;
    // 浮点数会有很小的误差，直接向上取整会出现多裁剪了一张图的情况
    if (fabs(cuts - round(cuts)) < 0.0001) {
        cuts = round(cuts);
    }
    int numCuts = static_cast<int>(std::ceil(cuts));
    return numCuts + 1;
}

static int calc_resolution_factor(int resolution)
{
    int expo = 0;
    while(pow(2, expo) < resolution) expo++;
    return expo - 1;
} 

static std::string calc_aspect_ratio_orientation(int width, int height)
{
    if (width < height)
        return  "vertical";
    else if(width > height)
        return "horizontal";
    else
        return "square";
}

static std::tuple<int, int, float, float> calc_ratio_and_slice(const std::string& orientation, int slide=1, float ratio=0.1)
{
    int slice_row, slice_col;
    float overlap_height_ratio, overlap_width_ratio;
    if (orientation == "vertical")
    {
        slice_row = slide;
        slice_col = slide * 2;
        overlap_height_ratio = ratio;
        overlap_width_ratio = ratio;
    }
    else if (orientation == "horizontal")
    {
        slice_row = slide * 2;
        slice_col = slide;
        overlap_height_ratio = ratio;
        overlap_width_ratio = ratio;
    }
    else if (orientation == "square")
    {
        slice_row = slide;
        slice_col = slide;
        overlap_height_ratio = ratio;
        overlap_width_ratio = ratio;
    }
    return std::make_tuple(slice_row, slice_col, overlap_height_ratio, overlap_width_ratio);
}

static std::tuple<int, int, float, float> calc_slice_and_overlap_params(
    const std::string& resolution, int width, int height, std::string orientation)
{
    int split_row, split_col;
    float overlap_height_ratio, overlap_width_ratio;
    if (resolution == "medium")
        std::tie(split_row, split_col, overlap_height_ratio, overlap_width_ratio) = calc_ratio_and_slice(
            orientation, 1, 0.8
        );

    else if (resolution == "high")
        std::tie(split_row, split_col, overlap_height_ratio, overlap_width_ratio) = calc_ratio_and_slice(
            orientation, 2, 0.4
        );

    else if (resolution == "ultra-high")
        std::tie(split_row, split_col, overlap_height_ratio, overlap_width_ratio) = calc_ratio_and_slice(
            orientation, 4, 0.4
        );
    else
    {
        split_col = 1;
        split_row = 1;
        overlap_width_ratio = 1;
        overlap_height_ratio = 1;
    }
    int slice_height = height / split_col;
    int slice_width = width / split_row;
    return std::make_tuple(slice_width, slice_height, overlap_height_ratio, overlap_width_ratio);
}

static std::tuple<int, int, float, float> get_resolution_selector(const std::string& resolution, int width, int height)
{
    std::string orientation = calc_aspect_ratio_orientation(width, height);
    return calc_slice_and_overlap_params(resolution, width, height, orientation);

}

static std::tuple<int, int, float, float> get_auto_slice_params(int width, int height)
{
    int resolution = height * width;
    int factor = calc_resolution_factor(resolution);
    if (factor <= 18)
        return get_resolution_selector("low", width, height);
    else if (18 <= factor && factor < 21)
        return get_resolution_selector("medium", width, height);
    else if (21 <= factor && factor < 24)
        return get_resolution_selector("high", width, height);
    else
        return get_resolution_selector("ultra-high", width, height);
}

void SliceImage::autoSlice(
        const tensor::Image& image,
        void* stream)
{
    int slice_width;
    int slice_height;
    float overlap_width_ratio;
    float overlap_height_ratio;
    std::tie(slice_width, slice_height, overlap_width_ratio, overlap_height_ratio) = get_auto_slice_params(image.width, image.height);
    slice(image, slice_width, slice_height, overlap_width_ratio, overlap_height_ratio, stream);
}

void SliceImage::slice(
        const tensor::Image& image, 
        const int slice_width,
        const int slice_height,
        const float overlap_width_ratio,
        const float overlap_height_ratio,
        void* stream)
{
    slice_width_  = slice_width;
    slice_height_ = slice_height;
    cudaStream_t stream_ = (cudaStream_t)stream;

    int width = image.width;
    int height = image.height;

    slice_num_h_ = calculateNumCuts(width, slice_width, overlap_width_ratio);
    slice_num_v_ = calculateNumCuts(height, slice_height, overlap_height_ratio);
    // printf("------------------------------------------------------\n"
    //     "CUDA SAHI CROP IMAGE ✂️\n"
    //     "------------------------------------------------------\n"
    //     "%-30s: %-10d\n"
    //     "%-30s: %-10d\n"
    //     "%-30s: %-10.2f\n"
    //     "%-30s: %-10.2f\n"
    //     "%-30s: %-10d\n"
    //     "%-30s: %-10d\n"
    //     "------------------------------------------------------\n", 
    //     "Slice width", slice_width_,
    //     "Slice height", slice_height_,
    //     "Overlap width ratio", overlap_width_ratio,
    //     "Overlap height ratio", overlap_height_ratio,
    //     "Number of horizontal cuts", slice_num_h_,
    //     "Number of vertical cuts", slice_num_v_);
    int slice_num            = slice_num_h_ * slice_num_v_;
    int overlap_width_pixel  = slice_width  * overlap_width_ratio;
    int overlap_height_pixel = slice_height * overlap_height_ratio;

    size_t size_image = 3 * width * height;
    size_t output_img_size = 3 * slice_width * slice_height;

    input_image_.gpu(size_image);
    output_images_.gpu(slice_num * output_img_size);
    checkRuntime(cudaMemsetAsync(output_images_.gpu(), 114, output_images_.gpu_bytes(), stream_));

    checkRuntime(cudaMemcpyAsync(input_image_.gpu(), image.bgrptr, size_image, cudaMemcpyHostToDevice, stream_));
    // checkRuntime(cudaStreamSynchronize(stream_));

    uint8_t* input_device = input_image_.gpu();
    uint8_t* output_device = output_images_.gpu();

    slice_start_point_.cpu(slice_num * 2);
    slice_start_point_.gpu(slice_num * 2);

    int* slice_start_point_ptr = slice_start_point_.cpu();
    
    for (int i = 0; i < slice_num_h_; i++)
    {
        int x = std::min(width - slice_width, std::max(0, i * (slice_width - overlap_width_pixel)));
        for (int j = 0; j < slice_num_v_; j++)
        {
            int y = std::min(height - slice_height, std::max(0, j * (slice_height - overlap_height_pixel)));
            int index = (i * slice_num_v_ + j) * 2;
            slice_start_point_ptr[index] = x;
            slice_start_point_ptr[index + 1] = y;
        }
    }
    
    checkRuntime(cudaMemcpyAsync(slice_start_point_.gpu(), slice_start_point_.cpu(), slice_num*2*sizeof(int), cudaMemcpyHostToDevice, stream_));
    checkRuntime(cudaStreamSynchronize(stream_));
    slice_plane(
        input_device, output_device, slice_start_point_.gpu(),
        width, height, 
        slice_width, slice_height, 
        slice_num_h_, slice_num_v_,
        stream);

    // checkRuntime(cudaStreamSynchronize(stream_));

    // for (int i = 0; i < slice_num_h_; i++)
    // {
    //     for (int j = 0; j < slice_num_v_; j++)
    //     {
    //         int index = i * slice_num_v_ + j;
    //         slice_position_[index*2]   = slice_start_point_ptr[index*2];
    //         slice_position_[index*2+1] = slice_start_point_ptr[index*2+1];

    //         // cv::Mat image = cv::Mat::zeros(slice_height, slice_width, CV_8UC3);
    //         // uint8_t* output_img_data = image.ptr<uint8_t>();
    //         // cudaMemcpyAsync(output_img_data, output_device+index*output_img_size, output_img_size*sizeof(uint8_t), cudaMemcpyDeviceToHost, stream_);
    //         // checkRuntime(cudaStreamSynchronize(stream_));
    //         // cv::imwrite(std::to_string(index) + ".png", image);
    //     }
    // }
}

}