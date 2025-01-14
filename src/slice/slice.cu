#include "slice/slice.hpp"
#include "common/check.hpp"
#include <cmath>

static __global__ void slice_kernel(
  const uint8_t*  image,
  uint8_t*  outs,
  const int width,
  const int height,
  const int slice_width,
  const int slice_height,
  const int slice_num_h,
  const int slice_num_v,
  const int overlap_width_pixel,
  const int overlap_height_pixel)
{
    const int out_size = 3 * slice_width * slice_height;
    int dx = blockDim.x * blockIdx.x + threadIdx.x;
    int dy = blockDim.y * blockIdx.y + threadIdx.y;
    if (dx >= width || dy >= height || dx < 0 || dy < 0)
    {
        return;
    }
    int offset = dy * width + dx;
    uint8_t b = image[3 * offset + 0];
    uint8_t g = image[3 * offset + 1];
    uint8_t r = image[3 * offset + 2];


    // 定义外部的动态大小享内存，存储切片范围
    extern __shared__ int slice_range[];

    int* slice_range_h = slice_range;
    int* slice_range_v = slice_range + slice_num_h * 2;

    // 计算切片的起始和结束位置，并存储在共享内存中
    if (threadIdx.x < slice_num_h) 
    {
        // 这里计算start的时候必须分两行，先计算start，再取0和start的最大值
        int start = threadIdx.x * (slice_width - overlap_width_pixel);
        start = max(start, 0);
        int end = start + slice_width;
        slice_range_h[threadIdx.x * 2] = start;
        slice_range_h[threadIdx.x * 2 + 1] = end;
        
    }

    if (threadIdx.y < slice_num_v) {
        int start = threadIdx.y * (slice_height - overlap_height_pixel);
        start = max(start, 0);
        int end = start + slice_height;
        slice_range_v[threadIdx.y * 2] = start;
        slice_range_v[threadIdx.y * 2 + 1] = end;
        
    }
    __syncthreads();

    for (int i = 0; i < slice_num_h; i++)
    {
        int sdx_start = slice_range_h[i * 2];
        int sdx_end   = slice_range_h[i * 2 + 1];

        for (int j = 0; j < slice_num_v; j++)
        {
            int sdy_start = slice_range_v[j * 2];
            int sdy_end = slice_range_v[j * 2 + 1];   
            if (dx >= sdx_start && dx < sdx_end && dy >= sdy_start && dy < sdy_end)
            {
                int image_id = i * slice_num_v + j;
                int sdx = dx - sdx_start;
                int sdy = dy - sdy_start;
                int soffset = sdy * slice_width + sdx;
                outs[image_id * out_size + 3 * soffset + 0] = b;
                outs[image_id * out_size + 3 * soffset + 1] = g;
                outs[image_id * out_size + 3 * soffset + 2] = r;
            }
        }
    }
}

static void slice_plane(const uint8_t* image,
    uint8_t*  outs,
    const int width,
    const int height,
    const int slice_width,
    const int slice_height,
    const int slice_num_h,
    const int slice_num_v,
    const int overlap_width_pixel,
    const int overlap_height_pixel,
    void* stream=nullptr)
{
    cudaStream_t stream_ = (cudaStream_t)stream;
    dim3 block(32, 32);
    dim3 grid((width + 31) / 32, (height + 31) / 32);

    int shared_memory_size = sizeof(int) * (slice_num_h + slice_num_v) * 2;

    slice_kernel<<<grid, block, shared_memory_size, stream_>>>(image, outs, 
                                    width, height, 
                                    slice_width, slice_height, 
                                    slice_num_h, slice_num_v, 
                                    overlap_width_pixel, overlap_height_pixel);
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
    printf("------------------------------------------------------\n"
            "CUDA SAHI CROP IMAGE ✂️\n"
            "Slice width                : %d\n"
            "Slice Height               : %d\n"
            "Overlap width  ratio       : %f\n"
            "Overlap height ratio       : %f\n"
            "Number of horizontal cuts  : %d\n"
            "Number of vertical cuts    : %d\n"
            "------------------------------------------------------\n", 
            slice_width_, slice_height_, overlap_width_ratio, overlap_height_ratio, slice_num_h_, slice_num_v_);
    // printf("%d,%d\n", slice_num_h_, slice_num_v_);
    int slice_num            = slice_num_h_ * slice_num_v_;
    int overlap_width_pixel  = slice_width  * overlap_width_ratio;
    int overlap_height_pixel = slice_height * overlap_height_ratio;

    size_t size_image = 3 * width * height;
    size_t output_img_size = 3 * slice_width * slice_height;

    input_image_.gpu(size_image);
    output_images_.gpu(slice_num * output_img_size);
    checkRuntime(cudaMemsetAsync(output_images_.gpu(), 114, output_images_.gpu_bytes(), stream_));
    slice_position_.resize(slice_num * 2);

    checkRuntime(cudaMemcpyAsync(input_image_.gpu(), image.bgrptr, size_image, cudaMemcpyHostToDevice, stream_));
    // checkRuntime(cudaStreamSynchronize(stream_));
    uint8_t* input_device = input_image_.gpu();
    uint8_t* output_device = output_images_.gpu();

    slice_plane(
        input_device, output_device, 
        width, height, 
        slice_width, slice_height, 
        slice_num_h_, slice_num_v_, 
        overlap_width_pixel, overlap_height_pixel, 
        stream);

    // checkRuntime(cudaStreamSynchronize(stream_));

    for (int i = 0; i < slice_num_h_; i++)
    {
        int x = std::max(0, i * (slice_width - overlap_width_pixel));
        for (int j = 0; j < slice_num_v_; j++)
        {
            int y = std::max(0, j * (slice_height - overlap_height_pixel));
            int index = i * slice_num_v_ + j;
            slice_position_[index*2]   = x;
            slice_position_[index*2+1] = y;

            // cv::Mat image = cv::Mat::zeros(slice_height, slice_width, CV_8UC3);
            // uint8_t* output_img_data = image.ptr<uint8_t>();
            // cudaMemcpyAsync(output_img_data, output_device+index*output_img_size, output_img_size*sizeof(uint8_t), cudaMemcpyDeviceToHost, stream_);
            // checkRuntime(cudaStreamSynchronize(stream_));
            // cv::imwrite(std::to_string(index) + ".png", image);
        }
    }
}

}