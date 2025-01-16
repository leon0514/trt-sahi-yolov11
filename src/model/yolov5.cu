#include "model/yolov5.hpp"
#include <vector>
#include <memory>
#include "slice/slice.hpp"
#include "model/affine.hpp"
#include "common/check.hpp"

#ifdef TRT10
#include "common/tensorrt.hpp"
namespace TensorRT = TensorRT10;
#else
#include "common/tensorrt8.hpp"
namespace TensorRT = TensorRT8;
#endif

#define GPU_BLOCK_THREADS 512

namespace yolov5
{

static const int NUM_BOX_ELEMENT = 8;  // left, top, right, bottom, confidence, class, keepflag, row_index(output)
static const int MAX_IMAGE_BOXES = 1024;

static dim3 grid_dims(int numJobs){
  int numBlockThreads = numJobs < GPU_BLOCK_THREADS ? numJobs : GPU_BLOCK_THREADS;
  return dim3(((numJobs + numBlockThreads - 1) / (float)numBlockThreads));
}

static dim3 block_dims(int numJobs){
  return numJobs < GPU_BLOCK_THREADS ? numJobs : GPU_BLOCK_THREADS;
}

static __host__ __device__ void affine_project(float *matrix, float x, float y, float *ox, float *oy) 
{
    *ox = matrix[0] * x + matrix[1] * y + matrix[2];
    *oy = matrix[3] * x + matrix[4] * y + matrix[5];
}

static __global__ void decode_kernel_v5(float *predict, int num_bboxes, int num_classes,
                                              int output_cdim, float confidence_threshold,
                                              float *invert_affine_matrix, float *parray, int *box_count,
                                              int max_image_boxes, int start_x, int start_y) 
{
    int position = blockDim.x * blockIdx.x + threadIdx.x;
    if (position >= num_bboxes) return;

    float *pitem = predict + output_cdim * position;
    float objectness = pitem[4];
    if (objectness < confidence_threshold) return;

    float *class_confidence = pitem + 5;
    
    float confidence = *class_confidence++;
    int label = 0;
    for (int i = 1; i < num_classes; ++i, ++class_confidence) 
    {
        if (*class_confidence > confidence) 
        {
            confidence = *class_confidence;
            label = i;
        }
    }
    confidence *= objectness;
    if (confidence < confidence_threshold) return;

    int index = atomicAdd(box_count, 1);
    if (index >= max_image_boxes) return;

    float cx = *pitem++;
    float cy = *pitem++;
    float width = *pitem++;
    float height = *pitem++;
    float left = cx - width * 0.5f;
    float top = cy - height * 0.5f;
    float right = cx + width * 0.5f;
    float bottom = cy + height * 0.5f;
    affine_project(invert_affine_matrix, left, top, &left, &top);
    affine_project(invert_affine_matrix, right, bottom, &right, &bottom);

    float *pout_item = parray + index * NUM_BOX_ELEMENT;
    *pout_item++ = left + start_x;
    *pout_item++ = top + start_y;
    *pout_item++ = right + start_x;
    *pout_item++ = bottom + start_y;
    *pout_item++ = confidence;
    *pout_item++ = label;
    *pout_item++ = 1;  // 1 = keep, 0 = ignore
    *pout_item++ = position;
}


static __device__ float box_iou(float aleft, float atop, float aright, float abottom, float bleft,
                                float btop, float bright, float bbottom)
{
    float cleft = max(aleft, bleft);
    float ctop = max(atop, btop);
    float cright = min(aright, bright);
    float cbottom = min(abottom, bbottom);

    float c_area = max(cright - cleft, 0.0f) * max(cbottom - ctop, 0.0f);
    if (c_area == 0.0f) return 0.0f;

    float a_area = max(0.0f, aright - aleft) * max(0.0f, abottom - atop);
    float b_area = max(0.0f, bright - bleft) * max(0.0f, bbottom - btop);
    return c_area / (a_area + b_area - c_area);
}


static __global__ void fast_nms_kernel(float *bboxes, int* box_count, int max_image_boxes, float threshold) 
{
    int position = (blockDim.x * blockIdx.x + threadIdx.x);
    // int count = min((int)*box_count, MAX_IMAGE_BOXES);
    int count = max_image_boxes;
    if (position >= count) return;

    // left, top, right, bottom, confidence, class, keepflag
    float *pcurrent = bboxes + position * NUM_BOX_ELEMENT;
    for (int i = 0; i < count; ++i) 
    {
        float *pitem = bboxes + i * NUM_BOX_ELEMENT;
        if (i == position || pcurrent[5] != pitem[5]) continue;

        if (pitem[4] >= pcurrent[4]) 
        {
            if (pitem[4] == pcurrent[4] && i < position) continue;

            float iou = box_iou(pcurrent[0], pcurrent[1], pcurrent[2], pcurrent[3], pitem[0], pitem[1],
                                pitem[2], pitem[3]);

            if (iou > threshold) 
            {
                pcurrent[6] = 0;  // 1=keep, 0=ignore
                return;
            }
        }
    }
}

static void decode_kernel_invoker(float *predict, int num_bboxes, int num_classes, int output_cdim,
                                  float confidence_threshold, float nms_threshold,
                                  float *invert_affine_matrix, float *parray, int* box_count, int max_image_boxes,
                                  int start_x, int start_y, cudaStream_t stream) 
{
    auto grid = grid_dims(num_bboxes);
    auto block = block_dims(num_bboxes);

    checkKernel(decode_kernel_v5<<<grid, block, 0, stream>>>(
            predict, num_bboxes, num_classes, output_cdim, confidence_threshold, invert_affine_matrix,
            parray, box_count, max_image_boxes, start_x, start_y));

    // grid = grid_dims(MAX_IMAGE_BOXES);
    // block = block_dims(MAX_IMAGE_BOXES);
    // checkKernel(fast_nms_kernel<<<grid, block, 0, stream>>>(parray, box_count, MAX_IMAGE_BOXES, nms_threshold));
}

static void fast_nms_kernel_invoker(float *parray, int* box_count, int max_image_boxes, float nms_threshold, cudaStream_t stream)
{
    auto grid = grid_dims(max_image_boxes);
    auto block = block_dims(max_image_boxes);
    checkKernel(fast_nms_kernel<<<grid, block, 0, stream>>>(parray, box_count, max_image_boxes, nms_threshold));
}

class Yolov5ModelImpl : public Infer 
{
public:
    // for sahi crop image
    std::shared_ptr<slice::SliceImage> slice_;
    std::shared_ptr<TensorRT::Engine> trt_;
    std::string engine_file_;

    tensor::Memory<int> box_count_;

    tensor::Memory<float> affine_matrix_;
    tensor::Memory<float>  input_buffer_, bbox_predict_, output_boxarray_;

    int network_input_width_, network_input_height_;
    affine::Norm normalize_;
    std::vector<int> bbox_head_dims_;
    bool isdynamic_model_ = false;

    float confidence_threshold_;
    float nms_threshold_;

    int num_classes_ = 0;

    virtual ~Yolov5ModelImpl() = default;

    void adjust_memory(int batch_size) 
    {
        // the inference batch_size
        size_t input_numel = network_input_width_ * network_input_height_ * 3;
        input_buffer_.gpu(batch_size * input_numel);
        bbox_predict_.gpu(batch_size * bbox_head_dims_[1] * bbox_head_dims_[2]);
        output_boxarray_.gpu(batch_size * (MAX_IMAGE_BOXES * NUM_BOX_ELEMENT));
        output_boxarray_.cpu(batch_size * (MAX_IMAGE_BOXES * NUM_BOX_ELEMENT));

        affine_matrix_.gpu(6);
        affine_matrix_.cpu(6);

        box_count_.gpu(1);
        box_count_.cpu(1);
    }

    void preprocess(int ibatch, affine::LetterBoxMatrix &affine, void *stream = nullptr)
    {
        affine.compute(std::make_tuple(slice_->slice_width_, slice_->slice_height_),
                    std::make_tuple(network_input_width_, network_input_height_));

        size_t input_numel = network_input_width_ * network_input_height_ * 3;
        float *input_device = input_buffer_.gpu() + ibatch * input_numel;
        size_t size_image = slice_->slice_width_ * slice_->slice_height_ * 3;

        float *affine_matrix_device = affine_matrix_.gpu();
        uint8_t *image_device = slice_->output_images_.gpu() + ibatch * size_image;

        float *affine_matrix_host = affine_matrix_.cpu();

        // speed up
        cudaStream_t stream_ = (cudaStream_t)stream;
        memcpy(affine_matrix_host, affine.d2i, sizeof(affine.d2i));
        checkRuntime(cudaMemcpyAsync(affine_matrix_device, affine_matrix_host, sizeof(affine.d2i),
                                    cudaMemcpyHostToDevice, stream_));

        affine::warp_affine_bilinear_and_normalize_plane(image_device, slice_->slice_width_ * 3, slice_->slice_width_,
                                                slice_->slice_height_, input_device, network_input_width_,
                                                network_input_height_, affine_matrix_device, 114,
                                                normalize_, stream_);
    }

    bool load(const std::string &engine_file, float confidence_threshold, float nms_threshold) 
    {
        trt_ = TensorRT::load(engine_file);
        if (trt_ == nullptr) return false;

        trt_->print();

        this->confidence_threshold_ = confidence_threshold;
        this->nms_threshold_ = nms_threshold;

        auto input_dim = trt_->static_dims(0);
        bbox_head_dims_ = trt_->static_dims(1);
        network_input_width_ = input_dim[3];
        network_input_height_ = input_dim[2];
        isdynamic_model_ = trt_->has_dynamic_dim();

        normalize_ = affine::Norm::alpha_beta(1 / 255.0f, 0.0f, affine::ChannelType::SwapRB);
        num_classes_ = bbox_head_dims_[2] - 5;
        return true;
    }


    virtual BoxArray forward(const tensor::Image &image, int slice_width, int slice_height, float overlap_width_ratio, float overlap_height_ratio, void *stream = nullptr) override 
    {
        slice_->slice(image, slice_width, slice_height, overlap_width_ratio, overlap_height_ratio, stream);
        return forwards(stream);
    }

    virtual BoxArray forward(const tensor::Image &image, void *stream = nullptr) override 
    {
        slice_->autoSlice(image, stream);
        return forwards(stream);
    }

    virtual BoxArray forwards(void *stream = nullptr) override 
    {
        int num_image = slice_->slice_num_h_ * slice_->slice_num_v_;
        if (num_image == 0) return {};
        
        auto input_dims = trt_->static_dims(0);
        int infer_batch_size = input_dims[0];
        if (infer_batch_size != num_image) 
        {
            if (isdynamic_model_) 
            {
                infer_batch_size = num_image;
                input_dims[0] = num_image;
                if (!trt_->set_run_dims(0, input_dims)) 
                {
                    printf("Fail to set run dims\n");
                    return {};
                }
            } 
            else 
            {
                if (infer_batch_size < num_image) 
                {
                    printf(
                        "When using static shape model, number of images[%d] must be "
                        "less than or equal to the maximum batch[%d].",
                        num_image, infer_batch_size);
                    return {};
                }
            }
        }
        adjust_memory(infer_batch_size);

        affine::LetterBoxMatrix affine_matrix;
        cudaStream_t stream_ = (cudaStream_t)stream;
        for (int i = 0; i < num_image; ++i)
            preprocess(i, affine_matrix, stream);

        float *bbox_output_device = bbox_predict_.gpu();
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

        int* box_count = box_count_.gpu();
        checkRuntime(cudaMemsetAsync(box_count, 0, sizeof(int), stream_));
        for (int ib = 0; ib < num_image; ++ib) 
        {
            int start_x = slice_->slice_position_[ib*2];
            int start_y = slice_->slice_position_[ib*2+1];
            float *boxarray_device =
                output_boxarray_.gpu() + ib * (MAX_IMAGE_BOXES * NUM_BOX_ELEMENT);
            float *affine_matrix_device = affine_matrix_.gpu();
            float *image_based_bbox_output =
                bbox_output_device + ib * (bbox_head_dims_[1] * bbox_head_dims_[2]);
            decode_kernel_invoker(image_based_bbox_output, bbox_head_dims_[1], num_classes_,
                                    bbox_head_dims_[2], confidence_threshold_, nms_threshold_,
                                    affine_matrix_device, boxarray_device, box_count, MAX_IMAGE_BOXES, start_x, start_y, stream_);
        }
        // checkRuntime(cudaStreamSynchronize(stream_));
        float *boxarray_device =  output_boxarray_.gpu();
        fast_nms_kernel_invoker(boxarray_device, box_count, MAX_IMAGE_BOXES * num_image, nms_threshold_, stream_);
        checkRuntime(cudaMemcpyAsync(output_boxarray_.cpu(), output_boxarray_.gpu(),
                                    output_boxarray_.gpu_bytes(), cudaMemcpyDeviceToHost, stream_));
        checkRuntime(cudaMemcpyAsync(box_count_.cpu(), box_count_.gpu(),
                                    box_count_.gpu_bytes(), cudaMemcpyDeviceToHost, stream_));
        checkRuntime(cudaStreamSynchronize(stream_));

        BoxArray result;
        // int imemory = 0;
        for (int ib = 0; ib < num_image; ++ib) 
        {
            
            float *parray = output_boxarray_.cpu() + ib * (MAX_IMAGE_BOXES * NUM_BOX_ELEMENT);
            int count = min(MAX_IMAGE_BOXES, *(box_count_.cpu()));
            for (int i = 0; i < count; ++i) 
            {
                float *pbox = parray + i * NUM_BOX_ELEMENT;
                int label = pbox[5];
                int keepflag = pbox[6];
                if (keepflag == 1) {
                    Box result_object_box(pbox[0], pbox[1], pbox[2], pbox[3], pbox[4], label);
                    result.emplace_back(result_object_box);
                }
            }
        }
        return result;
    }

};


Infer *loadraw(const std::string &engine_file, float confidence_threshold,
               float nms_threshold) 
{
    Yolov5ModelImpl *impl = new Yolov5ModelImpl();
    if (!impl->load(engine_file, confidence_threshold, nms_threshold)) 
    {
        delete impl;
        impl = nullptr;
    }
    impl->slice_ = std::make_shared<slice::SliceImage>();
    return impl;
}

std::shared_ptr<Infer> load(const std::string &engine_file, int gpu_id, float confidence_threshold,
               float nms_threshold) 
{
    checkRuntime(cudaSetDevice(gpu_id));
    return std::shared_ptr<Yolov5ModelImpl>((Yolov5ModelImpl *)loadraw(engine_file, confidence_threshold, nms_threshold));
}

}
