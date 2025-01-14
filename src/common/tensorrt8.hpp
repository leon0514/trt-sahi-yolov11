#ifndef TENSORRT8_HPP__
#define TENSORRT8_HPP__
#include <vector>
#include "common/memory.hpp"

namespace TensorRT8
{

enum class DType : int { FLOAT = 0, HALF = 1, INT8 = 2, INT32 = 3, BOOL = 4, UINT8 = 5 };

class Engine 
{
public:
    virtual bool forward(const std::vector<void *> &bindings, void *stream = nullptr,
                        void *input_consum_event = nullptr) = 0;
    virtual int index(const std::string &name) = 0;
    virtual std::vector<int> run_dims(const std::string &name) = 0;
    virtual std::vector<int> run_dims(int ibinding) = 0;
    virtual std::vector<int> static_dims(const std::string &name) = 0;
    virtual std::vector<int> static_dims(int ibinding) = 0;
    virtual int numel(const std::string &name) = 0;
    virtual int numel(int ibinding) = 0;
    virtual int num_bindings() = 0;
    virtual bool is_input(int ibinding) = 0;
    virtual bool set_run_dims(const std::string &name, const std::vector<int> &dims) = 0;
    virtual bool set_run_dims(int ibinding, const std::vector<int> &dims) = 0;
    virtual DType dtype(const std::string &name) = 0;
    virtual DType dtype(int ibinding) = 0;
    virtual bool has_dynamic_dim() = 0;
    virtual void print() = 0;
};

std::shared_ptr<Engine> load(const std::string &file);
std::string format_shape(const std::vector<int> &shape);

}  // namespace trt

#endif