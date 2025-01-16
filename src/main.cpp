#include "model/yolov11.hpp"
#include "common/timer.hpp"
#include "common/image.hpp"
#include "common/position.hpp"


void v5NoSlicedInfer();
void v5SlicedInfer();
void v11NoSlicedInfer();
void v11SlicedInfer();


int main()
{
    v5NoSlicedInfer();
    v5SlicedInfer();
    return 0;
}