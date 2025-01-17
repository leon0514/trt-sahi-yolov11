#include "model/yolov11.hpp"
#include "common/timer.hpp"
#include "common/image.hpp"
#include "common/position.hpp"


void v5NoSlicedInfer();
void v5SlicedInfer();
void v11NoSlicedInfer();
void v11SlicedInfer();

void v5_test_video();

int main()
{
    v5_test_video();
    // v5NoSlicedInfer();
    // v5SlicedInfer();
    return 0;
}