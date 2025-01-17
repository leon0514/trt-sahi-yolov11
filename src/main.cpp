#include "common/timer.hpp"

void v11NoSlicedInfer();
void v11SlicedInfer();
void v5NoSlicedInfer();
void v5SlicedInfer();

void v11SpeedTest();

int main()
{
    // v11SlicedInfer();
    // v5SlicedInfer();
    v11SpeedTest();
    return 0;
}