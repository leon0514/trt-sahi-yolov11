#include "common/timer.hpp"

void v11NoSlicedInfer();
void v11SlicedInfer();
void v5NoSlicedInfer();
void v5SlicedInfer();

void SpeedTest();

int main()
{
    v11SlicedInfer();
    // v5SlicedInfer();
    // SpeedTest();
    return 0;
}