#include "common/timer.hpp"

void v11NoSlicedInfer();
void v11SlicedInfer();
void v5NoSlicedInfer();
void v5SlicedInfer();

int main()
{
    v11SlicedInfer();
    v5SlicedInfer();
    return 0;
}