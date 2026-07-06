
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <vector>

#include "tensor.cuh"

int main()
{

    Tensor testin({ 1, 2, 3, 4, 5, 6 });
    testin.print();
    
    return 0;
}
