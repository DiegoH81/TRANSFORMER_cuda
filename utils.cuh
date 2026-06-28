#ifndef UTILS_H
#define UTILS_H

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

__global__
void sgd_update(float* data, float* gradient, float learning_rate, int size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= size)
        return;

    data[idx] += learning_rate * gradient[idx];
    gradient[idx] = 0.0f;
}

#endif