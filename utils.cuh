#ifndef UTILS_CUH
#define UTILS_CUH

#define CLIP_VAL 1.0f
#define WEIGHT_LIMIT 5.0f

#include "cuda_runtime.h"
#include "device_launch_parameters.h"



__device__
float clip_grad(float g)
{
 
    /*
    if (g > CLIP_VAL)
        return CLIP_VAL;
    if (g < -CLIP_VAL)
        return -CLIP_VAL;
    return g;
    */
}

__global__
void clamp_weights_krnl(float* weights, size_t size)
{
    /*
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= size)
        return;
    weights[idx] = fminf(WEIGHT_LIMIT, fmaxf(-WEIGHT_LIMIT, weights[idx]));
    */
}

#endif