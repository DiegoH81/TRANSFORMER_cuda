#ifndef CLS_TOKEN_H
#define CLS_TOKEN_H

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "tensor.cuh"
#include "layer.cuh"
#include "utils.cuh"

#include <vector>

__global__
void prepend_cls(float* cls, float* tokens, float* out, int d_model)
{
    int img_idx = blockIdx.x;
    int tok_idx = blockIdx.y;

    int inner_idx = threadIdx.x;

    if (tok_idx == 0) // First token appends CLS
        out[(img_idx * 17) * d_model + inner_idx] = cls[inner_idx];
    else
        out[(img_idx * 17 + tok_idx) * d_model + inner_idx] = tokens[(img_idx * 16 + tok_idx - 1) * d_model + inner_idx];
}

__global__
void cls_backward(float* grad_output, float* grad_cls, float* grad_previous, int d_model)
{
    int img_idx = blockIdx.x;
    int tok_idx = blockIdx.y;
    int inner_idx = threadIdx.x;

    if (tok_idx == 0)
        atomicAdd(&grad_cls[inner_idx], grad_output[img_idx * 17 * d_model + inner_idx]);
    else
    {
        int src = (img_idx * 17 + tok_idx) * d_model + inner_idx;
        int dst = (img_idx * 16 + tok_idx - 1) * d_model + inner_idx;
        grad_previous[dst] = grad_output[src];
    }
}

class CLSToken
{
public:
    Tensor cls, output;
    Tensor* previous;

    int dim_model, n_images, num_patches;

    CLSToken(int in_d_model, int in_num_patches, int in_n_images, Tensor* in_previous = nullptr) :
        dim_model(in_d_model), n_images(in_n_images), num_patches(in_num_patches), previous(in_previous)
    {
        cls.set_size(dim_model);
        cls.set_random(0.02f);
        output.set_size(n_images * (num_patches + 1) * dim_model);
    }

    void forward()
    {
        dim3 grid(n_images, num_patches + 1);
        prepend_cls << <grid, dim_model >> > ( cls.data, previous->data, output.data, dim_model );
        cudaDeviceSynchronize();
    }

    void backward()
    {
        dim3 grid(n_images, num_patches + 1);
        cls_backward << <grid, dim_model >> > ( output.gradient, cls.gradient, previous->gradient, dim_model );
        cudaDeviceSynchronize();
    }

    void update_weights(float in_learning_rate)
    {
        int threads = dim_model;
        float effective_lr = in_learning_rate / (float)n_images;

        sgd_update << <1, threads >> > ( cls.data, cls.gradient, effective_lr, dim_model );
        cudaDeviceSynchronize();
    }

    void zero_grad()
    {
        cls.zero_grad();
        output.zero_grad();
    }
};


#endif