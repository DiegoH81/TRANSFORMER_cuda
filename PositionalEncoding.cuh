#ifndef POSITIONAL_ENCODING_H
#define POSITIONAL_ENCODING_H

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "tensor.cuh"
#include "layer.cuh"
#include "utils.cuh"

#include <vector>

__global__
void add_positional(float* tokens, float* pos_emb, int d_model)
{
    int img_idx = blockIdx.x;
    int tok_idx = blockIdx.y;

    int inner_idx = threadIdx.x;

    int real_tok_idx = (img_idx * 17 + tok_idx) * d_model + inner_idx;
    int pos_idx = tok_idx * d_model + inner_idx;

    tokens[real_tok_idx] += pos_emb[pos_idx];
}

__global__
void positional_backward(float* grad_output, float* grad_pos_emb, int d_model)
{
    int img_idx = blockIdx.x;
    int tok_idx = blockIdx.y;
    int inner_idx = threadIdx.x;

    int grad_idx = (img_idx * 17 + tok_idx) * d_model + inner_idx;
    int pos_idx = tok_idx * d_model + inner_idx;

    atomicAdd(&grad_pos_emb[pos_idx], grad_output[grad_idx]);
}

class PositionalEncoding
{
public:
    Tensor pos_emb;
    Tensor* previous;
    int dim_model, n_images, sequence_len;

    PositionalEncoding(int in_sequence_len, int in_d_model, int in_n_images, Tensor* in_previous = nullptr)
        : sequence_len(in_sequence_len), dim_model(in_d_model), n_images(in_n_images), previous(in_previous)
    {
        pos_emb.set_size(sequence_len * dim_model);
        pos_emb.set_random(0.02f);
    }

    void forward()
    {
        dim3 grid(n_images, sequence_len);

        add_positional << <grid, dim_model >> > ( previous->data, pos_emb.data, dim_model);

        cudaDeviceSynchronize();
    }

    void backward()
    {
        dim3 grid(n_images, sequence_len);
        positional_backward << <grid, dim_model >> > ( previous->gradient, pos_emb.gradient, dim_model );
        cudaDeviceSynchronize();
    }

    void update_weights(float in_learning_rate)
    {
        int total = sequence_len * dim_model;
        int threads = 256;
        int blocks = (total + threads - 1) / threads;

        float effective_lr = in_learning_rate / (float)n_images;

        sgd_update << <blocks, threads >> > ( pos_emb.data, pos_emb.gradient, effective_lr, total );
        cudaDeviceSynchronize();
    }

    void zero_grad()
    {
        pos_emb.zero_grad();
    }
};

#endif