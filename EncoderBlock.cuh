#ifndef ENCODER_BLOCK_H
#define ENCODER_BLOCK_H

#include "cuda_runtime.h"
#include "device_launch_parameters.h"


#include "layer_norm.cuh"
#include "MultiHeadAttention.cuh"

__global__
void add_residual(float* x, float* residual, float* output, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n)
        output[i] = x[i] + residual[i];
}

__global__
void residual_backward(float* d_out, float* d_x, float* d_residual, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n)
    {
        atomicAdd(&d_x[i], d_out[i]);
        atomicAdd(&d_residual[i], d_out[i]);
    }
}

class EncoderBlock
{
public:
    int n_images, seq_len, dim_model, n_heads;

    LayerNorm  ln1, ln2;
    MultiHeadAttention mha;

    Layer ff1, ff2;

    Tensor residual1, residual2;
    Tensor after_mha, after_ff;

    Tensor output;
    Tensor* previous;

    EncoderBlock(int n_img, int seq, int dim, int heads, float lr)
        : n_images(n_img), seq_len(seq), dim_model(dim), n_heads(heads),
        ln1(dim, seq, n_img),
        ln2(dim, seq, n_img),
        mha(n_img, seq, dim, heads),
        ff1(dim, 4 * dim, n_img* seq, ActivationType::ReLu, nullptr),
        ff2(4 * dim, dim, n_img* seq, ActivationType::None, nullptr)
    {
        int total = n_img * seq * dim;
        residual1.set_size(total);
        residual2.set_size(total);
        after_mha.set_size(total);
        after_ff.set_size(total);
        output.set_size(total);
    }

    void forward()
    {
        int total = n_images * seq_len * dim_model;
        int threads = 256;
        int blocks = (total + threads - 1) / threads;


        cudaMemcpy(residual1.data, previous->data, total * sizeof(float), cudaMemcpyDeviceToDevice);

        // 1. LayerNorm1 + MHA
        ln1.previous = previous;
        ln1.forward();

        mha.previous = &ln1.output;
        mha.forward();

        // Residual 1 sumatory
        add_residual << <blocks, threads >> > (mha.output.data, residual1.data, after_mha.data, total);
        cudaDeviceSynchronize();

        // Residual 2
        cudaMemcpy(residual2.data, after_mha.data, total * sizeof(float), cudaMemcpyDeviceToDevice);

        // 2. LayerNorm2 + FeedForward
        ln2.previous = &after_mha;
        ln2.forward();

        // FForward
        ff1.previous_layer = &ln2.output;
        ff1.forward();

        ff2.previous_layer = &ff1.output;
        ff2.forward();

        // Residual 2 sumatory
        add_residual << <blocks, threads >> > (ff2.output.data, residual2.data, output.data, total);
        cudaDeviceSynchronize();
    }

    void backward(float learning_rate)
    {
        int total = n_images * seq_len * dim_model;
        int threads = 256;
        int blocks = (total + threads - 1) / threads;

        // -- Branch 2 --
        // Residual 2 backward
        residual_backward << <blocks, threads >> > ( output.gradient, ff2.output.gradient, residual2.gradient, total );
        cudaDeviceSynchronize();

        // FForward backward
        // ff2 backward
        ff2.apply_derivative();
        ff2.update_weights(learning_rate);
        ff2.compute_error_intermediate();

        // ff1 backward
        ff1.apply_derivative();
        ff1.update_weights(learning_rate);
        ff1.compute_error_intermediate();

        // LayerNom backward
        ln2.backward();
        ln2.update_weights(learning_rate);

        // Residual 1 backward
        residual_backward << <blocks, threads >> > ( residual2.gradient, after_mha.gradient, after_mha.gradient, total );
        cudaDeviceSynchronize();

        // -- Branch 1 --
        // Residual 1 backward:
        residual_backward << <blocks, threads >> > ( after_mha.gradient, mha.output.gradient, residual1.gradient, total );
        cudaDeviceSynchronize();

        mha.backward();
        mha.update_weights(learning_rate);

        ln1.backward();
        ln1.update_weights(learning_rate);

        residual_backward << <blocks, threads >> > (residual1.gradient, previous->gradient, previous->gradient, total );
        cudaDeviceSynchronize();
    }

    void zero_grad()
    {
        ln1.zero_grad();
        ln2.zero_grad();
        mha.zero_grad();
        ff1.zero_grad();
        ff2.zero_grad();
        residual1.zero_grad();
        residual2.zero_grad();
        after_mha.zero_grad();
        output.zero_grad();
    }
};

#endif
