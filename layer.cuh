#ifndef LAYER_H
#define LAYER_H

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "tensor.cuh"
#include "activation_function.cuh"

__global__
void layer_forward(float* weights, float* bias, float* input, float* output,
                   size_t input_size, size_t output_size, size_t batch_size)
{
    int output_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int batch_idx = blockIdx.y;

    if (output_idx >= output_size || batch_idx >= batch_size)
        return;

    float sum = bias[output_idx];

    for (size_t i = 0; i < input_size; i++)
        sum += weights[output_idx * input_size + i] * input[(batch_idx * input_size) + i];

    output[(batch_idx * output_size) + output_idx] = sum;
}

__global__
void layer_forward_shared(float* weights, float* bias, float* input, float* output,
                          size_t input_size, size_t output_size, size_t batch_size)
{
    int batch_idx = blockIdx.x;
    int neuron_idx = blockIdx.y;
    int tid = threadIdx.x;

    if (batch_idx >= batch_size || neuron_idx >= output_size) return;

    extern __shared__ float partial[]; // shared dinámica, tamaño = blockDim.x

    float val = 0.0f;
    if (tid < input_size)
        val = weights[neuron_idx * input_size + tid] * input[batch_idx * input_size + tid];
    partial[tid] = val;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride /= 2) {
        if (tid < stride) partial[tid] += partial[tid + stride];
        __syncthreads();
    }

    if (tid == 0)
        output[batch_idx * output_size + neuron_idx] = partial[0] + bias[neuron_idx];
}

__global__
void compute_ej(float* weights_data, float* input_gradients, float* output_gradient,
                size_t input_size, size_t output_size, size_t batch_size)
{
    int input_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int batch_idx = blockIdx.y;

    if (input_idx >= input_size || batch_idx >= batch_size)
        return;

    for (size_t i = 0; i < output_size; i++)
        input_gradients[(batch_idx * input_size) + input_idx] += output_gradient[(batch_idx * output_size) + i] * weights_data[(input_size * i) + input_idx];
}

__global__
void compute_ej_last_layer(float* output_data, float* output_gradients,
                           float* expected_output,
                           size_t output_size, size_t batch_size)
{
    int output_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int batch_idx = blockIdx.y;

    if (output_idx >= output_size || batch_idx >= batch_size)
        return;

    output_gradients[(batch_idx * output_size)  + output_idx] = expected_output[(batch_idx * output_size) + output_idx] - output_data[(batch_idx * output_size) + output_idx];
}

__global__
void update_weights_krnl(float* weights, float* input_data,
                         float* output_gradients, float* output_data, float learning_rate,
                         size_t input_size, size_t output_size, size_t batch_size)
{
    int weight_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int batch_idx = blockIdx.y;

    if (weight_idx >= output_size || batch_idx >= batch_size)
        return;

    
    for (size_t i = 0; i < input_size; i++)
    {
        float sum = learning_rate * input_data[(batch_idx * input_size) + i] * output_gradients[(batch_idx * output_size) + weight_idx];
        atomicAdd(&weights[weight_idx * input_size + i], sum / (float)batch_size);
    }
}

__global__
void update_bias_krnl(float* bias_data, float* output_gradients, float learning_rate,
                      size_t output_size, size_t batch_size)
{
    int bias_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int batch_idx = blockIdx.y;

    if (bias_idx >= output_size || batch_idx >= batch_size)
        return;

    float sum = learning_rate * output_gradients[(batch_idx * output_size) + bias_idx]; //* bias_data[batch_idx] 

    atomicAdd(&bias_data[bias_idx], sum / (float)batch_size);
}

class Layer
{
public:
    size_t input_size, output_size, batch_size;
    Tensor output, weights, bias, *previous_layer;
    ActivationType activation;

    Layer(size_t in_size, size_t out_size, size_t in_batch_size, ActivationType in_activation, Tensor* in_previous = nullptr) :
        input_size(in_size), output_size(out_size), batch_size(in_batch_size), activation(in_activation), previous_layer(in_previous)
    {
        weights.set_size(input_size * output_size);
        bias.set_size(output_size);

        weights.set_random(0.1f);
        bias.set_random(0.1f);

        output.set_size(output_size * batch_size);
    }

    void forward()
    {
        int threads = 256;
        int blocks_num = (output_size + threads - 1) / threads;
        dim3 blocks(blocks_num, batch_size);

        layer_forward << < blocks, threads >> > (weights.data, bias.data, previous_layer->data, output.data, input_size, output_size, batch_size);
        cudaDeviceSynchronize();

        if (activation == ActivationType::ReLu)
        {
            ReLu_forward << < blocks, threads >> > (output.data, output_size, batch_size);
            cudaDeviceSynchronize();
        }
    }

    void apply_derivative()
    {
        int threads = 256;
        int blocks_num = (output_size + threads - 1) / threads;
        dim3 blocks(blocks_num, batch_size);

        if (activation == ActivationType::ReLu)
        {
            ReLu_derivative << < blocks, threads >> > (output.gradient, output.data, output_size, batch_size);
            cudaDeviceSynchronize();
        }
    }

    void compute_error_intermediate()
    {
        int threads = 256;
        int blocks_num = (input_size + threads - 1) / threads;
        dim3 blocks(blocks_num, batch_size);

        if (previous_layer)
        {
            compute_ej << < blocks, threads >> > (weights.data, previous_layer->gradient, output.gradient, input_size, output_size, batch_size);
            cudaDeviceSynchronize();
        }
    }

    void compute_error_last(Tensor& in_expected)
    {
        int threads = 256;
        int blocks_num = (output_size + threads - 1) / threads;
        dim3 blocks(blocks_num, batch_size);

        compute_ej_last_layer << < blocks, threads >> > (output.data, output.gradient, in_expected.data, output_size, batch_size);
        cudaDeviceSynchronize();
    }

    void update_weights(float learning_rate, Tensor* temporal_input = nullptr)
    {
        int threads = 256;
        int blocks_num = (output_size + threads - 1) / threads;
        dim3 blocks(blocks_num, batch_size);

        update_weights_krnl << < blocks, threads >> > (weights.data, previous_layer->data, output.gradient, output.data, learning_rate, input_size, output_size, batch_size);
        cudaDeviceSynchronize();

        update_bias_krnl << < blocks, threads >> > (bias.data, output.gradient, learning_rate, output_size, batch_size);
        cudaDeviceSynchronize();
    }

    void zero_grad()
    {
        output.zero_grad();
        if (previous_layer)
            previous_layer->zero_grad();
    }
};


#endif