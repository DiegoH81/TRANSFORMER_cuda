#ifndef ACTIVATION_FUNCTION_H
#define ACTIVATION_FUNCTION_H

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "layer.cuh"
#include "tensor.cuh"

enum class ActivationType
{
	None,
	ReLu
};

__global__
void ReLu_forward(float *input, size_t in_size, size_t batch_size)
{
	int output_idx = blockIdx.x * blockDim.x + threadIdx.x;
	int batch_idx = blockIdx.y;

	if (output_idx >= in_size || batch_idx >= batch_size)
		return;


	input[(batch_idx * in_size) + output_idx] = fmaxf(0.0f, input[(batch_idx * in_size) + output_idx]);
}

__global__
void ReLu_derivative(float* input_grads, float* output_data, size_t in_size, size_t batch_size)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	int batch_idx = blockIdx.y;

	if (idx >= in_size || batch_idx >= batch_size)
		return;

	int new_idx = (batch_idx * in_size) + idx;

	if (output_data[new_idx] > 0)
		input_grads[new_idx] *= 1.0f;
	else
		input_grads[new_idx] *= 0.0f;
}

#endif