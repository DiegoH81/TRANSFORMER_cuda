#pragma once

#include <iostream>
#include <vector>
#include <random>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "layer.cuh"
#include "utils.cuh"

__global__ void layer_norm_forward(float* input, float* output, float* x_hat, float* inv_std, float* gamma, float* beta, int dim_model, int seq_len, float eps) {
	extern __shared__ float shmem[];
	float* shmem_mean = shmem;
	float* shmem_var = shmem + dim_model;

	int tid = threadIdx.x;
	int img = blockIdx.x;
	int token = blockIdx.y;

	int token_offset = (img * seq_len + token) * dim_model;

	float x = input[token_offset + tid];
	shmem_mean[tid] = x;
	shmem_var[tid] = 0.0f;
	__syncthreads();

	for (int stride = blockDim.x / 2; stride > 0; stride /= 2) {
		if (tid < stride) {
			shmem_mean[tid] += shmem_mean[tid + stride];
		}
		__syncthreads();
	}
	float mean = shmem_mean[0] / dim_model;

	shmem_var[tid] = (x - mean) * (x - mean);
	__syncthreads();

	for (int stride = blockDim.x / 2; stride > 0; stride /= 2) {
		if (tid < stride) {
			shmem_var[tid] += shmem_var[tid + stride];
		}
		__syncthreads();
	}
	float var = shmem_var[0] / dim_model;

	float is = 1.0f / sqrtf(var + eps);

	if (tid == 0) {
		inv_std[img * seq_len + token] = is;
	}

	float x_norm = (x - mean) * is;
	x_hat[token_offset + tid] = x_norm;
	output[token_offset + tid] = gamma[tid] * x_norm + beta[tid];
}

__global__ void layer_norm_backward(float* d_out, float* x_hat, float* inv_std, float* gamma, float* d_input, float* d_gamma, float* d_beta, int dim_model, int seq_len) {
	extern __shared__ float shmem[];
	float* shmem_dout = shmem;
	float* shmem_dout_xhat = shmem + dim_model;

	int tid = threadIdx.x;
	int img = blockIdx.x;
	int token = blockIdx.y;

	int token_offset = (img * seq_len + token) * dim_model;

	float dout = d_out[token_offset + tid];
	float xh = x_hat[token_offset + tid];
	float is = inv_std[img * seq_len + token];

	atomicAdd(&d_gamma[tid], dout * xh);
	atomicAdd(&d_beta[tid], dout);

	shmem_dout[tid] = dout;
	shmem_dout_xhat[tid] = dout * xh;
	__syncthreads();

	for (int stride = blockDim.x / 2; stride > 0; stride /= 2) {
		if (tid < stride) {
			shmem_dout[tid] += shmem_dout[tid + stride];
			shmem_dout_xhat[tid] += shmem_dout_xhat[tid + stride];
		}
		__syncthreads();
	}

	float sum_dout = shmem_dout[0];
	float sum_dout_xhat = shmem_dout_xhat[0];

	float grad = is * (1.0f / dim_model) * gamma[tid] * (dim_model * dout - sum_dout - xh * sum_dout_xhat);

	d_input[token_offset + tid] = grad;
}

class LayerNorm {
public:
	Tensor gamma, beta;
	Tensor output, x_hat, inv_std;
	Tensor* previous;
	int dim_model, n_images, sequence_len;

public:
	LayerNorm(int in_dim_model, int in_seq_len, int in_n_images, Tensor* in_previous = nullptr) :
		sequence_len(in_seq_len),dim_model(in_dim_model), n_images(in_n_images), previous(in_previous) {

		output.set_size(n_images * sequence_len * dim_model);
		x_hat.set_size(n_images * sequence_len * dim_model);
		inv_std.set_size(n_images * sequence_len);

		gamma.set_size(dim_model);
		beta.set_size(dim_model);

		std::vector<float> ones(dim_model, 1.0f);
		gamma.load(ones);
	}
	void forward() {
		dim3 grid(n_images, sequence_len);
		int threads = dim_model;
		size_t shmem = 2 * dim_model * sizeof(float);

		layer_norm_forward << <grid, threads, shmem >> > (
			previous->data, output.data, x_hat.data, inv_std.data,
			gamma.data, beta.data, dim_model, sequence_len, 1e-5f
			);
		cudaDeviceSynchronize();
	}
	void backward() {
		dim3 grid(n_images, sequence_len);
		int threads = dim_model;
		size_t shmem = 2 * dim_model * sizeof(float);

		layer_norm_backward << <grid, threads, shmem >> > (
			output.gradient, x_hat.data, inv_std.data, gamma.data,
			previous->gradient, gamma.gradient, beta.gradient,
			dim_model, sequence_len
			);
		cudaDeviceSynchronize();
	}
	void update_weights(float updt) {
		int threads = 256;
		int blocks = (dim_model + threads - 1) / threads;
		sgd_update << <blocks, threads >> > (gamma.data, gamma.gradient, updt, dim_model);
		sgd_update << <blocks, threads >> > (beta.data, beta.gradient, updt, dim_model);
		cudaDeviceSynchronize();
	}
	void zero_grad() {
		output.zero_grad();
		x_hat.zero_grad();
		inv_std.zero_grad();
		gamma.zero_grad();
		beta.zero_grad();
	}

};