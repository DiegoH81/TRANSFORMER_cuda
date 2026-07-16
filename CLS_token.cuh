#ifndef CLS_TOKEN
#define CLS_TOKEN

#include "patch_embedding.cuh"

__global__
void add_cls(float* input, float* output,
			 float* CLS_data,
			 size_t batch_size,
			 size_t patch_size,
			 size_t n_patches)
{
	int idx = (blockDim.x * blockIdx.x) + threadIdx.x;
	int batch_idx = blockIdx.y;

	int total_size = patch_size * n_patches;
	if (idx >= total_size || batch_idx >= batch_size)
		return;

	int initial_idx = batch_idx * (total_size + patch_size);
	output[initial_idx + patch_size + idx] = input[(batch_idx * total_size) + idx];

	if (idx == 0)
	{
		for (int i = 0; i < patch_size; i++)
			output[initial_idx + i] = CLS_data[i];
	}
}

__global__
void add_cls_backward(float* input_grad, float* output_grad,
					  float* CLS_data_grad,
					  size_t batch_size,
					  size_t patch_size,
					  size_t n_patches)
{
	int idx = (blockDim.x * blockIdx.x) + threadIdx.x;
	int batch_idx = blockIdx.y;

	int total_size = patch_size * n_patches;
	if (idx >= total_size || batch_idx >= batch_size)
		return;

	int initial_idx = batch_idx * (total_size + patch_size);
	input_grad[(batch_idx * total_size) + idx] = output_grad[initial_idx + patch_size + idx];

	if (idx == 0)
	{
		for (int i = 0; i < patch_size; i++)
			atomicAdd(&CLS_data_grad[i], output_grad[initial_idx + i]);
	}
}

__global__
void update_weights_CLS(float* CLS_data, float* CLS_grad, float learning_rate, size_t patch_dim, size_t batch_size)
{
	int idx = (blockDim.x * blockIdx.x) + threadIdx.x;
	if (idx >= patch_dim)
		return;

	CLS_data[idx] -= learning_rate * (CLS_grad[idx] / float(batch_size));
}

class CLS_token
{
public:
	size_t batch_size, n_patches, patch_dim;
	Tensor output, *previous;

	CLS_token(size_t in_n_patches, size_t in_patch_dim, Tensor* in_previous, size_t in_batch_size = 1) :
		batch_size(in_batch_size), n_patches(in_n_patches), patch_dim(in_patch_dim),
		previous(in_previous)
	{
		int total_size = (n_patches + 1) * patch_dim * batch_size;
		output.set_size(total_size);

		CLS_parameters.set_size(patch_dim);
		CLS_parameters.set_random(0.1f);
	}

	void forward()
	{
		int threads = 256;
		int blocks_num = ((patch_dim * n_patches) + threads - 1) / threads;

		dim3 blocks(blocks_num, batch_size);

		add_cls << < blocks, threads >> > (previous->data, output.data,
										   CLS_parameters.data,
										   batch_size, patch_dim, n_patches);
		cudaDeviceSynchronize();
	}

	void backward(float in_learning_rate)
	{
		int threads = 256;
		int blocks_num = ((patch_dim * n_patches) + threads - 1) / threads;

		dim3 blocks(blocks_num, batch_size);

		add_cls_backward <<< blocks, threads >>> (previous->gradient, output.gradient, CLS_parameters.gradient, 
												  batch_size, patch_dim, n_patches);
		cudaDeviceSynchronize();

		blocks_num = (patch_dim + threads - 1) / threads;
		update_weights_CLS <<< blocks_num, threads >>> (CLS_parameters.data, CLS_parameters.gradient, in_learning_rate,
														patch_dim, batch_size);
		cudaDeviceSynchronize();
	}

	void zero_grad()
	{
		CLS_parameters.zero_grad();
	}
private:
	Tensor CLS_parameters;
};

#endif