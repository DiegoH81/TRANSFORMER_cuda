#ifndef LAYER_NORM_CUH
#define LAYER_NORM_CUH

#define EPSILON 1e-5

#include "tensor.cuh"

__global__
void get_media(float* input, float* media_outputs,
			   size_t batch_size, size_t linear_dim, size_t n_patches)
{
	int idx = (blockDim.x * blockIdx.x) + threadIdx.x;
	int batch_idx = blockIdx.y;

	if ((idx >= linear_dim * n_patches) || (batch_idx >= batch_size))
		return;

	int start_idx = batch_idx * (linear_dim * n_patches);

	int current_patch = idx / linear_dim;
	int patch_idx = (batch_idx * n_patches) + current_patch;
	atomicAdd(&media_outputs[patch_idx], input[start_idx + idx] / float(linear_dim));
}

__global__
void get_varianza(float* input, float* media, float* varianza_outputs,
				  size_t batch_size, size_t linear_dim, size_t n_patches)
{
	int idx = (blockDim.x * blockIdx.x) + threadIdx.x;
	int batch_idx = blockIdx.y;

	if ((idx >= linear_dim * n_patches) || (batch_idx >= batch_size))
		return;

	int start_idx = batch_idx * (linear_dim * n_patches);

	int current_patch = idx / linear_dim;
	int patch_idx = (batch_idx * n_patches) + current_patch;

	float varianza = (pow((input[start_idx + idx] - media[patch_idx]), 2)) / float(linear_dim);

	atomicAdd(&varianza_outputs[patch_idx], varianza);
}

__global__
void add_layer_norm(float* input, float* output,
					float* beta, float* gamma,
					float* media, float* varianza,
					size_t batch_size, size_t linear_dim, size_t n_patches)
{
	int idx = (blockDim.x * blockIdx.x) + threadIdx.x;
	int batch_idx = blockIdx.y;

	if ((idx >= linear_dim * n_patches) || (batch_idx >= batch_size))
		return;

	int start_idx = batch_idx * (linear_dim * n_patches);
	int current_patch = idx / linear_dim;
	int patch_idx = (batch_idx * n_patches) + current_patch;


	float x_norm = (input[start_idx + idx] - media[patch_idx]) / sqrt(varianza[patch_idx] + EPSILON);

	output[start_idx + idx] = gamma[idx] * x_norm + beta[idx];
}

class LayerNorm
{
public:
	size_t batch_size, n_patches, linear_dim;
	Tensor output, *previous;

	LayerNorm(size_t in_n_patches, size_t in_linear_dim, Tensor* in_previous, size_t in_batch_size = 1) :
		batch_size(in_batch_size), n_patches(in_n_patches), linear_dim(in_linear_dim),
		previous(in_previous)
	{
		output.set_size(batch_size * n_patches * linear_dim);

		gamma.set_size(linear_dim);
		gamma.load(std::vector<float>(linear_dim, 1.0f));

		beta.set_size(linear_dim);

		media.set_size(batch_size * n_patches);
		varianza.set_size(batch_size * n_patches);
	}

	void forward()
	{
		int threads = 256;
		int blocks_num = ((linear_dim * n_patches) + threads - 1) / threads;

		dim3 blocks(blocks_num, batch_size);
		get_media << < blocks, threads >> > (previous->data, media.data, batch_size, linear_dim, n_patches);
		cudaDeviceSynchronize();

		get_varianza << < blocks, threads >> > (previous->data, media.data, varianza.data,
												batch_size, linear_dim, n_patches);
		cudaDeviceSynchronize();

		add_layer_norm << < blocks, threads >> > (previous->data, output.data,
												  beta.data, gamma.data, media.data, varianza.data,
												  batch_size, linear_dim, n_patches);
		cudaDeviceSynchronize();

		media.reset_data();
		varianza.reset_data();
	}


private:
	Tensor gamma, beta, media, varianza;
};

#endif