#ifndef POSITIONAL_ENCODING_CUH
#define POSITIONAL_ENCODING_CUH

#include "tensor.cuh"

__global__
void add_position(float* input, float* output,
				  float* POSITION_data,
				  size_t batch_size,
				  size_t patch_size,
				  size_t n_patches)
{
	int idx = (blockDim.x * blockIdx.x) + threadIdx.x;
	int batch_idx = blockIdx.y;

	if ((idx >= patch_size * n_patches) || (batch_idx >= batch_size))
		return;

	int start_idx = batch_idx * (patch_size * n_patches);

	output[start_idx + idx] = input[start_idx + idx] + POSITION_data[idx];
}

class PositionalEncoding
{
public:
	size_t batch_size, n_patches, patch_dim;
	Tensor* previous, output;

	PositionalEncoding(size_t in_n_patches, size_t in_patch_dim, Tensor* in_previous, size_t in_batch_size = 1) :
		batch_size(in_batch_size), n_patches(in_n_patches), patch_dim(in_patch_dim),
		previous(in_previous)
	{
		output.set_size(batch_size * n_patches * patch_dim);
		
		position_parameters.set_size(n_patches * patch_dim);
		position_parameters.set_random(0.1f);
	}

	void forward()
	{
		int threads = 256;
		int blocks_num = ((patch_dim * n_patches) + threads - 1) / threads;

		dim3 blocks(blocks_num, batch_size);
		add_position << < blocks, threads >> > (previous->data, output.data, position_parameters.data,
												batch_size, patch_dim, n_patches);

		cudaDeviceSynchronize();
	}
private:
	Tensor position_parameters;
};

#endif