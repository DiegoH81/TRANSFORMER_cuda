#ifndef PATCH_EMBEDDING_CUH
#define PATCH_EMBEDDING_CUH

#include "tensor.cuh"

__global__
void re_shape_image(float* input, float *output,
					size_t batch_size,
					size_t total_image_size,
					size_t image_size, size_t patch_x_num, size_t patch_dim)
{
	int idx = (blockIdx.x * blockDim.x) + threadIdx.x;
	int batch_idx = blockIdx.y;

	if (idx >= (image_size * image_size) || (batch_idx >= batch_size))
		return;

	int row = idx / image_size;
	int col = idx % image_size;

	int patch_row = row / patch_dim;
	int patch_col = col/ patch_dim;

	int patch_inner_row = row % patch_dim;
	int patch_inner_col = col % patch_dim;


	int square_size = patch_dim * patch_dim;
	int initial_idx = (patch_x_num * patch_row + patch_col) * square_size;

	int new_idx = initial_idx + (patch_inner_row * patch_dim + patch_inner_col);

	output[(total_image_size * batch_idx) + new_idx] = input[(total_image_size * batch_idx) + idx];
}


class PatchEmbedding
{
public:
	PatchEmbedding(size_t in_total_image_size, size_t in_p_dim, size_t in_batch_size = 1) :
		total_image_size(in_total_image_size),
		patch_dim(in_p_dim), previous(nullptr), output(),
		batch_size(in_batch_size),
		n_patches(total_image_size / (in_p_dim * in_p_dim))
	{
		size_t total_size = (patch_dim * patch_dim) * n_patches * batch_size;

		output.set_size(total_size);
	}

	void forward(Tensor* in_previous)
	{
		previous = in_previous;

		int threads = 256;
		int block_num = (total_image_size + threads - 1)/ threads;
		
		dim3 blocks(block_num, batch_size);

		re_shape_image << < blocks, threads >> > (previous->data,
		            							  output.data,
												  batch_size,
												  total_image_size,
												  28,
												  4,
												  patch_dim);
		cudaDeviceSynchronize();
	}

private:
	Tensor output, *previous;
	size_t total_image_size, patch_dim, n_patches, batch_size;
};

#endif