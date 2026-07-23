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

__global__
void add_position_backward(float* input_grad, float* output_grad,
						   float* POSITION_data_grad,
						   size_t batch_size,
						   size_t patch_size,
						   size_t n_patches)
{
	int idx = (blockDim.x * blockIdx.x) + threadIdx.x;
	int batch_idx = blockIdx.y;

	if ((idx >= patch_size * n_patches) || (batch_idx >= batch_size))
		return;

	int start_idx = batch_idx * (patch_size * n_patches);

	input_grad[start_idx + idx] = output_grad[start_idx + idx];
	atomicAdd(&POSITION_data_grad[idx], output_grad[start_idx + idx]);
}

__global__
void update_weights_pos(float* pos_data, float* pos_grad, float learning_rate,
					size_t n_patches, size_t linear_dim)
{
	int idx = (blockDim.x * blockIdx.x) + threadIdx.x;
	if (idx >= n_patches * linear_dim)
		return;

	pos_data[idx] += learning_rate * clip_grad(pos_grad[idx]);

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

	void backward(float in_learning_rate)
	{
		int threads = 256;
		int blocks_num = ((patch_dim * n_patches) + threads - 1) / threads;

		dim3 blocks(blocks_num, batch_size);
		add_position_backward <<< blocks, threads>>> (previous->gradient, output.gradient, 
													  position_parameters.gradient,
													  batch_size, patch_dim, n_patches);
		cudaDeviceSynchronize();

		float effective_lr = in_learning_rate / (float) batch_size;
		update_weights_pos << < blocks_num, threads >> > (position_parameters.data,
														  position_parameters.gradient, effective_lr,
														  n_patches, patch_dim);
		cudaDeviceSynchronize();
	}

	void zero_grad()
	{
		position_parameters.zero_grad();
		output.zero_grad();
	}

	void save_weights(std::ofstream& file)
	{
		write_tensor(file, position_parameters);
	}

	void load_weights(std::ifstream& file)
	{
		read_tensor(file, position_parameters);
	}
private:
	Tensor position_parameters;
};

#endif