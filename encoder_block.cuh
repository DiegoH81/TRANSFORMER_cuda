#ifndef ENCODER_BLOCK_CUH
#define ENCODER_BLOCK_CUH

#include "layer_norm.cuh"
#include "MHA.cuh"

__global__
void add_residual(float* original_input, float* attention_input,
				  float* output,
				  size_t batch_size,
				  size_t linear_dim,
				  size_t n_patches)
{
	int idx = (blockDim.x * blockIdx.x) + threadIdx.x;
	int batch_idx = blockIdx.y;

	if ((idx >= linear_dim * n_patches) || (batch_idx >= batch_size))
		return;

	int start_idx = batch_idx * (linear_dim * n_patches);

	output[start_idx + idx] = original_input[start_idx + idx] + attention_input[start_idx + idx];
}

__global__
void add_residual_backward(float* original_grad, float* attention_grad,
						   float* residual_grad,
						   size_t batch_size,
						   size_t linear_dim,
						   size_t n_patches)
{
	int idx = (blockDim.x * blockIdx.x) + threadIdx.x;
	int batch_idx = blockIdx.y;

	if ((idx >= linear_dim * n_patches) || (batch_idx >= batch_size))
		return;

	int start_idx = batch_idx * (linear_dim * n_patches);

	original_grad[start_idx + idx] += residual_grad[start_idx + idx];
	attention_grad[start_idx + idx] += residual_grad[start_idx + idx];
}

class EncoderBlock
{
public:
	size_t n_patches, linear_dim, batch_size;
	Tensor output, *previous;

	EncoderBlock(size_t in_n_patches, size_t in_linear_dim, Tensor* in_previous, size_t in_batch_size = 1) :
		n_patches(in_n_patches), linear_dim(in_linear_dim), previous(in_previous), batch_size(in_batch_size),
		layer_norm_part(n_patches, linear_dim, previous, batch_size),
		mha_part(n_patches, linear_dim, &layer_norm_part.output, batch_size),
		layer_post_residual(n_patches, linear_dim, &residual_output, batch_size),
		up_proj(linear_dim, linear_dim * 4, batch_size * n_patches, ActivationType::ReLu, &layer_post_residual.output),
		down_proj(linear_dim * 4, linear_dim, batch_size* n_patches, ActivationType::None, &up_proj.output)
	{
		residual_output.set_size(batch_size * n_patches * linear_dim);
		output.set_size(batch_size * n_patches * linear_dim);
	}

	void forward()
	{
		// First Part
		layer_norm_part.forward();
		mha_part.forward();

		int threads = 256;
		int blocks_num = ((linear_dim * n_patches) + threads - 1) / threads;

		dim3 blocks(blocks_num, batch_size);
		add_residual << < blocks, threads >> > (previous->data, mha_part.W_out.output.data, residual_output.data, batch_size, linear_dim, n_patches);
		cudaDeviceSynchronize();

		layer_post_residual.forward();

		// Second part
		up_proj.forward();
		down_proj.forward();

		add_residual << < blocks, threads >> > (residual_output.data, down_proj.output.data, output.data, batch_size, linear_dim, n_patches);
		cudaDeviceSynchronize();
	}

	void backward(float in_learning_rate)
	{
		int threads = 256;
		int blocks_num = ((linear_dim * n_patches) + threads - 1) / threads;

		dim3 blocks(blocks_num, batch_size);

		// Residual 2
		add_residual_backward << < blocks, threads >> > (residual_output.gradient, down_proj.output.gradient, output.gradient, batch_size, linear_dim, n_patches);
		cudaDeviceSynchronize();

		// Layers
		down_proj.compute_error_intermediate();
		down_proj.apply_derivative();
		down_proj.update_weights(in_learning_rate);

		up_proj.compute_error_intermediate();
		up_proj.apply_derivative();
		up_proj.update_weights(in_learning_rate);

		layer_post_residual.backward(in_learning_rate);

		// Residual 1
		add_residual_backward << < blocks, threads >> > (previous->gradient, mha_part.W_out.output.gradient, residual_output.gradient, batch_size, linear_dim, n_patches);
		cudaDeviceSynchronize();

		mha_part.backward(in_learning_rate);
		layer_norm_part.backward(in_learning_rate);
	}

	void zero_grad()
	{
		up_proj.zero_grad();
		down_proj.zero_grad();
		layer_post_residual.zero_grad();
		mha_part.zero_grad();
		layer_post_residual.zero_grad();
	}

private:
	Tensor residual_output;
	LayerNorm layer_norm_part, layer_post_residual;
	Layer up_proj, down_proj;
	MHA mha_part;
};

#endif