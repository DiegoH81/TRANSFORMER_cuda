#ifndef CLASSIFICATION_CUH
#define CLASSIFICATION_CUH

#include "tensor.cuh"
#include "layer.cuh"

__global__
void extract_CLS(float* encoder_input, float* CLS_output,
				 size_t batch_size,
				 size_t linear_dim,
				 size_t n_patches)
{
	int idx = (blockDim.x * blockIdx.x) + threadIdx.x;
	int batch_idx = blockIdx.y;

	if ((idx >= linear_dim) || (batch_idx >= batch_size))
		return;

	int encoder_idx = batch_idx * (linear_dim * n_patches);

	CLS_output[batch_idx * linear_dim + idx] = encoder_input[encoder_idx + idx];
}

__global__
void extract_CLS_backward(float* CLS_grad, float* encoder_grad,
						  size_t batch_size,
						  size_t linear_dim,
						  size_t n_patches)
{
	int idx = (blockDim.x * blockIdx.x) + threadIdx.x;
	int batch_idx = blockIdx.y;

	if ((idx >= linear_dim) || (batch_idx >= batch_size))
		return;

	int encoder_idx = batch_idx * (linear_dim * n_patches);


	encoder_grad[encoder_idx + idx] = CLS_grad[batch_idx * linear_dim + idx];
}

__global__
void softmax_forward(float* output, size_t output_size, size_t batch_size)
{
	int batch_idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (batch_idx >= batch_size)
		return;

	float* row = output + batch_idx * output_size;

	float max_val = row[0];
	for (size_t i = 1; i < output_size; i++)
		max_val = fmaxf(max_val, row[i]);

	float sum = 0.0f;
	for (size_t i = 0; i < output_size; i++)
	{
		row[i] = expf(row[i] - max_val);
		sum += row[i];
	}

	for (size_t i = 0; i < output_size; i++)
		row[i] /= sum;
}

class Classification
{
public:
	size_t batch_size, n_patches, linear_dim;
	Tensor* previous;
	Layer classification_layer;


	Classification(size_t in_n_patches, size_t in_linear_dim, Tensor* in_previous, size_t in_batch_size = 1):
		n_patches(in_n_patches), linear_dim(in_linear_dim), previous(in_previous), batch_size(in_batch_size),
		classification_layer(linear_dim, 10, batch_size, ActivationType::None, &CLS_data)
	{
		CLS_data.set_size(batch_size * linear_dim);
	}

	void forward()
	{
		int threads = 256;
		int blocks_num = (linear_dim + threads - 1) / threads;

		dim3 blocks(blocks_num, batch_size);
		extract_CLS << < blocks, threads >> > (previous->data, CLS_data.data, batch_size, linear_dim, n_patches);
		cudaDeviceSynchronize();

		classification_layer.forward();

		int sm_threads = 32;
		int sm_blocks = (batch_size + sm_threads - 1) / sm_threads;
		softmax_forward << < sm_blocks, sm_threads >> > (classification_layer.output.data, 10, batch_size);
		cudaDeviceSynchronize();
	}

	void backward(Tensor& in_expected, float in_learning_rate)
	{
		classification_layer.compute_error_last(in_expected);
		classification_layer.apply_derivative();
		classification_layer.update_weights(in_learning_rate);
		classification_layer.compute_error_intermediate();


		int threads = 256;
		int blocks_num = (linear_dim + threads - 1) / threads;

		dim3 blocks(blocks_num, batch_size);
		extract_CLS_backward << < blocks, threads >> > (CLS_data.gradient, previous->gradient, batch_size, linear_dim, n_patches);
		cudaDeviceSynchronize();
	}

	void zero_grad()
	{
		classification_layer.zero_grad();
		CLS_data.zero_grad();
	}

	void save_weights(std::ofstream& file)
	{
		classification_layer.save_weights(file);
	}

	void load_weights(std::ifstream& file)
	{
		classification_layer.load_weights(file);
	}

private:
	Tensor CLS_data;
};


#endif