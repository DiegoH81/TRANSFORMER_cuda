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
void get_d_gamma_beta(float* d_gamma, float* d_beta, float* output_grad, float* input,
					  float* beta, float* gamma, float* media, float* varianza,
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
	
	int dim_idx = idx % linear_dim;
	
	atomicAdd(&d_gamma[dim_idx], output_grad[start_idx + idx] * x_norm);
	atomicAdd(&d_beta[dim_idx], output_grad[start_idx + idx]);
}

__global__
void layer_norm_backward(float* input_grad, float* output_grad, 
						 float* sum_dxhat, float* sum_dxhat_xhat,
						 float* input,
						 float* beta, float* gamma, float* media, float* varianza,
						 size_t batch_size, size_t linear_dim, size_t n_patches)
{
	int idx = (blockDim.x * blockIdx.x) + threadIdx.x;
	int batch_idx = blockIdx.y;

	if ((idx >= linear_dim * n_patches) || (batch_idx >= batch_size))
		return;

	int start_idx = batch_idx * (linear_dim * n_patches);

	int current_patch = idx / linear_dim;
	int patch_idx = (batch_idx * n_patches) + current_patch;


	float std_val = sqrt(varianza[patch_idx] + EPSILON);
	float x_norm = (input[start_idx + idx] - media[patch_idx]) / std_val;
	int dim_idx = idx % linear_dim;

	float dxhat = output_grad[start_idx + idx] * gamma[dim_idx];
	float d = (float)linear_dim;
	float dx = (dxhat - sum_dxhat[patch_idx] / d - x_norm * sum_dxhat_xhat[patch_idx] / d) / std_val;


	input_grad[start_idx + idx] += dx;
}

__global__
void get_dxhat_sums(float* sum_dxhat, float* sum_dxhat_xhat, float* output_grad, float* input,
	float* beta, float* gamma, float* media, float* varianza,
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
	int dim_idx = idx % linear_dim;
	
	float dxhat = output_grad[start_idx + idx] * gamma[dim_idx];

	atomicAdd(&sum_dxhat[patch_idx], dxhat);
	atomicAdd(&sum_dxhat_xhat[patch_idx], dxhat * x_norm);
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

	int dim_idx = idx % linear_dim;
	output[start_idx + idx] = gamma[dim_idx] * x_norm + beta[dim_idx];
}


__global__
void update_weights(float* input_data, float* input_gradient, float learning_rate,
						 size_t linear_dim, size_t batch_size, size_t n_patches)
{
	int idx = (blockDim.x * blockIdx.x) + threadIdx.x;
	if (idx >= linear_dim)
		return;

	input_data[idx] += (learning_rate * clip_grad(input_gradient[idx])) / float(n_patches);

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

		sum_dxhat.set_size(batch_size * n_patches);
		sum_dxhat_xhat.set_size(batch_size * n_patches);
	}

	void forward()
	{
		media.reset_data();
		varianza.reset_data();

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
	}

	void backward(float in_learning_rate)
	{
		sum_dxhat.reset_data();
		sum_dxhat_xhat.reset_data();

		float effective_lr = in_learning_rate / (float)(batch_size * linear_dim);

		int threads = 256;
		int blocks_num = ((linear_dim * n_patches) + threads - 1) / threads;

		dim3 blocks(blocks_num, batch_size);
		
		get_d_gamma_beta << < blocks, threads >> > (gamma.gradient, beta.gradient, output.gradient,
													previous->data, beta.data, gamma.data, media.data,
													varianza.data, batch_size, linear_dim, n_patches);
		cudaDeviceSynchronize();
		
		get_dxhat_sums << < blocks, threads >> > (sum_dxhat.data, sum_dxhat_xhat.data, output.gradient,
												  previous->data, beta.data, gamma.data,
												  media.data, varianza.data,
												  batch_size, linear_dim, n_patches);
		cudaDeviceSynchronize();

		layer_norm_backward << < blocks, threads >> > (previous->gradient, output.gradient,
								 sum_dxhat.data, sum_dxhat_xhat.data, previous->data, beta.data, gamma.data, media.data, varianza.data,
								 batch_size, linear_dim, n_patches);
		cudaDeviceSynchronize();

		// Updating weights
		blocks_num = (linear_dim + threads - 1) / threads;
		update_weights << < blocks_num, threads >> > (beta.data, beta.gradient, effective_lr, linear_dim, batch_size, n_patches);
		cudaDeviceSynchronize();

		update_weights << < blocks_num, threads >> > (gamma.data, gamma.gradient, effective_lr, linear_dim, batch_size, n_patches);
		cudaDeviceSynchronize();
	}

	void zero_grad()
	{
		beta.zero_grad();
		gamma.zero_grad();
		output.zero_grad();
	}

	void save_weights(std::ofstream& file)
	{
		write_tensor(file, gamma);
		write_tensor(file, beta);
	}

	void load_weights(std::ifstream& file)
	{
		read_tensor(file, gamma);
		read_tensor(file, beta);
	}

private:
	Tensor gamma, beta, media, varianza,
		   sum_dxhat, sum_dxhat_xhat;
};

#endif