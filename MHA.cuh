#ifndef MHA_CUH
#define MHA_CUH

#include "tensor.cuh"
#include "layer.cuh"

__global__
void mat_mul_and_scale(float* query, float* key, float* score_matrix,
					   float sqrt_dim,
					   size_t n_heads,
					   size_t head_dim,
					   size_t batch_size, size_t linear_dim, size_t n_patches)
{
	int matrix_idx = (blockDim.x * blockIdx.x) + threadIdx.x;
	int batch_idx = blockIdx.y;
	int head_idx = blockIdx.z;

	if ((matrix_idx >= n_patches * n_patches) || (batch_idx >= batch_size) || (head_idx >= n_heads))
		return;


	int row_idx = matrix_idx / n_patches; // Which patch
	int col_idx = matrix_idx % n_patches; // Which patch

	int start_idx_query = (batch_idx * linear_dim * n_patches) + (row_idx * linear_dim) + (head_idx * head_dim);
	int start_idx_key = (batch_idx * linear_dim * n_patches) + (col_idx * linear_dim) + (head_idx * head_dim);

	int output_mat_idx = (batch_idx * n_heads * n_patches * n_patches) + (head_idx * n_patches * n_patches) +
							(row_idx * n_patches) + col_idx;

	score_matrix[output_mat_idx] = 0.0f;
	for (int i = 0; i < head_dim; i++)
		score_matrix[output_mat_idx] += query[start_idx_query + i] * key[start_idx_key + i];

	score_matrix[output_mat_idx] /= sqrt_dim;
}

__global__
void mat_mul_backward(float* query, float* query_grad,
					  float* key, float* key_grad, float* score_matrix_grad,
					  float sqrt_dim,
					  size_t n_heads,
					  size_t head_dim,
					  size_t batch_size, size_t linear_dim, size_t n_patches)
{
	int matrix_idx = (blockDim.x * blockIdx.x) + threadIdx.x;
	int batch_idx = blockIdx.y;
	int head_idx = blockIdx.z;

	if ((matrix_idx >= n_patches * n_patches) || (batch_idx >= batch_size) || (head_idx >= n_heads))
		return;


	int row_idx = matrix_idx / n_patches; // Which patch
	int col_idx = matrix_idx % n_patches; // Which patch

	int start_idx_query = (batch_idx * linear_dim * n_patches) + (row_idx * linear_dim) + (head_idx * head_dim);
	int start_idx_key = (batch_idx * linear_dim * n_patches) + (col_idx * linear_dim) + (head_idx * head_dim);

	int output_mat_idx = (batch_idx * n_heads * n_patches * n_patches) + (head_idx * n_patches * n_patches) +
		(row_idx * n_patches) + col_idx;

	float score_g = score_matrix_grad[output_mat_idx] / sqrt_dim;
	for (int i = 0; i < head_dim; i++)
	{
		atomicAdd(&query_grad[start_idx_query + i], key[start_idx_key + i] * score_g);
		atomicAdd(&key_grad[start_idx_key + i], query[start_idx_query + i] * score_g);
	}
}

__global__
void row_max(float* score_matrix, float* max_out,
	size_t n_heads, size_t head_dim,
	size_t batch_size, size_t linear_dim, size_t n_patches)
{
	int row_idx = blockIdx.x;   // una fila por bloque, simple
	int batch_idx = blockIdx.y;
	int head_idx = blockIdx.z;

	if (row_idx >= n_patches || batch_idx >= batch_size || head_idx >= n_heads)
		return;

	int base = (batch_idx * n_heads * n_patches * n_patches) + (head_idx * n_patches * n_patches) + (row_idx * n_patches);

	float m = -FLT_MAX;
	for (int c = 0; c < n_patches; c++)
		m = fmaxf(m, score_matrix[base + c]);

	int acc_idx = (batch_idx * n_heads * n_patches) + (head_idx * n_patches) + row_idx;
	max_out[acc_idx] = m;
}

__global__
void exp_matrix(float* score_matrix,
				float* acumulator,
				float* max_vals,
				size_t n_heads,
				size_t head_dim,
				size_t batch_size, size_t linear_dim, size_t n_patches)
{
	int matrix_idx = (blockDim.x * blockIdx.x) + threadIdx.x;
	int batch_idx = blockIdx.y;
	int head_idx = blockIdx.z;

	if ((matrix_idx >= n_patches * n_patches) || (batch_idx >= batch_size) || (head_idx >= n_heads))
		return;


	int row_idx = matrix_idx / n_patches; // Which patch
	int col_idx = matrix_idx % n_patches; // Which patch

	int mat_idx = (batch_idx * n_heads * n_patches * n_patches) + (head_idx * n_patches * n_patches) +
			      (row_idx * n_patches) + col_idx;

	int acumulator_idx = (batch_idx * n_heads * n_patches) + (head_idx * n_patches) + row_idx;


	score_matrix[mat_idx] = exp(score_matrix[mat_idx] - max_vals[acumulator_idx]);

	atomicAdd(&acumulator[acumulator_idx], score_matrix[mat_idx]);
}

__global__
void div_matrix(float* score_matrix,
	float* acumulator,
	size_t n_heads,
	size_t head_dim,
	size_t batch_size, size_t linear_dim, size_t n_patches)
{
	int matrix_idx = (blockDim.x * blockIdx.x) + threadIdx.x;
	int batch_idx = blockIdx.y;
	int head_idx = blockIdx.z;

	if ((matrix_idx >= n_patches * n_patches) || (batch_idx >= batch_size) || (head_idx >= n_heads))
		return;


	int row_idx = matrix_idx / n_patches; // Which patch
	int col_idx = matrix_idx % n_patches; // Which patch

	int mat_idx = (batch_idx * n_heads * n_patches * n_patches) + (head_idx * n_patches * n_patches) +
		(row_idx * n_patches) + col_idx;

	int acumulator_idx = (batch_idx * n_heads * n_patches) + (head_idx * n_patches) + row_idx;

	score_matrix[mat_idx] /= acumulator[acumulator_idx];
}

__global__
void sum_matrix(float* score_matrix,
				float* output_grad,
				float* acumulator,
				size_t n_heads,
				size_t head_dim,
				size_t batch_size, size_t linear_dim, size_t n_patches)
{
	int matrix_idx = (blockDim.x * blockIdx.x) + threadIdx.x;
	int batch_idx = blockIdx.y;
	int head_idx = blockIdx.z;

	if ((matrix_idx >= n_patches * n_patches) || (batch_idx >= batch_size) || (head_idx >= n_heads))
		return;


	int row_idx = matrix_idx / n_patches; // Which patch
	int col_idx = matrix_idx % n_patches; // Which patch

	int mat_idx = (batch_idx * n_heads * n_patches * n_patches) + (head_idx * n_patches * n_patches) +
		(row_idx * n_patches) + col_idx;

	int acumulator_idx = (batch_idx * n_heads * n_patches) + (head_idx * n_patches) + row_idx;

	atomicAdd(&acumulator[acumulator_idx], score_matrix[mat_idx] * output_grad[mat_idx]);
	
}

__global__
void matrix_adjust(float* score_matrix,
				   float* score_matrix_grad,
				   float* acumulator,
				   size_t n_heads,
				   size_t head_dim,
				   size_t batch_size, size_t linear_dim, size_t n_patches)
{
	int matrix_idx = (blockDim.x * blockIdx.x) + threadIdx.x;
	int batch_idx = blockIdx.y;
	int head_idx = blockIdx.z;

	if ((matrix_idx >= n_patches * n_patches) || (batch_idx >= batch_size) || (head_idx >= n_heads))
		return;


	int row_idx = matrix_idx / n_patches; // Which patch
	int col_idx = matrix_idx % n_patches; // Which patch

	int mat_idx = (batch_idx * n_heads * n_patches * n_patches) + (head_idx * n_patches * n_patches) +
		(row_idx * n_patches) + col_idx;

	int acumulator_idx = (batch_idx * n_heads * n_patches) + (head_idx * n_patches) + row_idx;


	score_matrix_grad[mat_idx] = score_matrix[mat_idx] * (score_matrix_grad[mat_idx] - acumulator[acumulator_idx]);
}

__global__
void multiply_value(float* score_matrix,
					float* value_vector,
					float* output,
					size_t n_heads,
					size_t head_dim,
					size_t batch_size, size_t linear_dim, size_t n_patches)
{
	int matrix_idx = (blockDim.x * blockIdx.x) + threadIdx.x;
	int batch_idx = blockIdx.y;

	if ((matrix_idx >= n_patches * linear_dim) || (batch_idx >= batch_size))
		return;


	int row_idx = matrix_idx / linear_dim;
	int col_idx = matrix_idx % linear_dim;

	int head_idx = col_idx / head_dim; // Which matrix to go in score matrix

	int value_idx = (batch_idx * n_patches * linear_dim) + (head_idx * head_dim) + (col_idx % head_dim);
	int score_matrix_idx = (batch_idx * n_heads * n_patches * n_patches) + (head_idx * n_patches * n_patches) + (row_idx * n_patches);
	int output_idx = (batch_idx * n_patches * linear_dim) + (row_idx * linear_dim) + col_idx;

	output[output_idx] = 0.0f;
	for (int i = 0; i < n_patches; i++)
		output[output_idx] += score_matrix[score_matrix_idx + i] * value_vector[value_idx + (i * linear_dim)];	
}

__global__
void multiply_value_backward(float* score_matrix,
							 float* score_matrix_grad,
							 float* value_vector,
							 float* value_vector_grad,
						     float* output_grad,
							 size_t n_heads,
							 size_t head_dim,
							 size_t batch_size, size_t linear_dim, size_t n_patches)
{
	int matrix_idx = (blockDim.x * blockIdx.x) + threadIdx.x;
	int batch_idx = blockIdx.y;

	if ((matrix_idx >= n_patches * linear_dim) || (batch_idx >= batch_size))
		return;


	int row_idx = matrix_idx / linear_dim;
	int col_idx = matrix_idx % linear_dim;

	int head_idx = col_idx / head_dim; // Which matrix to go in score matrix

	int value_idx = (batch_idx * n_patches * linear_dim) + (head_idx * head_dim) + (col_idx % head_dim);
	int score_matrix_idx = (batch_idx * n_heads * n_patches * n_patches) + (head_idx * n_patches * n_patches) + (row_idx * n_patches);
	int output_idx = (batch_idx * n_patches * linear_dim) + (row_idx * linear_dim) + col_idx;

	
	float out_g = output_grad[output_idx];
	for (int i = 0; i < n_patches; i++)
	{
		atomicAdd(&score_matrix_grad[score_matrix_idx + i], value_vector[value_idx + (i * linear_dim)] * out_g);
		atomicAdd(&value_vector_grad[value_idx + (i * linear_dim)], score_matrix[score_matrix_idx + i] * out_g);
	}
}

class MHA
{
public:
	size_t batch_size, n_patches, linear_dim, num_heads, head_dim;
	Tensor score_matrix, acumulator_exp, row_max_vals, attention_output, *previous;
	Layer W_out;
	

	MHA(size_t in_n_patches, size_t in_linear_dim, Tensor* in_previous, size_t in_batch_size = 1) :
		batch_size(in_batch_size), n_patches(in_n_patches), linear_dim(in_linear_dim),
		previous(in_previous),
		num_heads(8), head_dim(linear_dim / num_heads),
		W_query(linear_dim, linear_dim, batch_size * n_patches, ActivationType::None, previous),
		W_key(linear_dim, linear_dim, batch_size* n_patches, ActivationType::None, previous),
		W_value(linear_dim, linear_dim, batch_size * n_patches, ActivationType::None, previous),
		W_out(linear_dim, linear_dim, batch_size* n_patches, ActivationType::None, &attention_output)
	{
		score_matrix.set_size(batch_size * num_heads * n_patches * n_patches);
		acumulator_exp.set_size(batch_size * num_heads * n_patches);
		row_max_vals.set_size(batch_size * num_heads * n_patches);

		attention_output.set_size(batch_size * (n_patches * linear_dim)); // One output mat per Data
	}

	void forward()
	{
		W_query.forward();
		W_key.forward();
		W_value.forward();
		acumulator_exp.reset_data();

		int threads = 256;
		int blocks_num = ((n_patches * n_patches) + threads - 1) / threads;

		dim3 blocks(blocks_num, batch_size, num_heads);

		float d_root = std::sqrt(head_dim);
		mat_mul_and_scale << < blocks, threads >> > (W_query.output.data,
													 W_key.output.data, score_matrix.data,
													 d_root, num_heads,
													 head_dim, batch_size, linear_dim, n_patches);
		cudaDeviceSynchronize();

		// Softmax PART
		dim3 max_blocks(n_patches, batch_size, num_heads);
		row_max << < max_blocks, 1 >> > (score_matrix.data, row_max_vals.data, 
										 num_heads, head_dim, batch_size, linear_dim, n_patches);
		cudaDeviceSynchronize();

		exp_matrix << < blocks, threads >> > (score_matrix.data, acumulator_exp.data,
											  row_max_vals.data, num_heads, head_dim, batch_size, linear_dim, n_patches);
		cudaDeviceSynchronize();
		
		div_matrix << < blocks, threads >> > (score_matrix.data, acumulator_exp.data,
											  num_heads, head_dim, batch_size, linear_dim, n_patches);
		cudaDeviceSynchronize();

		// Final mul
		threads = 256;
		blocks_num = ((linear_dim * n_patches) + threads - 1) / threads;

		blocks = dim3(blocks_num, batch_size);
		multiply_value << < blocks, threads >> > (score_matrix.data, W_value.output.data,
												  attention_output.data, num_heads, head_dim, batch_size,
												  linear_dim, n_patches);
		cudaDeviceSynchronize();

		W_out.forward();
	}

	void backward(float in_learning_rate)
	{
		float effective_lr = in_learning_rate / (float)(batch_size * linear_dim);


		W_out.apply_derivative();
		W_out.update_weights(effective_lr);
		W_out.compute_error_intermediate();

		int threads = 256;
		int blocks_num = ((linear_dim * n_patches) + threads - 1) / threads;

		dim3 blocks = dim3(blocks_num, batch_size);
		multiply_value_backward << < blocks, threads >> > (score_matrix.data, score_matrix.gradient,
			 											   W_value.output.data, W_value.output.gradient,
														   attention_output.gradient, num_heads, head_dim, batch_size,
														   linear_dim, n_patches);
		cudaDeviceSynchronize();

		// Softmax
		acumulator_exp.reset_data();
		threads = 256;
		blocks_num = ((n_patches * n_patches) + threads - 1) / threads;

		blocks = dim3(blocks_num, batch_size, num_heads);
		sum_matrix << < blocks, threads >> > (score_matrix.data, score_matrix.gradient, acumulator_exp.data,
											  num_heads, head_dim, batch_size, linear_dim, n_patches);
		cudaDeviceSynchronize();

		matrix_adjust << < blocks, threads >> > (score_matrix.data, score_matrix.gradient,
												 acumulator_exp.data,
												 num_heads, head_dim, batch_size, linear_dim, n_patches);
		cudaDeviceSynchronize();

		float d_root = std::sqrt(head_dim);
		mat_mul_backward << < blocks, threads >> > (W_query.output.data, W_query.output.gradient,
													W_key.output.data, W_key.output.gradient,
													score_matrix.gradient, d_root,
													num_heads, head_dim, batch_size, linear_dim, n_patches);
		cudaDeviceSynchronize();

		// Initial
		W_key.apply_derivative();
		W_key.update_weights(effective_lr);
		W_key.compute_error_intermediate();

		W_query.apply_derivative();
		W_query.update_weights(effective_lr);
		W_query.compute_error_intermediate();


		W_value.apply_derivative();
		W_value.update_weights(effective_lr);
		W_value.compute_error_intermediate();
	}
	
	void zero_grad()
	{
		W_query.zero_grad();
		W_key.zero_grad();
		W_value.zero_grad();
		score_matrix.zero_grad();
		acumulator_exp.reset_data();
		attention_output.zero_grad();
		W_out.zero_grad();
	}

	void save_weights(std::ofstream& file)
	{
		W_query.save_weights(file);
		W_key.save_weights(file);
		W_value.save_weights(file);
		W_out.save_weights(file);
	}

	void load_weights(std::ifstream& file)
	{
		W_query.load_weights(file);
		W_key.load_weights(file);
		W_value.load_weights(file);
		W_out.load_weights(file);
	}
private:
	Layer W_query, W_key, W_value;
};

#endif