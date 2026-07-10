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
void exp_matrix(float* score_matrix,
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




	score_matrix[mat_idx] = exp(score_matrix[mat_idx]);
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

class MHA
{
public:
	Tensor * previous;
	size_t batch_size, n_patches, linear_dim, num_heads, head_dim;

	MHA(size_t in_n_patches, size_t in_linear_dim, Tensor* in_previous, size_t in_batch_size = 1) :
		batch_size(in_batch_size), n_patches(in_n_patches), linear_dim(in_linear_dim),
		previous(in_previous),
		num_heads(8), head_dim(linear_dim / num_heads),
		W_query(linear_dim, linear_dim, batch_size * n_patches, ActivationType::None, previous),
		W_key(linear_dim, linear_dim, batch_size* n_patches, ActivationType::None, previous),
		W_value(linear_dim, linear_dim, batch_size * n_patches, ActivationType::None, previous)
	{
		score_matrix.set_size(batch_size * num_heads * n_patches * n_patches);
		acumulator_exp.set_size(batch_size * num_heads * n_patches);
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

		exp_matrix << < blocks, threads >> > (score_matrix.data, acumulator_exp.data,
											  num_heads, head_dim, batch_size, linear_dim, n_patches);
		cudaDeviceSynchronize();
		
		div_matrix << < blocks, threads >> > (score_matrix.data, acumulator_exp.data,
											  num_heads, head_dim, batch_size, linear_dim, n_patches);
		cudaDeviceSynchronize();
	}
	
private:
	Layer W_query, W_key, W_value;
	Tensor score_matrix, acumulator_exp;

};

#endif