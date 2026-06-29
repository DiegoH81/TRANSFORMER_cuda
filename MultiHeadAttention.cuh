#pragma once

#include <cassert>
#include <cfloat>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "tensor.cuh"
#include "utils.cuh"

__global__
void mha_linear(const float* __restrict__ in, const float* __restrict__ W, const float* __restrict__ b, float* __restrict__ out, int seq_len, int dim_model)
{
	int img = blockIdx.x;
	int tok = blockIdx.y;

	int in_base = (img * seq_len + tok) * dim_model;
	int out_base = (img * seq_len + tok) * dim_model;

	for (int col = threadIdx.x; col < dim_model; col += blockDim.x) {
		float sum = 0.0f;
		for (int k = 0; k < dim_model; k++)
			sum += in[in_base + k] * W[col * dim_model + k];
		out[out_base + col] = sum + b[col];
	}
}

__global__
void mha_scores(const float* __restrict__ Q, const float* __restrict__ K, float* __restrict__ S, int seq_len, int n_heads, int head_dim, float scale)
{
	int img = blockIdx.x;
	int h = blockIdx.y;
	int i = blockIdx.z;
	int j = threadIdx.x;

	if (j >= seq_len) return;

	int qk_stride = n_heads * seq_len * head_dim;
	int q_base = img * qk_stride + h * seq_len * head_dim + i * head_dim;
	int k_base = img * qk_stride + h * seq_len * head_dim + j * head_dim;

	float dot = 0.0f;
	for (int k = 0; k < head_dim; k++)
		dot += Q[q_base + k] * K[k_base + k];

	int s_stride = n_heads * seq_len * seq_len;
	S[img * s_stride + h * seq_len * seq_len + i * seq_len + j] = dot * scale;
}

__global__
void mha_softmax(float* __restrict__ S, int seq_len, int n_heads)
{
	int img = blockIdx.x;
	int h = blockIdx.y;
	int i = blockIdx.z;

	int base = (img * n_heads + h) * seq_len * seq_len + i * seq_len;

	extern __shared__ float smem[];

	int tid = threadIdx.x;

	float val = -FLT_MAX;
	for (int j = tid; j < seq_len; j += blockDim.x)
		val = fmaxf(val, S[base + j]);
	smem[tid] = val;
	__syncthreads();

	for (int stride = blockDim.x / 2; stride > 0; stride /= 2) {
		if (tid < stride)
			smem[tid] = fmaxf(smem[tid], smem[tid + stride]);
		__syncthreads();
	}
	float mx = smem[0];

	float sum = 0.0f;
	for (int j = tid; j < seq_len; j += blockDim.x) {
		float e = expf(S[base + j] - mx);
		S[base + j] = e;
		sum += e;
	}
	smem[tid] = sum;
	__syncthreads();

	for (int stride = blockDim.x / 2; stride > 0; stride /= 2) {
		if (tid < stride)
			smem[tid] += smem[tid + stride];
		__syncthreads();
	}
	float inv_sum = 1.0f / smem[0];

	for (int j = tid; j < seq_len; j += blockDim.x)
		S[base + j] *= inv_sum;
}

__global__
void mha_attn_out(const float* __restrict__ A, const float* __restrict__ V, float* __restrict__ O, int seq_len, int n_heads, int head_dim)
{
	int img = blockIdx.x;
	int h = blockIdx.y;
	int i = blockIdx.z;
	int k = threadIdx.x;

	if (k >= head_dim) return;

	int a_base = (img * n_heads + h) * seq_len * seq_len + i * seq_len;
	int v_base = (img * n_heads + h) * seq_len * head_dim;
	int o_base = (img * n_heads + h) * seq_len * head_dim + i * head_dim;

	float sum = 0.0f;
	for (int j = 0; j < seq_len; j++)
		sum += A[a_base + j] * V[v_base + j * head_dim + k];

	O[o_base + k] = sum;
}

__global__
void mha_concat(const float* __restrict__ O, float* __restrict__ concat, int seq_len, int n_heads, int head_dim)
{
	int img = blockIdx.x;
	int tok = blockIdx.y;
	int dim_model = n_heads * head_dim;

	for (int d = threadIdx.x; d < dim_model; d += blockDim.x) {
		int h = d / head_dim;
		int k = d % head_dim;

		int src = (img * n_heads + h) * seq_len * head_dim + tok * head_dim + k;
		int dst = (img * seq_len + tok) * dim_model + d;

		concat[dst] = O[src];
	}
}

__global__
void mha_linear_backward_W(const float* __restrict__ d_out, const float* __restrict__ in, float* __restrict__ dW, int seq_len, int dim_model)
{
	int img = blockIdx.x;
	int tok = blockIdx.y;

	int base = (img * seq_len + tok) * dim_model;

	for (int col = threadIdx.x; col < dim_model; col += blockDim.x) {
		float dy = d_out[base + col];
		for (int k = 0; k < dim_model; k++)
			atomicAdd(&dW[col * dim_model + k], dy * in[base + k]);
	}
}

__global__
void mha_bias_backward(const float* __restrict__ d_out, float* __restrict__ db, int seq_len, int dim_model)
{
	int img = blockIdx.x;
	int tok = blockIdx.y;

	int base = (img * seq_len + tok) * dim_model;

	for (int col = threadIdx.x; col < dim_model; col += blockDim.x)
		atomicAdd(&db[col], d_out[base + col]);
}

__global__
void mha_linear_backward_in(const float* __restrict__ d_out, const float* __restrict__ W, float* __restrict__ d_in, int seq_len, int dim_model)
{
	int img = blockIdx.x;
	int tok = blockIdx.y;

	int base = (img * seq_len + tok) * dim_model;

	for (int k = threadIdx.x; k < dim_model; k += blockDim.x) {
		float sum = 0.0f;
		for (int col = 0; col < dim_model; col++)
			sum += d_out[base + col] * W[col * dim_model + k];
		d_in[base + k] += sum;
	}
}

__global__
void mha_backward_AV(const float* __restrict__ dO, const float* __restrict__ A, const float* __restrict__ V, float* __restrict__ dA, float* __restrict__ dV, int seq_len, int n_heads, int head_dim)
{
	int img = blockIdx.x;
	int h = blockIdx.y;
	int i = blockIdx.z;
	int j = threadIdx.x;

	if (j >= seq_len) return;

	int o_base = (img * n_heads + h) * seq_len * head_dim + i * head_dim;
	int v_base = (img * n_heads + h) * seq_len * head_dim + j * head_dim;
	int a_base = (img * n_heads + h) * seq_len * seq_len + i * seq_len;

	float da = 0.0f;
	for (int k = 0; k < head_dim; k++)
		da += dO[o_base + k] * V[v_base + k];
	dA[a_base + j] = da;

	float aij = A[a_base + j];
	for (int k = 0; k < head_dim; k++)
		atomicAdd(&dV[v_base + k], aij * dO[o_base + k]);
}

__global__
void mha_softmax_backward(float* __restrict__ dA, const float* __restrict__ A, int seq_len, int n_heads)
{
	int img = blockIdx.x;
	int h = blockIdx.y;
	int i = blockIdx.z;

	int base = (img * n_heads + h) * seq_len * seq_len + i * seq_len;

	extern __shared__ float smem[];

	int tid = threadIdx.x;

	float dot = 0.0f;
	for (int j = tid; j < seq_len; j += blockDim.x)
		dot += dA[base + j] * A[base + j];
	smem[tid] = dot;
	__syncthreads();

	for (int stride = blockDim.x / 2; stride > 0; stride /= 2) {
		if (tid < stride)
			smem[tid] += smem[tid + stride];
		__syncthreads();
	}
	dot = smem[0];

	for (int j = tid; j < seq_len; j += blockDim.x)
		dA[base + j] = A[base + j] * (dA[base + j] - dot);
}

__global__
void mha_backward_QK(const float* __restrict__ dS, const float* __restrict__ Q, const float* __restrict__ K, float* __restrict__ dQ, float* __restrict__ dK, int seq_len, int n_heads, int head_dim, float scale)
{
	int img = blockIdx.x;
	int h = blockIdx.y;
	int i = blockIdx.z;
	int k = threadIdx.x;

	if (k >= head_dim) return;

	int qk_stride = n_heads * seq_len * head_dim;
	int q_base = img * qk_stride + h * seq_len * head_dim + i * head_dim;
	int ds_base = (img * n_heads + h) * seq_len * seq_len + i * seq_len;

	float dq = 0.0f;
	for (int j = 0; j < seq_len; j++)
		dq += dS[ds_base + j] * K[img * qk_stride + h * seq_len * head_dim + j * head_dim + k];
	dQ[q_base + k] = dq * scale;

	for (int j = 0; j < seq_len; j++)
		atomicAdd(&dK[img * qk_stride + h * seq_len * head_dim + j * head_dim + k], dS[ds_base + j] * Q[q_base + k] * scale);
}

__global__
void mha_split(const float* __restrict__ in, float* __restrict__ out, int seq_len, int n_heads, int head_dim)
{
	int img = blockIdx.x;
	int tok = blockIdx.y;
	int dim_model = n_heads * head_dim;

	for (int d = threadIdx.x; d < dim_model; d += blockDim.x) {
		int h = d / head_dim;
		int k = d % head_dim;

		int src = (img * seq_len + tok) * dim_model + d;
		int dst = (img * n_heads + h) * seq_len * head_dim + tok * head_dim + k;

		out[dst] = in[src];
	}
}

__global__
void mha_unsplit(const float* __restrict__ in, float* __restrict__ out, int seq_len, int n_heads, int head_dim)
{
	int img = blockIdx.x;
	int tok = blockIdx.y;
	int dim_model = n_heads * head_dim;

	for (int d = threadIdx.x; d < dim_model; d += blockDim.x) {
		int h = d / head_dim;
		int k = d % head_dim;

		int src = (img * n_heads + h) * seq_len * head_dim + tok * head_dim + k;
		int dst = (img * seq_len + tok) * dim_model + d;

		out[dst] = in[src];
	}
}

class MultiHeadAttention {
public:
	int n_images, seq_len, dim_model, n_heads, head_dim;

	Tensor Wq, Wk, Wv, Wo;
	Tensor bq, bk, bv, bo;

	Tensor Q_proj, K_proj, V_proj;
	Tensor Q, K, V;
	Tensor scores;
	Tensor attn_out;
	Tensor concat;
	Tensor output;

	Tensor* previous;

	MultiHeadAttention(int in_n_images, int in_seq_len, int in_dim_model, int in_n_heads, Tensor* in_previous = nullptr)
		: n_images(in_n_images), seq_len(in_seq_len), dim_model(in_dim_model), n_heads(in_n_heads), head_dim(in_dim_model / in_n_heads), previous(in_previous)
	{
		assert(dim_model % n_heads == 0 && "dim_model must be divisible by n_heads");

		int W_size = dim_model * dim_model;
		int qkv_tok = n_images * seq_len * dim_model;
		int qkv_hd = n_images * n_heads * seq_len * head_dim;
		int s_size = n_images * n_heads * seq_len * seq_len;

		Wq.set_size(W_size);
		Wq.set_random(sqrtf(2.0f / dim_model));
		Wk.set_size(W_size);
		Wk.set_random(sqrtf(2.0f / dim_model));
		Wv.set_size(W_size);
		Wv.set_random(sqrtf(2.0f / dim_model));
		Wo.set_size(W_size);
		Wo.set_random(sqrtf(2.0f / dim_model));

		bq.set_size(dim_model);
		bk.set_size(dim_model);
		bv.set_size(dim_model);
		bo.set_size(dim_model);

		Q_proj.set_size(qkv_tok);
		K_proj.set_size(qkv_tok);
		V_proj.set_size(qkv_tok);
		Q.set_size(qkv_hd);
		K.set_size(qkv_hd);
		V.set_size(qkv_hd);
		scores.set_size(s_size);
		attn_out.set_size(qkv_hd);
		concat.set_size(qkv_tok);
		output.set_size(qkv_tok);
	}

	void forward()
	{
		dim3 grid_tok(n_images, seq_len);
		constexpr int BLK = 256;

		int smx_threads = 1;
		while (smx_threads < seq_len && smx_threads < 1024) smx_threads *= 2;
		size_t smx_smem = smx_threads * sizeof(float);

		mha_linear << <grid_tok, BLK >> > (previous->data, Wq.data, bq.data, Q_proj.data, seq_len, dim_model);
		mha_linear << <grid_tok, BLK >> > (previous->data, Wk.data, bk.data, K_proj.data, seq_len, dim_model);
		mha_linear << <grid_tok, BLK >> > (previous->data, Wv.data, bv.data, V_proj.data, seq_len, dim_model);
		cudaDeviceSynchronize();

		mha_split << <grid_tok, BLK >> > (Q_proj.data, Q.data, seq_len, n_heads, head_dim);
		mha_split << <grid_tok, BLK >> > (K_proj.data, K.data, seq_len, n_heads, head_dim);
		mha_split << <grid_tok, BLK >> > (V_proj.data, V.data, seq_len, n_heads, head_dim);
		cudaDeviceSynchronize();

		float scale = 1.0f / sqrtf((float)head_dim);
		dim3 grid_scores(n_images, n_heads, seq_len);
		mha_scores << <grid_scores, seq_len >> > (Q.data, K.data, scores.data, seq_len, n_heads, head_dim, scale);
		cudaDeviceSynchronize();

		mha_softmax << <grid_scores, smx_threads, smx_smem >> > (scores.data, seq_len, n_heads);
		cudaDeviceSynchronize();

		mha_attn_out << <grid_scores, head_dim >> > (scores.data, V.data, attn_out.data, seq_len, n_heads, head_dim);
		cudaDeviceSynchronize();

		mha_concat << <grid_tok, BLK >> > (attn_out.data, concat.data, seq_len, n_heads, head_dim);
		cudaDeviceSynchronize();

		mha_linear << <grid_tok, BLK >> > (concat.data, Wo.data, bo.data, output.data, seq_len, dim_model);
		cudaDeviceSynchronize();
	}

	void backward()
	{
		dim3 grid_tok(n_images, seq_len);
		dim3 grid_scores(n_images, n_heads, seq_len);
		float scale = 1.0f / sqrtf((float)head_dim);
		constexpr int BLK = 256;

		int smx_threads = 1;
		while (smx_threads < seq_len && smx_threads < 1024) smx_threads *= 2;
		size_t smx_smem = smx_threads * sizeof(float);

		mha_linear_backward_W << <grid_tok, BLK >> > (output.gradient, concat.data, Wo.gradient, seq_len, dim_model);
		mha_bias_backward << <grid_tok, BLK >> > (output.gradient, bo.gradient, seq_len, dim_model);
		mha_linear_backward_in << <grid_tok, BLK >> > (output.gradient, Wo.data, concat.gradient, seq_len, dim_model);
		cudaDeviceSynchronize();

		mha_unsplit << <grid_tok, BLK >> > (concat.gradient, attn_out.gradient, seq_len, n_heads, head_dim);
		cudaDeviceSynchronize();

		mha_backward_AV << <grid_scores, seq_len >> > (attn_out.gradient, scores.data, V.data, scores.gradient, V.gradient, seq_len, n_heads, head_dim);
		cudaDeviceSynchronize();

		mha_softmax_backward << <grid_scores, smx_threads, smx_smem >> > (scores.gradient, scores.data, seq_len, n_heads);
		cudaDeviceSynchronize();

		mha_backward_QK << <grid_scores, head_dim >> > (scores.gradient, Q.data, K.data, Q.gradient, K.gradient, seq_len, n_heads, head_dim, scale);
		cudaDeviceSynchronize();

		mha_unsplit << <grid_tok, BLK >> > (Q.gradient, Q_proj.gradient, seq_len, n_heads, head_dim);
		mha_unsplit << <grid_tok, BLK >> > (K.gradient, K_proj.gradient, seq_len, n_heads, head_dim);
		mha_unsplit << <grid_tok, BLK >> > (V.gradient, V_proj.gradient, seq_len, n_heads, head_dim);
		cudaDeviceSynchronize();

		mha_linear_backward_W << <grid_tok, BLK >> > (Q_proj.gradient, previous->data, Wq.gradient, seq_len, dim_model);
		mha_linear_backward_W << <grid_tok, BLK >> > (K_proj.gradient, previous->data, Wk.gradient, seq_len, dim_model);
		mha_linear_backward_W << <grid_tok, BLK >> > (V_proj.gradient, previous->data, Wv.gradient, seq_len, dim_model);

		mha_bias_backward << <grid_tok, BLK >> > (Q_proj.gradient, bq.gradient, seq_len, dim_model);
		mha_bias_backward << <grid_tok, BLK >> > (K_proj.gradient, bk.gradient, seq_len, dim_model);
		mha_bias_backward << <grid_tok, BLK >> > (V_proj.gradient, bv.gradient, seq_len, dim_model);

		mha_linear_backward_in << <grid_tok, BLK >> > (Q_proj.gradient, Wq.data, previous->gradient, seq_len, dim_model);
		mha_linear_backward_in << <grid_tok, BLK >> > (K_proj.gradient, Wk.data, previous->gradient, seq_len, dim_model);
		mha_linear_backward_in << <grid_tok, BLK >> > (V_proj.gradient, Wv.data, previous->gradient, seq_len, dim_model);
		cudaDeviceSynchronize();
	}

	void update_weights(float lr)
	{
		int W_size = dim_model * dim_model;
		int threads = 256;
		int blocks = (W_size + threads - 1) / threads;

		float effective_lr = lr / (float)(n_images * seq_len);

		sgd_update << <blocks, threads >> > (Wq.data, Wq.gradient, effective_lr, W_size);
		sgd_update << <blocks, threads >> > (Wk.data, Wk.gradient, effective_lr, W_size);
		sgd_update << <blocks, threads >> > (Wv.data, Wv.gradient, effective_lr, W_size);
		sgd_update << <blocks, threads >> > (Wo.data, Wo.gradient, effective_lr, W_size);

		int b_blocks = (dim_model + threads - 1) / threads;
		float effective_lr_b = lr / (float)(n_images * seq_len);
		sgd_update << <b_blocks, threads >> > (bq.data, bq.gradient, effective_lr_b, dim_model);
		sgd_update << <b_blocks, threads >> > (bk.data, bk.gradient, effective_lr_b, dim_model);
		sgd_update << <b_blocks, threads >> > (bv.data, bv.gradient, effective_lr_b, dim_model);
		sgd_update << <b_blocks, threads >> > (bo.data, bo.gradient, effective_lr_b, dim_model);
		cudaDeviceSynchronize();
	}

	void zero_grad()
	{
		Wq.zero_grad();
		Wk.zero_grad();
		Wv.zero_grad();
		Wo.zero_grad();
		bq.zero_grad();
		bk.zero_grad();
		bv.zero_grad();
		bo.zero_grad();
		Q_proj.zero_grad();
		K_proj.zero_grad();
		V_proj.zero_grad();
		Q.zero_grad();
		K.zero_grad();
		V.zero_grad();
		scores.zero_grad();
		attn_out.zero_grad();
		concat.zero_grad();
		output.zero_grad();
	}
};